import sls
import torch
import torchvision
import tqdm
import pandas as pd
import pprint
import itertools
import os
import pylab as plt
import exp_configs
import time
import math

from src import models
from src import datasets
from src import utils as ut
from src import metrics

from others import pls

import argparse

from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler

# TODO: (amishkin) can the PLS state be saved and loaded? Something might go wrong with
# the underlying Gaussian process.

def trainval_pls(exp_dict, savedir, datadir, metrics_flag=True):
	'''
		PLS-specific training and validation loop.
	'''
	pprint.pprint(exp_dict)

	# Load Train Dataset
	train_set = datasets.get_dataset(dataset_name=exp_dict["dataset"],
									 train_flag=True,
									 datadir=datadir,
									 exp_dict=exp_dict)

	train_loader = DataLoader(train_set,
							  drop_last=False,
							  shuffle=True,
							  batch_size=exp_dict["batch_size"])

	# Load Val Dataset
	val_set = datasets.get_dataset(dataset_name=exp_dict["dataset"],
								   train_flag=False,
								   datadir=datadir,
								   exp_dict=exp_dict)

	# Load model
	model = models.get_model(exp_dict["model"],
							 train_set=train_set).cuda()

	# Choose loss and metric function
	if exp_dict["loss_func"] == 'logistic_loss':
		loss_function = logistic_loss_grad_moments
	else:
		raise ValueError("PLS only supports the logistic loss.")

	# Load Optimizer
	opt = pls.PLS(model, exp_dict["max_epoch"], exp_dict["batch_size"], expl_policy='exponential')

	# Resume from last saved state_dict
	if (not os.path.exists(savedir + "/run_dict.pkl") or
		not os.path.exists(savedir + "/score_list.pkl")):
		ut.save_pkl(savedir + "/run_dict.pkl", {"running":1})
		score_list = []
		s_epoch = 0
	else:
		score_list = ut.load_pkl(savedir + "/score_list.pkl")
		model.load_state_dict(torch.load(savedir + "/model_state_dict.pth"))
		opt.load_state_dict(torch.load(savedir + "/opt_state_dict.pth"))
		s_epoch = score_list[-1]["epoch"] + 1

	# PLS-specific tracking for iterations and epochs:
	epoch = s_epoch
	iter_num = 0
	iters_per_epoch = math.ceil(len(train_loader.dataset) / exp_dict['batch_size'])
	new_epoch = True

	while epoch < exp_dict["max_epoch"]:
		for images,labels in tqdm.tqdm(train_loader):
			# record metrics at the start of a new epoch
			if metrics_flag and new_epoch:
				new_epoch = False
				score_dict = {"epoch": epoch}

				# 1. Compute train loss over train set
				score_dict["train_loss"] = metrics.compute_metric_on_dataset(model, train_set,
													metric_name=exp_dict["loss_func"])

				# 2. Compute val acc over val set
				score_dict["val_acc"] = metrics.compute_metric_on_dataset(model, val_set,
															metric_name=exp_dict["acc_func"])

				# 3. Train over train loader
				model.train()
				print("%d - Training model with %s..." % (epoch, exp_dict["loss_func"]))

				s_time = time.time()

			images, labels = images.cuda(), labels.cuda()
			closure = grad_moment_closure_factory(model, images, labels, loss_function)

			# For PLS, calls to optimizer.step() do not correspond to a single optimizer step.
			# Instead, they correspond to one evaluation in the line-search, which may or many
			# not be accepted.
			opt.step(closure)

			# Epoch and iteration tracking.
			if opt.state['complete']:
				iter_num = iter_num + 1

				# potentially increment the epoch counter.
				if iter_num % iters_per_epoch == 0:
					epoch = epoch + 1
					new_epoch = True

			# compute metrics at end of previous epoch
			if new_epoch:
				e_time = time.time()

				# Record step size and batch size
				score_dict["step_size"] = opt.state["step_size"]
				score_dict["batch_size"] =  train_loader.batch_size
				score_dict["train_epoch_time"] = e_time - s_time

				# Add score_dict to score_list
				score_list += [score_dict]

				# Report and save
				print(pd.DataFrame(score_list).tail())
				ut.save_pkl(savedir + "/score_list.pkl", score_list)
				ut.torch_save(savedir + "/model_state_dict.pth", model.state_dict())
				ut.torch_save(savedir + "/opt_state_dict.pth", opt.state_dict())
				print("Saved: %s" % savedir)

	return score_list

#################
### Utilities ###
#################


# Closure for computing the gradient and moments:

def grad_moment_closure_factory(model, X, y, loss_function):

	def closure():
		# zero the gradients on the prediction model:
		model.zero_grad()

		logits = model.forward(X)

		losses, gradient_moments = loss_function(model, logits, X, y)
		full_loss = torch.mean(losses)
		full_loss.backward()

		return losses, gradient_moments

	return closure

def logistic_loss_grad_moments(model, logits, x, y):
	'''
		Compute and return the logistic loss and the second moment of the gradient.
		This function works *only* works for linear models.
	'''
	n,d = x.size()
	# convert targets into {-1,1}
	y = (2 * y) - 1
	# compute individual losses
	logits = -torch.mul(y, logits.squeeze())
	losses = F.softplus(logits)

	grad_moments = []
	from_index = 0
	params = list(model.parameters())

	if not isinstance(params, dict):
		params = [{'params': params}]

	for i, group in enumerate(params):
		grad_moment_group = []
		for j, p in enumerate(group['params']):
			to_index = from_index + p.numel()
			# compute the individual gradients
			if to_index <= d:
				x_vals = x[:,from_index:to_index]
			else: # bias
				x_vals = torch.ones_like(y).unsqueeze(dim=1)

			p_ind_grads = x_vals.mul(-y.unsqueeze(1)).mul(F.sigmoid(logits.unsqueeze(1)))

			# check that the gradient is correct
			# full_grad = torch.sum(p_ind_grads, dim=0)  / n
			p_grad_moment = torch.sum(p_ind_grads**2, dim=0) / n
			# save the gradients
			grad_moment_group.append(p_grad_moment)
			# update the index of x.
			from_index = from_index + to_index
		grad_moments.append(grad_moment_group)

	return losses, grad_moments
