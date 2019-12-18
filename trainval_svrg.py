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

from src import models
from src import datasets
from src import utils as ut
from src import metrics

from others import svrg

import argparse

from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler

def trainval_svrg(exp_dict, savedir, datadir, metrics_flag=True):
    '''
        SVRG-specific training and validation loop.
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
    loss_function = metrics.get_metric_function(exp_dict["loss_func"])

    # lookup the learning rate
    lr = get_svrg_step_size(exp_dict)

    # Load Optimizer
    opt = get_svrg_optimizer(model, loss_function, train_loader=train_loader, lr=lr)

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

    for epoch in range(s_epoch, exp_dict["max_epoch"]):
        score_dict = {"epoch": epoch}

        if metrics_flag:
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
        for images,labels in tqdm.tqdm(train_loader):
            images, labels = images.cuda(), labels.cuda()

            opt.zero_grad()
            closure = lambda svrg_model : loss_function(svrg_model, images, labels,
                                                                    backwards=True)
            opt.step(closure)

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


def get_svrg_optimizer(model, loss_function, train_loader, lr):
    n = len(train_loader.dataset)
    full_grad_closure = svrg.full_loss_closure_factory(train_loader,
                                                       loss_function,
                                                       grad=True)
    opt = svrg.SVRG(model,
                    train_loader.batch_size,
                    lr,
                    n,
                    full_grad_closure,
                    m=len(train_loader))

    return opt

# learning rates selected by cross-validation.
lr_dict = {
    "logistic_loss": {          "rcv1": 500,
                                "mushrooms": 500,
                                "ijcnn": 500,
                                "w8a": 0.0025 },
}

def get_svrg_step_size(exp_dict):
    if exp_dict["loss_func"] in lr_dict:
        lr = lr_dict[exp_dict["loss_func"]][exp_dict["dataset"]]
    else:
        lr = 0.1

    return lr
