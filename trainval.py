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
import numpy as np

from src import models
from src import datasets
from src import optimizers
from src import utils as ut
from src import metrics

import argparse

from torch.backends import cudnn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler

cudnn.benchmark = True


def trainval(exp_dict, savedir, datadir, metrics_flag=True):
    # Set seed
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    pprint.pprint(exp_dict)

    # Load Train Dataset
    train_set = datasets.get_dataset(dataset_name=exp_dict["dataset"],
                                     train_flag=True,
                                     datadir=datadir,
                                     exp_dict=exp_dict)

    train_loader = DataLoader(train_set,
                              drop_last=True,
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

    # Load Optimizer
    n_batches_per_epoch = len(train_set)/float(exp_dict["batch_size"])
    opt = optimizers.get_optimizer(opt=exp_dict["opt"],
                                   params=model.parameters(),
                                   n_batches_per_epoch =n_batches_per_epoch)

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
        # Set seed
        np.random.seed(epoch)
        torch.manual_seed(epoch)
        torch.cuda.manual_seed_all(epoch)

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

            if exp_dict["opt"]["name"] in exp_configs.ours_opt_list + ["l4"]:
                closure = lambda : loss_function(model, images, labels, backwards=False)
                opt.step(closure)

            else:
                loss = loss_function(model, images, labels)
                loss.backward()
                opt.step()

        e_time = time.time()

        # Record step size and batch size
        score_dict["step_size"] = opt.state["step_size"]
        score_dict["n_forwards"] = opt.state["n_forwards"]
        score_dict["n_backwards"] = opt.state["n_backwards"]
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--exp_group_list', nargs="+")
    parser.add_argument('-sb', '--savedir_base', required=True)
    parser.add_argument('-d', '--datadir', required=True)
    parser.add_argument("-r", "--reset",  default=0, type=int)
    parser.add_argument("-ei", "--exp_id", default=None)
    parser.add_argument("-mf", "--metrics_flag", default=1, type=int)

    args = parser.parse_args()
    exp_list = []
    for exp_group_name in args.exp_group_list:
        exp_list += exp_configs.EXP_GROUPS[exp_group_name]

    # loop over experiments
    for exp_dict in exp_list:
        exp_id = ut.hash_dict(exp_dict)

        if args.exp_id is not None and args.exp_id != exp_id:
            continue

        savedir = args.savedir_base + "/%s/" % exp_id
        os.makedirs(savedir, exist_ok=True)
        ut.save_json(savedir+"/exp_dict.json", exp_dict)

        # check if experiment exists
        if args.reset:
            if os.path.exists(savedir + "/score_list.pkl"):
                os.remove(savedir + "/score_list.pkl")
            if os.path.exists(savedir + "/run_dict.pkl"):
                os.remove(savedir + "/run_dict.pkl")

        # do trainval
        trainval(exp_dict=exp_dict, savedir=savedir, datadir=args.datadir,
                    metrics_flag=args.metrics_flag)