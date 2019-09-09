import sls
import torch
import torchvision
import tqdm
import pandas as pd
import pprint 
import itertools
import os
import pylab as plt

from src import models
from src import datasets
from src import optimizers
from src import utils as ut

import argparse

from torch.nn import functional as F
from torch.utils.data import DataLoader


def trainval(exp_dict):
    pprint.pprint(exp_dict)

    # Load Dataset
    train_set = datasets.get_dataset(dataset_name=exp_dict["dataset"], 
                                     datadir=".data/")
    train_loader = DataLoader(train_set, drop_last=True, shuffle=True, batch_size=128)

    # Load model
    model = models.get_model(exp_dict["model"]).cuda()

    # Load Optimizer
    opt = optimizers.get_optimizer(exp_dict["opt"], model.parameters())

    if exp_dict["opt"] in ["sgd_armijo"]:
        requires_closure = True
    else:
        requires_closure = False

    score_list = []
    for epoch in range(exp_dict["max_epoch"]):
        # =================================
        # 1. Compute metrics over train loader
        model.eval()
        print("Evaluating Epoch %d" % epoch)

        loss_sum = 0.
        for images, labels in tqdm.tqdm(train_loader):
            images, labels = images.cuda(), labels.cuda()

            with torch.no_grad():
                loss_sum += compute_loss(model, images, labels)

        train_loss = float(loss_sum / len(train_set))
        score_list += [{"train_loss":train_loss, "epoch":epoch}]

        # =================================
        # 2. Train over train loader
        model.train()
        print("Training Epoch %d" % epoch)

        for images,labels in tqdm.tqdm(train_loader):
            images, labels = images.cuda(), labels.cuda()

            opt.zero_grad()

            if requires_closure:
                closure = lambda : compute_loss(model, images, labels)
                opt.step(closure)
            else:
                loss = compute_loss(model, images, labels)
                loss.backward()
                opt.step()

        print(pd.DataFrame(score_list))
    
    return score_list

def compute_loss(model, images, labels):
    probs = F.log_softmax(model(images), dim=1)
    loss = F.nll_loss(probs, labels, reduction="sum")

    return loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-o", "--opts", nargs="+",  default=["sgd_armijo", "adam"])
    parser.add_argument("-d", "--dataset", default="mnist")
    parser.add_argument("-m", "--model", default="mlp")
    
    parser.add_argument("-e", "--max_epoch",  default=3, type=int)
    parser.add_argument("-s", "--savedir_base",  default=".data/")
    parser.add_argument("-p", "--plot",  default=0, type=int)
    parser.add_argument("-r", "--reset",  default=0, type=int)
    args = parser.parse_args()

    # loop over optimizers
    for opt in args.opts:
        # get experiment id
        exp_dict = {"dataset": args.dataset, 
                    "model": args.model, 
                    "opt": opt,
                    "max_epoch":args.max_epoch}
        exp_id = ut.hash_dict(exp_dict)
        savedir = args.savedir_base + "/%s/" % exp_id

         # check if experiment exists
        if not args.reset and os.path.exists(savedir + "/score_list.pkl"):
            # load score list
            score_list = ut.load_pkl(savedir + "/score_list.pkl")
        else:
            # do trainval
            score_list = trainval(exp_dict=exp_dict)
        
            # save files
            os.makedirs(savedir, exist_ok=True)

            ut.save_json(savedir + "/exp_dict.json", exp_dict)
            ut.save_pkl(savedir + "/score_list.pkl", score_list)
            print("Saved in %s" % savedir)

        # plot
        if args.plot:
            score_df = pd.DataFrame(score_list)
            plt.plot(score_df["epoch"], score_df["train_loss"], label=opt)

    # save plot figure
    if args.plot:
        plt.legend()
        plt.xlabel("Epochs")
        plt.ylabel("Train Loss")
        plt.title("%s_%s" % (args.dataset, args.model))
        plt.savefig(args.savedir_base + "/plot_%s_%s.jpg" % (args.dataset, args.model))
