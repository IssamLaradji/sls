import tqdm
import datasets, models
import torch
from pprint import pprint
from torch.backends import cudnn
import utils as ut
import sgd_armijo
import numpy as np
import time
import argparse
import pandas as pd

cudnn.benchmark = True
PATH_BASE = "results/"


def main(args):
    # Create experiment dictionary and save
    exp_dict = {
        "dataset": "CIFAR10",
        "model": "resnet34",
        "opt": args.opt,
        "batch_size": 128,
        "epochs": 200
    }

    pprint(exp_dict)
    exp_id = ut.hash_dict(exp_dict)
    ut.save_yaml("%s/%s/exp_dict.yaml" % (PATH_BASE, exp_id), exp_dict)

    # Get train loader
    train_set = datasets.MainDataset(
        split="train", dataset=exp_dict["dataset"])
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=exp_dict["batch_size"],
        num_workers=1,
        pin_memory=True,
        shuffle=True,
        drop_last=True)

    # Get val loader
    val_set = datasets.MainDataset(split="val", dataset=exp_dict["dataset"])

    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=500, num_workers=1, pin_memory=True, shuffle=False)

    # Get model and optimizer
    model = models.MainModel(exp_dict["model"]).cuda()

    if exp_dict["opt"] == "Adam":
        model.opt = torch.optim.Adam(model.parameters())

    elif exp_dict["opt"] == "SGD_Armijo":
        model.opt = sgd_armijo.SGD_Armijo(
            model,
            batch_size=exp_dict["batch_size"],
            dataset_size=len(train_set.dataset))
    else:
        raise ValueError("opt %s does not exist..." % exp_dict["opt"])

    history = {"iters": 0, "scoreList": []}
    for e in range(exp_dict["epochs"]):
        # Set seed
        np.random.seed(history["iters"])
        torch.manual_seed(history["iters"])
        torch.cuda.manual_seed_all(history["iters"])

        # ---------------------------------------------
        # Compute metrics
        model.eval()

        score_dict = {}

        # test metrics
        print("Epoch:%d - Computing Test Acc" % len(history["scoreList"]))
        test_error = compute_val_metric(model, val_loader)
        score_dict["test_acc"] = (1 - test_error) * 100

        # train metrics
        print("Epoch:%d - Computing Train Loss & Acc" % len(
            history["scoreList"]))
        train_loss, train_error = (compute_train_metric(model, train_loader))

        score_dict["train_loss"] = train_loss
        score_dict["train_acc"] = (1 - train_error) * 100

        # ---------------------------------------------
        # Train for one epoch
        model.train()
        print("Training epoch %d" % len(history["scoreList"]))
        score_dict.update(fitEpoch(model, train_loader, verbose=1))
        score_dict["history_iters"] = history["iters"]

        # Update history
        history["scoreList"] += [score_dict]
        history["iters"] += score_dict["iters"]
        history["scoreList"] = history["scoreList"]

        # Report & Save
        print("\n", pd.DataFrame(history["scoreList"]).tail(), "\n")
        ut.save_pkl("%s/%s/history.pkl" % (PATH_BASE, exp_id), history)


# ====================================================
# Helpers
def fitEpoch(model, train_loader, verbose=1):
    model.train()
    n_batches = len(train_loader)

    # start training
    loss_sum = 0.
    s_time = time.time()
    pbar = tqdm.tqdm(total=n_batches)
    for i, batch in enumerate(train_loader):
        loss = model.step(batch)
        if np.isnan(loss):
            raise ValueError('loss=nan ...')

        loss_sum += loss
        pbar.set_description("Loss: %.2f" % (loss_sum / (i + 1)))
        pbar.update(1)
    pbar.close()

    score_dict = {
        "loss": loss_sum / n_batches,
        "s_time": s_time,
        "e_time": time.time(),
        "iters": n_batches,
        "batch_size": train_loader.batch_size,
        "n_train": len(train_loader.dataset)
    }

    return score_dict


@torch.no_grad()
def compute_train_metric(model, loader):
    model.eval()

    error = 0.
    loss = 0.
    n = 0

    pbar = tqdm.tqdm(total=len(loader))
    for _, batch in enumerate(loader):
        loss_i, error_i = compute_loss_error(model, batch)

        n += batch["images"].shape[0]
        loss += loss_i
        error += error_i
        pbar.update(1)

    pbar.close()

    return float(loss / n), float(error / n)


@torch.no_grad()
def compute_val_metric(model, loader):
    model.eval()

    error = 0.
    n = 0

    pbar = tqdm.tqdm(total=len(loader))
    for _, batch in enumerate(loader):
        error_i = compute_error(model, batch)
        error += error_i
        n += batch["images"].shape[0]

        pbar.update(1)

    pbar.close()

    return float(error / n)


def compute_error(model, batch):
    Ai = batch["images"].cuda()
    yi = batch["targets"].cuda()
    ni = batch["images"].shape[0]

    logits = model.forward(Ai)
    error = model.error_fn(logits, yi) * ni

    return error


def compute_loss_error(model, batch):
    Ai = batch["images"].cuda()
    yi = batch["targets"].cuda()
    ni = batch["images"].shape[0]

    logits = model.forward(Ai)

    loss = model.objective(logits, yi) * ni
    error = model.error_fn(logits, yi) * ni

    return loss, error


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', '--opt', default="Adam")
    args = parser.parse_args()

    main(args)
