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
PATH_BASE = "/mnt/datasets/public/issam/prototypes/sls/borgy/results/"


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
    exp_id = hash_dict(exp_dict)
    save_yaml("%s/%s/exp_dict.yaml" % (PATH_BASE, exp_id), exp_dict)

    # Get train loader
    train_set = datasets.MainDataset(
        split="train", dataset=exp_dict["dataset"])
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=exp_dict["batch_size"],
        num_workers=0,
        pin_memory=True,
        shuffle=True,
        drop_last=True)

    # Get val loader
    val_set = datasets.MainDataset(split="val", dataset=exp_dict["dataset"])

    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=500, num_workers=0, pin_memory=True, shuffle=False)

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
        save_pkl("%s/%s/history.pkl" % (PATH_BASE, exp_id), history)


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


# ===============================
# Model 
import torch
import torch.nn as nn
import torch.nn.functional as F


class MainModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()

        # Define Model
        if model_name == "resnet34":
            self.model = ResNet34()

        self.error_fn = softmax_error
        self.objective = softmax_loss

    def forward(self, x):
        return self.model(x)

    def get_grads(self):
        return get_grads(list(self.parameters()))

    def step(self, batch):
        self.train()

        Ai = batch["images"].cuda()
        yi = batch["targets"].cuda()

        def closure(compute_grad=True):
            if compute_grad:
                self.zero_grad()
                
            logits = self.forward(Ai)
            loss = self.objective(logits, yi)

            if compute_grad:
                loss.backward()

            return loss

        minibatch_loss = self.opt.step(closure)
        return float(minibatch_loss)

    @torch.no_grad()
    def predict(self, batch, **options):
        self.eval()
        Ai = batch["images"].cuda()
        logits = self(Ai)

        return logits


# =========================================
# ResNet34
def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


import torch
import hashlib
import yaml
import pickle, os
import torch.nn.functional as F


# =============================================
# loss and error function
def softmax_loss(logits, y):
    loglik = F.cross_entropy(logits, y)
    return loglik


def softmax_error(logits, y):
    _, pred_class = torch.max(logits, 1)
    wrong = (pred_class != y)
    err = wrong.float().mean()
    return err


def get_grads(param_groups):
    grad_norm = 0
    gradient = []

    if not isinstance(param_groups[0], dict):
        param_groups = [{'params': param_groups}]

    for i, group in enumerate(param_groups):
        grad_group = []
        for j, p in enumerate(group['params']):
            grad_copy = torch.zeros_like(p.grad.data).copy_(p.grad.data)
            grad_group.append(grad_copy)
            grad_norm = grad_norm + torch.sum(torch.mul(grad_copy, grad_copy))

        gradient.append(grad_group)

    return gradient, torch.sqrt(grad_norm)


# ===================================
# misc
def hash_dict(dictionary):
    dict2hash = ""
    for k in sorted(dictionary.keys()):
        if k == "mode":
            continue

        if isinstance(dictionary[k], dict):
            v = hash_dict(dictionary[k])
        else:
            v = dictionary[k]

        dict2hash += "%s_%s_" % (str(k), str(v))
        
    return hashlib.md5(dict2hash.encode()).hexdigest()


def create_dirs(fname):
    if "/" not in fname:
        return

    if not os.path.exists(os.path.dirname(fname)):
        os.makedirs(os.path.dirname(fname))


def save_yaml(fname, data, arrays2list=False):
    create_dirs(fname)

    with open(fname, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)
    print("SAVED: %s" % fname)


def load_yaml(fname):
    with open(fname, 'r') as outfile:
        yaml_fuile = yaml.load(outfile, Loader=yaml.FullLoader)
    return yaml_fuile


def save_pkl(fname, dict):
    create_dirs(fname)
    with open(fname, "wb") as f:
        pickle.dump(dict, f)
    print("SAVED: %s" % fname)


def load_pkl(fname):
    with open(fname, "rb") as f:
        return pickle.load(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', '--opt', default="SGD_Armijo")
    args = parser.parse_args()

    main(args)

