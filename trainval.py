import os
import argparse
import torchvision
import pandas as pd
import torch 
import numpy as np
import time
import pprint
import tqdm
import exp_configs
from src import datasets, models, optimizers, metrics

from haven import haven_utils as hu
from haven import haven_results as hr
from haven import haven_chk as hc
from haven import haven_jupyter as hj


def trainval(exp_dict, savedir_base, datadir, reset=False, metrics_flag=True):
    # bookkeeping
    # ---------------

    # get experiment directory
    exp_id = hu.hash_dict(exp_dict)
    savedir = os.path.join(savedir_base, exp_id)

    if reset:
        # delete and backup experiment
        hc.delete_experiment(savedir, backup_flag=True)
    
    # create folder and save the experiment dictionary
    os.makedirs(savedir, exist_ok=True)
    hu.save_json(os.path.join(savedir, 'exp_dict.json'), exp_dict)
    pprint.pprint(exp_dict)
    print('Experiment saved in %s' % savedir)
    
    # set seed
    # ---------------
    seed = 42 + exp_dict['runs']
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Dataset
    # -----------

    # Load Train Dataset
    train_set = datasets.get_dataset(dataset_name=exp_dict["dataset"],
                                     train_flag=True,
                                     datadir=datadir,
                                     exp_dict=exp_dict)

    train_loader = torch.utils.data.DataLoader(train_set,
                              drop_last=True,
                              shuffle=True,
                              batch_size=exp_dict["batch_size"])

    # Load Val Dataset
    val_set = datasets.get_dataset(dataset_name=exp_dict["dataset"],
                                   train_flag=False,
                                   datadir=datadir,
                                   exp_dict=exp_dict)


    # Model
    # -----------
    model = models.get_model(exp_dict["model"],
                             train_set=train_set).cuda()
    # Choose loss and metric function
    loss_function = metrics.get_metric_function(exp_dict["loss_func"])

    # Load Optimizer
    n_batches_per_epoch = len(train_set)/float(exp_dict["batch_size"])
    opt = optimizers.get_optimizer(opt=exp_dict["opt"],
                                   params=model.parameters(),
                                   n_batches_per_epoch =n_batches_per_epoch)

    # Checkpoint
    # -----------
    model_path = os.path.join(savedir, 'model.pth')
    score_list_path = os.path.join(savedir, 'score_list.pkl')
    opt_path = os.path.join(savedir, 'opt_state_dict.pth')

    if os.path.exists(score_list_path):
        # resume experiment
        score_list = hu.load_pkl(score_list_path)
        model.load_state_dict(torch.load(model_path))
        opt.load_state_dict(torch.load(opt_path))
        s_epoch = score_list[-1]['epoch'] + 1
    else:
        # restart experiment
        score_list = []
        s_epoch = 0

    # Train & Val
    # ------------
    print('Starting experiment at epoch %d/%d' % (s_epoch, exp_dict['max_epoch']))

    for epoch in range(s_epoch, exp_dict['max_epoch']):
        # Set seed
        np.random.seed(exp_dict['runs']+epoch)
        torch.manual_seed(exp_dict['runs']+epoch)
        torch.cuda.manual_seed_all(exp_dict['runs']+epoch)

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

        # Record metrics
        score_dict["step_size"] = opt.state["step_size"]
        score_dict["n_forwards"] = opt.state["n_forwards"]
        score_dict["n_backwards"] = opt.state["n_backwards"]
        score_dict["batch_size"] =  train_loader.batch_size
        score_dict["train_epoch_time"] = e_time - s_time

        score_list += [score_dict]

        # Report and save
        print(pd.DataFrame(score_list).tail())
        hu.save_pkl(score_list_path, score_list)
        hu.torch_save(model_path, model.state_dict())
        hu.torch_save(opt_path, opt.state_dict())
        print("Saved: %s" % savedir)

    print('Experiment completed')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--exp_group_list', nargs='+')
    parser.add_argument('-sb', '--savedir_base', required=True)
    parser.add_argument('-d', '--datadir', required=True)
    parser.add_argument('-r', '--reset',  default=0, type=int)
    parser.add_argument('-ei', '--exp_id', default=None)

    args = parser.parse_args()

    # Collect experiments
    # -------------------
    if args.exp_id is not None:
        # select one experiment
        savedir = os.path.join(args.savedir_base, args.exp_id)
        exp_dict = hu.load_json(os.path.join(savedir, 'exp_dict.json'))        
        
        exp_list = [exp_dict]
        
    else:
        # select exp group
        exp_list = []
        for exp_group_name in args.exp_group_list:
            exp_list += exp_configs.EXP_GROUPS[exp_group_name]


    # Run experiments
    # ----------------------------
    for exp_dict in exp_list:
        # do trainval
        trainval(exp_dict=exp_dict,
                savedir_base=args.savedir_base,
                datadir=args.datadir,
                reset=args.reset)