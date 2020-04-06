import os
import itertools

from haven import haven_utils as hu

ours_opt_list =  [
        "sgd_armijo",
         "seg",
           "sgd_goldstein",
           "sgd_polyak",
            ]

others_opt_list  = ["coin", "adabound", "adam"]
opt_list = ours_opt_list + others_opt_list
kernel_opt_list = opt_list + ["svrg", "pls"]
kernel_datasets = ["mushrooms", "ijcnn", "rcv1", "w8a"]

opt_list_kernel_mf =[{"name":"sgd_armijo", "gamma":1.5},
                     {"name":"seg", "gamma":1.5},
                     {"name":"sgd_goldstein", "gamma":1.5},
                     {"name":"sgd_polyak", "c":0.5, "momentum":0.9, "gamma":1.5},]

opt_list_deep =[{"name":"sgd_armijo", "gamma":2},
                     {"name":"seg", "gamma":2},
                     {"name":"sgd_goldstein", "gamma":2},
                     {"name":"sgd_polyak", "c":0.1, "momentum":0.6, "gamma":2},]

opt_list_sgd = [{"name":"sgd", "lr":lr} for lr in [1e-1,1e-2,1e-3,1e-4]]

run_list = [0,1,2,3,4]

EXP_GROUPS = {
        "mnist":{"dataset":["mnist"],
            "model":["mlp"],
            "loss_func": ["softmax_loss"],
            "opt":[{"name":"sgd_armijo", "gamma":2}, {"name":"adam"}, ],
            "acc_func":["softmax_accuracy"],
            "batch_size":[128],
            "max_epoch":[200],
            "runs":[0]},
        
        "cifar100":{"dataset":["cifar100"],
            "model":["resnet34_100"],
            "loss_func": ["softmax_loss"],
            "opt":[{"name":"adam"}, {"name":"sgd_armijo", "gamma":2}],
            "acc_func":["softmax_accuracy"],
            "batch_size":[128],
            "max_epoch":[200],
            "runs":[0]},
        
        
    # matrix factorization
    "figure2":{"dataset":["matrix_fac"],
            "model":["matrix_fac_1", "matrix_fac_4", "matrix_fac_10", "linear_fac"],
            "loss_func": ["squared_loss"],
            "opt":opt_list_kernel_mf + others_opt_list,
            "acc_func":["mse"],
            "batch_size":[100],
            "max_epoch":[50],
            "runs":run_list},

    # main dl experiments
    "figure3_mnist":{"dataset":["mnist"],
            "model":["mlp"],
            "loss_func": ["softmax_loss"],
            "opt":opt_list_deep + others_opt_list,
            "acc_func":["softmax_accuracy"],
            "batch_size":[128],
            "max_epoch":[200],
            "runs":run_list},
        
        

    "figure3_cifar10":{"dataset":["cifar10"],
            "model":["resnet34", "densenet121"],
            "loss_func": ["softmax_loss"],
            "opt":opt_list_deep + others_opt_list,
            "acc_func":["softmax_accuracy"],
            "batch_size":[128],
            "max_epoch":[200],
            "runs":run_list},

     "figure3_cifar100":{"dataset":["cifar100"],
            "model":["resnet34_100", "densenet121_100"],
            "loss_func": ["softmax_loss"],
            "opt":opt_list_deep + others_opt_list,
            "acc_func":["softmax_accuracy"],
            "batch_size":[128],
            "max_epoch":[200],
            "runs":run_list},

        "polyak":{"dataset":["cifar10"],
            "model":["resnet34", "densenet121"],
            "loss_func": ["softmax_loss"],
            "opt":["sgd_polyak"],
            "acc_func":["softmax_accuracy"],
            "batch_size":[128],
            "max_epoch":[200],
            "runs":run_list},

        "cifar10":{"dataset":["cifar10"],
            "model":["resnet34"],
            "loss_func": ["softmax_loss"],
            "opt":["sgd_armijo"],
            "acc_func":["softmax_accuracy"],
            "batch_size":[128],
            "max_epoch":[200],
            "runs":[0,1,6,7]},

    # kernel experiments
    "figure6":{"dataset":kernel_datasets,
                        "model":["linear"],
                        "loss_func": ["logistic_loss"],
                        "acc_func": ["logistic_accuracy"],
                        "opt":kernel_opt_list,
                        "batch_size":[100],
                        "max_epoch":[35],
                        "runs":run_list},
    # =============================================================
    # others

    # just sgd_armijo vs. adam
    "sanity": {"dataset":["cifar10"],
            "model":[ "resnet34"],
            "loss_func": ["softmax_loss"],
            "opt":["sgd_armijo", "adam"],
            "acc_func":["softmax_accuracy"],
            "batch_size":[128],
            "max_epoch":[200]},

    "dropout": {"dataset":["cifar10"],
            "model":[ "resnet34"],
            "loss_func": ["softmax_loss"],
            "opt":["sgd_armijo", "adam"],
            "acc_func":["softmax_accuracy"],
            "batch_size":[128],
            "max_epoch":[200]},

    "quick": {"dataset":["cifar10"],
            "model":[ "resnet34", "densenet121"],
            "loss_func": ["softmax_loss"],
            "opt":["sgd_goldstein_pytorch", "sgd_goldstein", "sgd_armijo_pytorch", "sgd_armijo", "adam"],
            "acc_func":["softmax_accuracy"],
            "batch_size":[128],
            "max_epoch":[200]},

    "test_svrg": {"dataset":["mushrooms", "ijcnn", 'rcv1', 'w8a'],
            "model":["linear"],
            "loss_func": ['logistic_loss'],
            "acc_func": ["logistic_accuracy"],
            "opt":['svrg'],
            "batch_size":[100],
            "max_epoch":[2],
            "runs":[0,1]},
    "test_pls": {"dataset":["ijcnn"],
            "model":["linear"],
            "loss_func": ['logistic_loss'],
            "acc_func": ["logistic_accuracy"],
            "opt":['pls'],
            "batch_size":[100],
            "max_epoch":[35],
            "runs":[0,1]},

    #=========================================


            }

EXP_GROUPS = {k:hu.cartesian_exp_group(v) for k,v in EXP_GROUPS.items()}
