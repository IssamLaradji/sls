import os 


EXP_GROUPS = {"mnist":{"dataset":["mnist"],
                        "model":["mlp"],
                        "opt":["sgd_armijo", "adam"],
                        "batch_size":[128],
                        "max_epoch":[1000]}}
                        
SAVEDIR_PATH = '/mnt/datasets/public/issam/prototypes/sls'

if not os.path.exists(SAVEDIR_PATH):
    SAVEDIR_PATH = "data/"