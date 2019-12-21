# This file should be made to view and aggregate results
#%%
import itertools
import pprint
import argparse
import sys
import os
import pylab as plt
import pandas as pd
import sys
import numpy as np
import hashlib 
import pickle
import json
import glob

import copy

savedir_base = '/mnt/datasets/public/issam/prototypes/sls/borgy'

opt_list  = ["adagrad", "adam", "lbfgs", "ssn_armijo", "sgd_armijo"]
loss_list = [ "squared_hinge_loss", "logistic_loss"]

sys.path.append("/home/issam/Research_Ground/sls_private")
import exp_configs as cg 
from src import mlkit
EXP_GROUPS = cg.EXP_GROUPS

def main():
    exp_groups = [  
        # "quick",
        # "figure3_mnist",
        # "mnist",
        # "cifar10"
        # "figure2",
        # "figure3_mnist",
        # "figure3_cifar10",
        "polyak"
                    ]
    # aggregate exp_configs
    exp_list = []
    for exp_group_name in exp_groups:
        exp_list += EXP_GROUPS[exp_group_name]

    # mlkit.zip_score_list(exp_list, 

    #                      savedir_base=savedir_base, 
    #                      out_fname="%s/cifar100.zip"%savedir_base,
    #                      include_list=["score_list.pkl", "exp_dict.json",
    #                                    "run_dict.json", "borgy_dict.json"])

    # stop
    
    exp_list = get_filtered_exp_list(exp_list, 
                                     regard_dict={
                                        #  "model":"densenet121_100",
                                        #  "dataset":"mnist",
                                        #  "dataset":"cifar100",
                                          "runs":0,
                                        #  "matrix_fac",
                                        #  "runs":0,
                                        #  "dataset":"rcv1",
                                        #  "dataset":"ijcnn",
                                        #  "dataset":"mushrooms",
                                        #  "dataset":"w8a",
                                        #  "dataset":"synthetic",
                                        #  "dataset":"rcv1",
                                        # "model":"resnet34",
                                        # "opt":"sgd_polyak",
                                        
                                        # "margin":1.,

                                    #     "opt":"lbfgs_constant",
                                        
                                        
                                    #     "lm":0,
                                    #    "loss":"squared_hinge_loss",
                                    #    "history_size":10,
                                    #    "loss":"logistic_loss",
                                    #    "margin":0.5
                                     }, 
                                      disregard_dict={"opt":["seg", 
                                      "sgd",
                                    #   "sgd_polyak"
                                      ]}
                                     )
  




    # exp_list = get_filtered_exp_list(exp_list, 
    #                                  regard_dict={
    #                                     #  "dataset":"rcv1",
    #                                     "opt":"ssn_armijo",
    #                                      "margin":0.1,
    #                                     "runs":0,
    #                                    "loss":"logistic_loss"}, 
    #                                 #  disregard_dict={"opt":"sgd_armijo"}
    #                                 )
  
    # exp_list = get_filtered_exp_list(exp_list, 
    #                 regard_dict={
    #                     "dataset":"ijcnn",
                        
    #                     # "runs":0,
    #                     # "batch_grow_factor":
    #                     },
                   
    #             #   "loss":"logistic_loss"}
                  
    #                 disregard_dict={"batch_size":32, 
    #                    "opt":"lbfgs"}
    #                 )

    print("#exps: %d" % len(exp_list))

    df = get_borgy_df(exp_list,  col_list=None, 
                savedir_base=savedir_base)
    
    display(df)
    display(df.loc[df['job_state'] == "RUNNING"])
    display(df.loc[df['job_state'] == "FAILED"])

    # # juy
    df = get_dataframe_score_list(exp_list=exp_list, savedir_base=savedir_base)
    display(df)

    
    row_list = [
        "train_loss",
        "step_size",
     "val_acc", 
    #  "step_size"
    #  
    
    #  "step_size"
    #  "batch_size", "step_size","train_epoch_time"
    ]


    # fig = get_plot(exp_configs, row_list, savedir_base)
   
    fig = get_plot_coupled(exp_list, row_list, savedir_base, 
    
            title_list=("dataset", ),
            legend_list=("opt","batch_size", "model",
            # "batch_grow_factor","lm","loss", "margin","lr",
            # "batch_grow_factor",
        #    "lm"
        ),
            # avg_runs=1,
            # e_epoch=50,
            s_epoch=30,
            )
    
    # fig.savefig(SAVEDIR_BASE + "/plots/exps.jpg")
  
    print("saved...")

# ==========================================
# Utils
# ==========================================

def get_borgy_df(exp_list, savedir_base, col_list=None):
    sys.path.append("/mnt/datasets/public/issam/haven")
    from mlkit import borgy_manager as bm
    job_list_dict = bm._get_job_list_dict(exp_list, savedir_base=savedir_base)

    borgy_list = []
    for exp_dict in exp_list:
        result_dict = {}

        exp_meta = get_exp_meta(exp_dict, savedir_base)
        result_dict["exp_id"] = exp_meta["exp_id"]


        borgy_dict_fname = os.path.join(exp_meta["savedir"],
                                        "borgy_dict.json")
        score_list_fname = os.path.join(exp_meta["savedir"],
                                        "score_list.pkl")
        # Job results
        if os.path.exists(borgy_dict_fname):
            borgy_dict = load_json(borgy_dict_fname)
            job_id = borgy_dict["job_id"]
            if job_id not in job_list_dict:
                continue
            job = job_list_dict[job_id]

            result_dict["job_id"] = job_id
            result_dict["job_state"] = job.state
            result_dict["batch_size"] = exp_dict["batch_size"]
            result_dict["batch_grow_factor"] = exp_dict.get("batch_grow_factor")
            result_dict["dataset"] = exp_dict["dataset"]
            result_dict["loss_func"] = exp_dict["loss_func"]

            # if os.path.exists(score_list_fname):
            #     result_dict["epoch"] = load_pkl(score_list_fname)[-1]["epoch"]

            if job.state == "FAILED":
                err_fname = os.path.join(exp_meta["savedir"], "err.txt")
                result_dict["err"] = read_text(err_fname)




        borgy_list += [result_dict]

    df =  pd.DataFrame(borgy_list).set_index("exp_id")
    if col_list:
        df = df[[c for c in col_list if c in df.columns]]
    if "job_state" in df:
        stats = np.vstack(np.unique(df['job_state'],return_counts=True)).T
        print([{a:b} for (a,b) in stats])
    return df

def get_dataframe_exp_list(exp_list,  col_list=None, savedir_base=""):
    
    meta_list = []
    for exp_dict in exp_list:
        exp_meta = get_exp_meta(exp_dict, savedir_base)
        meta_dict = copy.deepcopy(flatten_dict(exp_dict))

        meta_dict["exp_id"] = exp_meta["exp_id"]
        # meta_dict["savedir"] = exp_meta["savedir"]
        # meta_dict["command"] = exp_meta["command"]
        
        if meta_dict == {}:
            continue

        meta_list += [meta_dict]
    df =  pd.DataFrame(meta_list).set_index("exp_id")

    if col_list:
        df = df[[c for c in col_list if c in df.columns]]

    return df

def get_dataframe_score_list(exp_list, col_list=None, savedir_base=None):
    score_list_list = []

    # aggregate results
    for exp_dict in exp_list:
        result_dict = {}

        exp_meta = get_exp_meta(exp_dict, savedir_base=savedir_base)
        result_dict["exp_id"] = exp_meta["exp_id"]
        if not os.path.exists(exp_meta["savedir"]+"/score_list.pkl"):
            score_list_list += [result_dict]
            continue

        score_list_fname = os.path.join(exp_meta["savedir"], "score_list.pkl")

        if os.path.exists(score_list_fname):
            score_list = load_pkl(score_list_fname)
            score_df = pd.DataFrame(score_list)
            if len(score_list):
                score_dict_last = score_list[-1]
                for k, v in score_dict_last.items():
                    if "float" not  in str(score_df[k].dtype):
                        result_dict[k] = v
                    else:
                        result_dict[k] = "%.3f (%.3f-%.3f)" % (v, score_df[k].min(), score_df[k].max())

        score_list_list += [result_dict]

    df = pd.DataFrame(score_list_list).set_index("exp_id")
    
    # join with exp_dict df
    df_exp_list = get_dataframe_exp_list(exp_list, col_list=col_list)
    df = join_df_list([df, df_exp_list])
    
    # filter columns
    if col_list:
        df = df[[c for c in col_list if c in df.columns]]

    return df

def get_plot(exp_groups, savedir_base, col_list, 
             col_label=None,
             col_list_title=None):

    nrows = len(col_list)
    ncols = len(exp_groups)
    fig, axs = plt.subplots(nrows=max(1,nrows), 
                            ncols=max(1, ncols), 
                            figsize=(ncols*6, nrows*6))
    if axs.ndim == 1:
        axs = axs[:, None]
    for i, col in enumerate(col_list):
        for j, exp_config_name in enumerate(exp_groups):
            exp_list = cartesian_exp_config(cfg.EXP_CONFIG[exp_config_name])
        
            for exp_dict in exp_list:
                exp_meta = get_exp_meta(exp_dict, savedir_base=savedir_base)
                path = exp_meta["savedir"] + "/score_list.pkl"
                if os.path.exists(path):
                    score_list = load_pkl(path)
                    score_df = pd.DataFrame(score_list)
                    axs[i,j].plot(score_df["epoch"], score_df[col],
                                        label=exp_dict[col_label])
            # prepare figure
            axs[i,j].set_ylabel(col)
            axs[i,j].set_xlabel("epochs")
            if col_list_title is not None:
                axs[i,j].set_title("_".join([exp_dict[c] for c in col_list_title]))
            
            axs[i,j].legend( loc='upper right', bbox_to_anchor=(0.5, -0.05))  

                
    return fig

def view_images(exp_list, savedir_base, image_dir, n_images=100):
    for exp_dict in exp_list:
        exp_meta = get_exp_meta(exp_dict, savedir_base=savedir_base)
        pprint.pprint(exp_dict)
        meta_dict = copy.deepcopy(flatten_dict(exp_dict))

        meta_dict["exp_id"] = exp_meta["exp_id"]
        savedir = exp_meta["savedir"] + "/" + image_dir
        
        show_folder(savedir, n_images=n_images)
    

# ===========================================
# helpers
def join_df_list(df_list):
    result_df = df_list[0]
    for i in range(1, len(df_list)):
        result_df = result_df.join(df_list[i], how="outer", lsuffix='_%d'%i, rsuffix='')
    return result_df

def show_folder(savedir, n_images=100):
    image_list = glob.glob(savedir + "*.jpg") + glob.glob(savedir + "*.png")
    image_list.sort(key=os.path.getmtime)
    image_list = image_list[::-1]
    if len(image_list) == 0:
        print("\nno images found at %s" % savedir)
        return
    for i, fname in enumerate(image_list):
        if i < n_images:
            # print(i)
            plt.figure(figsize=(11,4))
            plt.imshow(plt.imread(fname))
            plt.title(extract_fname(fname))

            # if os.path.exists(fname.replace(".jpg",".json")):
            #     pprint(load_json(fname.replace(".jpg",".json")))
            meta_name = fname.split(".")[0] + ".pkl"
            if os.path.exists(meta_name):
                plt.title(load_pkl(meta_name))
            plt.tight_layout()
            plt.show()


def get_plot_coupled(exp_list, row_list, savedir_base, 
                    title_list=None,
                     legend_list=None, avg_runs=0, 
                     s_epoch=None,e_epoch=None):
    nrows = len(row_list)
    # ncols = len(exp_configs)
    ncols = 1
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, 
                            figsize=(ncols*6, nrows*6))
 
    for i, row in enumerate(row_list):
        # exp_list = cartesian_exp_config(EXP_GROUPS[exp_config_name])
    
        for exp_dict in exp_list:
            exp_meta = get_exp_meta(exp_dict, savedir_base=savedir_base)
            path = exp_meta["savedir"] + "/score_list.pkl"
            if os.path.exists(path) and os.path.exists(exp_meta["savedir"] + "/exp_dict.json"):
                if exp_dict.get("runs") is None or not avg_runs:
                    mean_list = load_pkl(path)
                    mean_df = pd.DataFrame(mean_list)
                    std_df = None

                elif exp_dict.get("runs") == 0:
                    # score_list = load_pkl(path)
                    mean_df, std_df = get_score_list_across_runs(exp_dict, savedir_base=savedir_base)

                else:
                    continue
                
                if s_epoch:
                    axs[i].plot(mean_df["epoch"][s_epoch:], 
                        mean_df[row][s_epoch:],
                                label="_".join([str(exp_dict.get(k)) for k in legend_list]))

                elif e_epoch:
                    axs[i].plot(mean_df["epoch"][:e_epoch], mean_df[row][:e_epoch],
                                label="_".join([str(exp_dict.get(k)) for k in legend_list]))
                else:
                    axs[i].plot(mean_df["epoch"], mean_df[row],
                                label="_".join([str(exp_dict.get(k)) for k in legend_list]))
                if std_df is not None:
                    # do shading
                    offset = 0
                    # print(mean_df[row][offset:] - std_df[row][offset:])
                    # adsd
                    axs[i].fill_between(mean_df["epoch"][offset:], 
                            mean_df[row][offset:] - std_df[row][offset:],
                            mean_df[row][offset:] + std_df[row][offset:], 
                            # color = label2color[labels[i]],  
                            alpha=0.5)

        # prepare figure
        if "loss" in row:   
            axs[i].set_yscale("log")
            axs[i].set_ylabel(row + " (log)")
        else:
            axs[i].set_ylabel(row)
        axs[i].set_xlabel("epochs")
        axs[i].set_title("_".join([str(exp_dict.get(k)) for k in title_list]))
                            

        axs[i].legend( loc='upper right', bbox_to_anchor=(0.5, -0.05))  
        # axs[i].set_ylim(.90, .94)  
    plt.grid(True)  
               
    return fig

def get_score_list_across_runs(exp_dict, savedir_base):    
    exp_meta = get_exp_meta(exp_dict, savedir_base=savedir_base)
    score_list = load_pkl(exp_meta["savedir"] + "/score_list.pkl")
    keys = score_list[0].keys()
    result_dict = {}
    for k in keys:
        result_dict[k] = np.ones((exp_dict["max_epoch"], 5))*-1
    

    bad_keys = set()
    for r in [0,1,2,3,4]:
        exp_dict_new = copy.deepcopy(exp_dict)
        exp_dict_new["runs"]  = r
        exp_meta = get_exp_meta(exp_dict_new, savedir_base=savedir_base)
        score_list_new = load_pkl(exp_meta["savedir"] + "/score_list.pkl")

        df = pd.DataFrame(score_list_new)
        for k in keys:
            values =  np.array(df[k])
            if values.dtype == "O":
                bad_keys.add(k)
                continue
            result_dict[k][:values.shape[0], r] = values

    for k in keys:
        if k in bad_keys:
                continue
        assert -1 not in result_dict[k]

    mean_df = pd.DataFrame()
    std_df = pd.DataFrame()
    for k in keys:
        mean_df[k] = result_dict[k].mean(axis=1)
        std_df[k] = result_dict[k].std(axis=1)
    return mean_df, std_df
def get_plot(exp_configs, col_list, savedir_base):
    nrows = len(col_list)
    ncols = len(exp_configs)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, 
                            figsize=(ncols*6, nrows*6))
 
    for i, col in enumerate(col_list):
        
        for j, exp_config_name in enumerate(exp_configs):
            

            exp_list = cartesian_exp_config(EXP_GROUPS[exp_config_name])
        
            for exp_dict in exp_list:
                exp_meta = get_exp_meta(exp_dict, savedir_base=savedir_base)
                path = exp_meta["savedir"] + "/score_list.pkl"
                if os.path.exists(path):
                    score_list = load_pkl(path)
                    score_df = pd.DataFrame(score_list)
                    axs[i,j].plot(score_df["epoch"], score_df[col],
                                label="%s"%exp_dict["opt"])
            # prepare figure
            if "loss" in col:   
                axs[i,j].set_yscale("log")
                axs[i,j].set_ylabel(col + " (log)")
            else:
                axs[i,j].set_ylabel(col)
            axs[i,j].set_xlabel("epochs")
            axs[i,j].set_title("%s_%s_margin:%s" % 
                                (exp_dict["dataset"], 
                                exp_dict["model"],
                                exp_dict["margin"]))

            axs[i,j].legend( loc='upper right', bbox_to_anchor=(0.5, -0.05)) 

                
    return fig

    
def get_table(exp_configs, col_list, savedir_base):
    nrows = len(col_list)
    ncols = len(exp_configs)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, 
                            figsize=(ncols*6, nrows*6))
    # i = -1
    for i, col in enumerate(col_list):
        # i += 1
        for j, exp_config_name in enumerate(exp_configs):
            exp_list = cartesian_exp_config(EXP_GROUPS[exp_config_name])
            for exp_dict in exp_list:
                exp_meta = get_exp_meta(exp_dict, savedir_base=savedir_base)
                path = exp_meta["savedir"] + "/score_list.pkl"
                if os.path.exists(path):
                    score_list = load_pkl(path)
                    score_df = pd.DataFrame(score_list)
                    axs[i,j].plot(score_df["epoch"], score_df[col],
                                label="%s"%exp_dict["opt"])

                axs[i,j].set_ylabel(col) 
                axs[i,j].set_xlabel("epochs")
                axs[i,j].set_title("%s_%s_margin:%s" % (exp_dict["dataset"], exp_dict["model"],exp_dict["margin"]))
                axs[i,j].legend( loc='upper right', bbox_to_anchor=(0.5, -0.05)) 
    return fig

def cartesian_exp_config(exp_config):
    # Create the cartesian product
    exp_list_raw = (dict(zip(exp_config.keys(), values))
                    for values in itertools.product(*exp_config.values()))

    # Convert into a list
    exp_list = []
    for exp_dict in exp_list_raw:
        exp_list += [exp_dict]
    return exp_list

def get_exp_meta(exp_dict, savedir_base, mode=None, remove_keys=None,
                 fname=None, workdir=None):
    exp_dict_new = copy.deepcopy(exp_dict)

    if remove_keys:
        for k in remove_keys:
            if k in exp_dict_new:
                del exp_dict_new[k]

    if mode is not None:
        exp_dict_new["mode"] = mode

    exp_id = hash_dict(exp_dict_new)
    savedir = "%s/%s" % (savedir_base, exp_id)
    if not fname:
        fname = extract_fname(os.path.abspath(sys.argv[0]))
    if not workdir:
        workdir = sys.path[0]

    exp_meta = {}
    exp_meta["exp_id"] = exp_id
    exp_meta["command"] = ("python %s -s %s" % (fname, savedir))
    exp_meta["workdir"] = workdir
    exp_meta["savedir"] = savedir

    return exp_meta




# ================================================
# Utils 
def hash_dict(dictionary):
    """Create a hash for a dictionary."""
    dict2hash = ""

    for k in sorted(dictionary.keys()):
        if isinstance(dictionary[k], dict):
            v = hash_dict(dictionary[k])
        else:
            v = dictionary[k]

        dict2hash += "%s_%s_" % (str(k), str(v))

    return hashlib.md5(dict2hash.encode()).hexdigest()

def load_pkl(fname):
    """Load the content of a pkl file."""
    with open(fname, "rb") as f:
        return pickle.load(f)

def load_json(fname, decode=None):
    with open(fname, "r") as json_file:
        d = json.load(json_file)

    return d

def read_text(fname):
    # READS LINES
    with open(fname, "r", encoding="utf-8") as f:
        lines = f.readlines()
        # lines = [line.decode('utf-8').strip() for line in f.readlines()]
    return lines

def extract_fname(directory):
    import ntpath
    return ntpath.basename(directory)

def flatten_dict(exp_dict):
    result_dict = {}
    for k in exp_dict:
        # print(k, exp_dict)
        if isinstance(exp_dict[k], dict):
            for k2 in exp_dict[k]:
                result_dict[k2] = exp_dict[k][k2]
        else:
            result_dict[k] = exp_dict[k]
    return result_dict

def filter_flag(exp_dict, regard_dict=None, disregard_dict=None):
    # regard dict
    flag_filter = False
    flattened = flatten_dict(exp_dict)
    if regard_dict:
        for k in regard_dict:
            if flattened.get(k) != regard_dict[k]:
                flag_filter = True
                break

    # disregard dict
    if disregard_dict:
        for k in disregard_dict:
            if flattened.get(k) in disregard_dict[k]:
                flag_filter = True
                break

    return flag_filter


def get_filtered_exp_list(exp_list, regard_dict=None, disregard_dict=None):
    fname_list = glob.glob(savedir_base + "/*/exp_dict.json")

    exp_list_new = []
    for exp_dict in exp_list:
        if filter_flag(exp_dict, regard_dict, disregard_dict):
            continue
        exp_list_new += [exp_dict]

    return exp_list_new

def get_filtered_exp_list_savedir(savedir_base, regard_dict=None, disregard_dict=None):
    fname_list = glob.glob(savedir_base + "/*/exp_dict.json")

    exp_list_new = []
    for fname in fname_list:
        exp_dict = load_json(fname)
        if filter_flag(exp_dict, regard_dict, disregard_dict):
            continue
        exp_list_new += [exp_dict]

    return exp_list_new

# def get_filtered_exp_list(exp_list, regard_dict=None, disregard_dict=None):
#     exp_list_new = []
#     for exp_dict in exp_list:
#         if filter_flag(exp_dict, regard_dict, disregard_dict):
#             continue
#         exp_list_new += [exp_dict]
#     return exp_list_new

if __name__ == "__main__":
    main()
    

# %%
