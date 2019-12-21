import argparse
import sys
import pandas as pd 
import os
import pylab as plt
from src import utils as ut
import exp_configs


def main(exp_list, savedir_base, exp_group_name):
    print("#exps: %d" % len(exp_list))

    df = get_dataframe_score_list(exp_list=exp_list, savedir_base=savedir_base)
    print(df)

    fig = get_plot(exp_list, ["train_loss", "val_acc"], savedir_base, 
            title_list=("dataset", ),
            legend_list=("opt","batch_size", "model",
        ),
            )
    
    fname = "results/%s.jpg" % exp_group_name
    fig.savefig(fname)
    print("saved result in %s" % fname)

def get_plot(exp_list, row_list, savedir_base, 
                    title_list=None,
                     legend_list=None, avg_runs=0, 
                     s_epoch=None,e_epoch=None):
    ncols = len(row_list)
    # ncols = len(exp_configs)
    nrows = 1
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, 
                            figsize=(ncols*6, nrows*6))
 
    for i, row in enumerate(row_list):
        # exp_list = cartesian_exp_config(EXP_GROUPS[exp_config_name])
    
        for exp_dict in exp_list:
            exp_id = ut.hash_dict(exp_dict)
            savedir = savedir_base + "/%s/" % exp_id 

            path = savedir + "/score_list.pkl"
            if os.path.exists(path) and os.path.exists(savedir + "/exp_dict.json"):
                mean_list = ut.load_pkl(path)
                mean_df = pd.DataFrame(mean_list)

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
                
        if "loss" in row:   
            axs[i].set_yscale("log")
            axs[i].set_ylabel(row + " (log)")
        else:
            axs[i].set_ylabel(row)
        axs[i].set_xlabel("epochs")
        axs[i].set_title("_".join([str(exp_dict.get(k)) for k in title_list]))
                            

        axs[i].legend(loc="best")  
        # axs[i].set_ylim(.90, .94)  
    plt.grid(True)  
               
    return fig

def get_dataframe_score_list(exp_list, col_list=None, savedir_base=None):
    score_list_list = []

    # aggregate results
    for exp_dict in exp_list:
        result_dict = {}

        exp_id = ut.hash_dict(exp_dict)
        result_dict["exp_id"] = exp_id
        savedir = savedir_base + "/%s/" % exp_id 
        if not os.path.exists(savedir + "/score_list.pkl"):
            score_list_list += [result_dict]
            continue

        score_list_fname = os.path.join(savedir, "score_list.pkl")

        if os.path.exists(score_list_fname):
            score_list = ut.load_pkl(score_list_fname)
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
    
    # filter columns
    if col_list:
        df = df[[c for c in col_list if c in df.columns]]

    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--exp_group_list', nargs="+")
    parser.add_argument('-sb', '--savedir_base', required=True)

    args = parser.parse_args()

    # aggregate exp_configs
    exp_list = []
    for exp_group_name in args.exp_group_list:
        exp_list += exp_configs.EXP_GROUPS[exp_group_name]

    main(exp_list,
         savedir_base=args.savedir_base, exp_group_name=exp_group_name)

    
