import hashlib 
import pickle
import json
import os
import itertools
import torch


def cartesian_exp_group(exp_group_name):
    exp_list = []
    exp_list_raw = (dict(zip(exp_group_name.keys(), values))
                    for values in itertools.product(*exp_group_name.values()))

    for exp_dict in exp_list_raw:
        exp_list += [exp_dict]
    return exp_list

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


def save_pkl(fname, data):
    """Save data in pkl format."""
    # Save file
    with open(fname, "wb") as f:
        pickle.dump(data, f)


def load_pkl(fname):
    """Load the content of a pkl file."""
    with open(fname, "rb") as f:
        return pickle.load(f)

def load_json(fname, decode=None):
    with open(fname, "r") as json_file:
        d = json.load(json_file)

    return d

def save_json(fname, data):
    with open(fname, "w") as json_file:
        json.dump(data, json_file, indent=4, sort_keys=True)

def torch_save(fname, obj):
    """"Save data in torch format."""
    # Define names of temporal files
    fname_tmp = fname + ".tmp"

    torch.save(obj, fname_tmp)
    os.rename(fname_tmp, fname)
