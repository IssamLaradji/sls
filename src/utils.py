import hashlib 
import pickle
import json
import os
<<<<<<< HEAD
=======
import itertools
import torch
import numpy as np

>>>>>>> 88b9e8dd238f7e125e67430c5f9e96533a878d67


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

<<<<<<< HEAD

def save_pkl(fname, data):
    """Save data in pkl format."""
    # Save file
    with open(fname, "wb") as f:
        pickle.dump(data, f)
=======
def save_pkl(fname, data):
    """Save data in pkl format."""
    # Save file
    fname_tmp = fname + "_tmp.pkl"
    with open(fname_tmp, "wb") as f:
        pickle.dump(data, f)
    os.rename(fname_tmp, fname)
>>>>>>> 88b9e8dd238f7e125e67430c5f9e96533a878d67


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
<<<<<<< HEAD
        json.dump(data, json_file, indent=4, sort_keys=True)
=======
        json.dump(data, json_file, indent=4, sort_keys=True)

def torch_save(fname, obj):
    """"Save data in torch format."""
    # Define names of temporal files
    fname_tmp = fname + ".tmp"

    torch.save(obj, fname_tmp)
    os.rename(fname_tmp, fname)

def read_text(fname):
    # READS LINES
    with open(fname, "r", encoding="utf-8") as f:
        lines = f.readlines()
        # lines = [line.decode('utf-8').strip() for line in f.readlines()]
    return lines

    
>>>>>>> 88b9e8dd238f7e125e67430c5f9e96533a878d67
