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
