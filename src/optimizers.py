import sls
import torch

def get_optimizer(opt_name, model_parameters):
    if opt_name == "sgd_armijo":
        opt = sls.SgdArmijo(model_parameters, n_batches_in_epoch=1000)

    if opt_name == "adam":
        opt = torch.optim.Adam(model_parameters, lr=1e-3)

    return opt