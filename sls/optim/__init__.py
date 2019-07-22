import torch


def wolfe_line_search(step_size, step_size_old, loss, grad_norm,
                      loss_temp, c, beta):
    found = 0

    # computing the new break condition
    break_condition = loss_temp - \
        (loss - (step_size) * c * grad_norm**2)

    if (break_condition <= 0):
        found = 1

    else:
        # decrease the step-size by a multiplicative factor
        step_size = step_size * beta

    return found, step_size, step_size_old


def reset_step(step_size, n_batches, beta_2=None, reset_option=1,
               init_step_size=None):
    if reset_option == 0:
        pass

    elif reset_option == 1:
        beta_2 = beta_2 or 2.0
        # try to increase the step-size up to maximum ETA
        step_size = min(
            step_size * beta_2**(1./n_batches),
            10.0)

    elif reset_option == 2:
        step_size = init_step_size

    return step_size

def get_grads(model):
    param_groups = list(model.parameters())

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




from . import sgd_armijo