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


def compute_grad_norm(grad_list):
    grad_norm = 0.
    for g in grad_list:
        if g is None:
            continue
        grad_norm += torch.sum(torch.mul(g, g))
    grad_norm = torch.sqrt(grad_norm)
    return grad_norm


def get_grad_list(params):
    return [p.grad for p in params]