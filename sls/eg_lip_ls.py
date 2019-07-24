import math
import copy

import torch
from . import utils as ut


class EG_LIP_LS(torch.optim.Optimizer):
    ''' PyTorch Implementation of SEG with Lipschitz line-search
    '''
    def __init__(self, params, max_epochs, batch_size, init_step_size=1, n=1, reset_option=0, sigma=0.5, beta=0.5, beta_2=None, bound_step_size=False):

        defaults = dict(max_epochs=max_epochs, batch_size=batch_size, init_step_size=init_step_size, n=n, sigma=sigma, beta=beta, reset_option=reset_option, beta_2=beta_2, bound_step_size=bound_step_size)
        super().__init__(params, defaults)


        self.state['step'] = 0
        self.state['step_size'] = init_step_size

        # book-keeping for the Lipschitz line-search
        self.state['x_prev'] = copy.deepcopy(self.param_groups)

        self.state['n_forwards'] = 0
        self.state['n_backwards'] = 0

    def step(self, closure):

        step_size = reset_step(self.state, self.defaults)

        # call the closure to get loss and compute gradients
        loss = closure()
        loss.backward()

        self.state['n_forwards'] += 1
        self.state['n_backwards'] += 1

        # save the current parameters:
        x_current = copy.deepcopy(self.param_groups)
        self.state['x_prev'] = copy.deepcopy(self.param_groups)
        # save the gradient at the current parameters:

        gradient = [p.grad for p in self.params]
        grad_norm = optim.compute_grad_norm(gradient)

        # only do the check if the gradient norm is big enough
        if grad_norm >= 1e-8:

            # check if condition is satisfied
            found = 0

            for e in range(100):
                # try a prospective step
                self._try_update(step_size, x_current, gradient)

                # compute the loss at the next step; Lipschitz line-search requires new gradients.
                loss_temp = closure(compute_grad=True)

                self.state['n_forwards'] += 1
                self.state['n_backwards'] += 1

                gradient_temp = [p.grad for p in self.params]
                grad_norm_temp = optim.compute_grad_norm(gradient_temp)

                g_norm = _compute_grad_diff(gradient_temp, gradient)
                x_norm = _compute_iter_diff(x_current)

                # implements the lipschitz condition in the paper
                c = (step_size / self.defaults['sigma'])**2
                break_condition = float(c * g_norm - x_norm)
                if (break_condition <= 0):

                    found = 1
                    break

                else:

                    # decrease the step-size by a multiplicative factor
                    step_size = step_size * self.defaults['beta']

            if found == 0:
                self._try_update(1e-6, x_current, gradient)

        else:
            self._try_update(step_size, x_current, gradient)

        # save the new step-size
        self.state['step_size'] = step_size
        self.state['step'] = self.state['step'] + 1

        # take the extra gradient step.
        self.EG_step(closure)

        return loss

    def EG_step(self, closure):

        # call the closure to get loss and compute gradients.
        loss = closure(compute_grad=True)
        self._update_func_evals_counters(backward_called=True)

        # save the gradient at the current parameters:
        gradient, grad_norm = self.model.get_grads()

        self._try_update(self.state['step_size'], self.state['x_prev'], gradient)

        return loss


def _try_update(param_groups, step_size, x_current, gradient):

    with torch.no_grad():
        for i, group in enumerate(param_groups):
            for j, p in enumerate(group['params']):
                # update models parameters using SGD update
                p.data = x_current[i]['params'][j] - step_size * gradient[i][j]


def _compute_iter_diff(param_groups, x_current):

    x_norm = 0
    with torch.no_grad():
        for i, group in enumerate(param_groups):
            for j, p in enumerate(group['params']):
                iter_diff = x_current[i]['params'][j] - p.data
                x_norm = x_norm + torch.sum(torch.mul(iter_diff, iter_diff))

    return x_norm

def _compute_grad_diff(param_groups, g_current, g_prev):

    g_norm = 0
    with torch.no_grad():
        for i, group in enumerate(param_groups):
            for j, p in enumerate(group['params']):
                g_diff = g_current[i][j] - g_prev[i][j]
                g_norm = g_norm + torch.sum(torch.mul(g_diff, g_diff))

    return g_norm