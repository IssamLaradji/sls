import torch
import copy

from . import utils as ut

class SgdArmijo(torch.optim.Optimizer):
    def __init__(self,
                 params,
                 n_batches_in_epoch,
                 init_step_size=1,
                 c=0.1,
                 beta=0.9,
                 beta_2=None,
                 reset_option=1):
        params = list(params)
        super().__init__(params, {})

        self.params = params
        self.c = c
        self.beta = beta
        self.init_step_size = init_step_size
        self.state['step'] = 0

        self.state['step_size'] = init_step_size

        self.beta_2 = beta_2

        self.n_batches_in_epoch = n_batches_in_epoch

        self.state['n_forwards'] = 0
        self.state['n_backwards'] = 0

        self.reset_option = reset_option

    def step(self, closure):
        batch_step_size = self.state['step_size']

        step_size = ut.reset_step(step_size=batch_step_size,
                                   n_batches=self.n_batches_in_epoch,
                                   beta_2=self.beta_2,
                                   reset_option=self.reset_option,
                                   init_step_size=self.init_step_size)

        # get loss and compute gradients
        loss = closure()
        loss.backward()

        # increment # forward-backward calls
        self.state['n_forwards'] += 1
        self.state['n_backwards'] += 1

        # save the current parameters:
        params_current = copy.deepcopy(self.params)
        grad_current = ut.get_grad_list(self.params)

        grad_norm = ut.compute_grad_norm(grad_current)

        # only do the check if the gradient norm is big enough
        with torch.no_grad():
            if grad_norm >= 1e-8:
                # check if condition is satisfied
                found = 0
                step_size_old = step_size

                for e in range(100):
                    # try a prospective step
                    self._try_update(step_size, params_current, grad_current)

                    # compute the loss at the next step; no need to compute gradients.
                    loss_next = closure()
                    self.state['n_forwards'] += 1

                    wolfe_results = ut.wolfe_line_search(step_size=step_size,
                                                      step_size_old=step_size_old,
                                                      loss=loss,
                                                      grad_norm=grad_norm,
                                                      loss_temp=loss_next,
                                                      c=self.c,
                                                      beta=self.beta)

                    found, step_size, step_size_old = wolfe_results

                    if found == 1:
                        break

                if found == 0:
                    self._try_update(1e-6, params_current, grad_current)

            else:
                self._try_update(step_size, params_current, grad_current)

        # save the new step-size
        self.state['step_size'] = step_size
        self.state['step'] += 1

        return loss

    def _try_update(self, step_size, params_current, grad_current):
        zipped = zip(self.params, params_current, grad_current)

        for p_next, p_current, g_current in zipped:
            p_next.data = p_current - step_size * g_current


