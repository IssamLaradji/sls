import torch
import copy
import time

from . import utils as ut

class Sls(torch.optim.Optimizer):
    def __init__(self,
                 params,
                 n_batches_per_epoch=500,
                 init_step_size=1,
                 c=0.1,
                 beta_b=0.9,
                 gamma=2.0,
                 beta_f=2.0,
                 reset_option=1,
                 eta_max=10,
                 bound_step_size=True,
                 line_search_fn="armijo"):
        params = list(params)
        super().__init__(params, {})

        self.line_search_fn = line_search_fn
        self.params = params
        self.c = c
        self.beta_f = beta_f
        self.beta_b = beta_b
        self.eta_max = eta_max
        self.bound_step_size = bound_step_size
        self.gamma = gamma
        self.init_step_size = init_step_size
        self.state['step'] = 0

        self.state['step_size'] = init_step_size

        self.n_batches_per_epoch = n_batches_per_epoch

        self.state['n_forwards'] = 0
        self.state['n_backwards'] = 0

        self.reset_option = reset_option

    def step(self, closure):
        # deterministic closure
        seed = time.time()
        def closure_deterministic():
            with ut.random_seed_torch(int(seed)):
                return closure()

        batch_step_size = self.state['step_size']

        step_size = ut.reset_step(step_size=batch_step_size,
                                   n_batches_per_epoch=self.n_batches_per_epoch,
                                   gamma=self.gamma,
                                   reset_option=self.reset_option,
                                   init_step_size=self.init_step_size)

        # get loss and compute gradients
        loss = closure_deterministic()
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
                    ut.try_sgd_update(self.params, step_size, params_current, grad_current)

                    # compute the loss at the next step; no need to compute gradients.
                    loss_next = closure_deterministic()
                    self.state['n_forwards'] += 1

                    # =================================================
                    # Line search
                    if self.line_search_fn == "armijo":
                        armijo_results = ut.check_armijo_conditions(step_size=step_size,
                                                      step_size_old=step_size_old,
                                                      loss=loss,
                                                      grad_norm=grad_norm,
                                                      loss_next=loss_next,
                                                      c=self.c,
                                                      beta_b=self.beta_b)
                        found, step_size, step_size_old = armijo_results
                        if found == 1:
                            break
                    
                    elif self.line_search_fn == "goldstein":
                        goldstein_results = ut.check_goldstein_conditions(step_size=step_size,
                                                                 loss=loss,
                                                                 grad_norm=grad_norm,
                                                                 loss_next=loss_next,
                                                                 c=self.c,
                                                                 beta_b=self.beta_b,
                                                                 beta_f=self.beta_f,
                                                                 bound_step_size=self.bound_step_size,
                                                                 eta_max=self.eta_max)

                        found = goldstein_results["found"]
                        step_size = goldstein_results["step_size"]

                        if found == 3:
                            break
            
                # if line search exceeds max_epochs
                if found == 0:
                    ut.try_sgd_update(self.params, 1e-6, params_current, grad_current)

        # save the new step-size
        self.state['step_size'] = step_size
        self.state['step'] += 1

        return loss

