import torch
import copy
import time
import math

from . import utils as ut

class SlsAcc(torch.optim.Optimizer):
    def __init__(self,
                 params,
                 init_step_size=1,
                 c=0.1,
                 beta_b=0.9,
                 gamma=2.0,
                 momentum=0.6,
                 reset_option=0,
                 acceleration_method="polyak"):
        params = list(params)
        super().__init__(params, {})

        self.params = params
        self.momentum = momentum
        self.c = c
        self.beta_b = beta_b
        self.gamma = gamma
        self.init_step_size = init_step_size
        self.acceleration_method = acceleration_method
        self.state['step'] = 0

        self.state['step_size'] = init_step_size


        self.state['n_forwards'] = 0
        self.state['n_backwards'] = 0

        self.reset_option = reset_option

        if acceleration_method == "polyak":
            self.state['params_current'] = copy.deepcopy(self.params)
       
        elif acceleration_method == "nesterov":
            self.state['y_params_old'] = copy.deepcopy(self.params)

            self.state['lambda_old'] = 0
            self.state['lambda_current'] = 1
            self.state['tau'] = 1
        
        else:
            raise ValueError("%s is not supported" % acceleration_method)

    def step(self, closure):
        # deterministic closure
        seed = time.time()
        def closure_deterministic():
            with ut.random_seed_torch(int(seed)):
                return closure()

        batch_step_size = self.state['step_size']

        step_size = ut.reset_step(step_size=batch_step_size,
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

        # Polyak: Keep track of params_current (it will be params_old)
        if self.acceleration_method == "polyak":
            params_old = copy.deepcopy(self.state['params_current'])
            self.state['params_current'] = params_current

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
                   
                # if line search exceeds max_epochs
                if found == 0:
                    ut.try_sgd_update(self.params, 1e-6, params_current, grad_current)

        if self.acceleration_method == "polyak":
            polyak_update(self.params, self.state['params_current'], grad_current,
                                              params_old, self.momentum)

        elif self.acceleration_method == "nesterov":
            y_params = copy.deepcopy(self.params)

            nesterov_update(self.params, grad_current,
                            y_params_old=self.state["y_params_old"],
                            gamma=self.state['tau'])

            self.state["y_params_old"] = copy.deepcopy(y_params)

            lambda_tmp = self.state['lambda_current']
            self.state['lambda_current'] = (1 + math.sqrt(1 + 4 * self.state['lambda_old']  *
                                                          self.state['lambda_old'] )) / 2
            self.state['lambda_old'] = lambda_tmp
            self.state['tau'] = (1 - self.state['lambda_old']) / self.state['lambda_current']

        # save the new step-size
        self.state['step_size'] = step_size
        self.state['step'] += 1

        return loss


def polyak_update(params_model, params_current, grad_current, params_old, momentum):
    zipped = zip(params_model, params_current, grad_current, params_old)

    for p_model, p_current, g_current, p_old in zipped:
        if g_current is None:
            continue

        p_model.data = p_model.data + momentum * (p_current - p_old.to(p_model.data.device))

def nesterov_update(params_model, grad_current, y_params_old, gamma):
    zipped = zip(params_model, grad_current, y_params_old)

    for p_model, g_current, y_old in zipped:
        if g_current is None:
            continue

        p_model.data = (1-gamma) * p_model.data + gamma * y_old.to(p_model.data.device)
