import torch
import copy
import time

from . import utils as ut

class SlsEg(torch.optim.Optimizer):
    def __init__(self,
                 params,
                 n_batches_per_epoch=500,
                 init_step_size=1,
                 c=0.9,
                 beta_b=0.9,
                 gamma=2.0,
                 reset_option=1):
        params = list(params)
        super().__init__(params, {})

        self.params = params
        self.c = c
        self.beta_b = beta_b
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
        grad_current = copy.deepcopy(ut.get_grad_list(self.params))

        grad_norm = ut.compute_grad_norm(grad_current)

        # only do the check if the gradient norm is big enough
        with torch.no_grad():
            if grad_norm >= 1e-8:
                # check if condition is satisfied
                found = 0
                step_size_old = step_size

                for e in range(100):
                    # try prospective step 'w -> w0.5'
                    ut.try_sgd_update(self.params, step_size, params_current, grad_current)

                    # =============================
                    self.zero_grad()
                    # compute the loss at the next step; no need to compute gradients.
                    with torch.enable_grad():
                        loss_next = closure_deterministic()
                        loss_next.backward()

                    self.state['n_forwards'] += 1
                    self.state['n_backwards'] += 1

                    grad_new = [p.grad for p in self.params]

                    grad_diff_norm = compute_diff_norm(grad_new, grad_current)
                    params_diff_norm = compute_diff_norm(self.params, params_current)

                    z = (step_size / self.c) ** 2

                    if float(z * grad_diff_norm - params_diff_norm) > 0:
                        step_size = step_size * self.beta_b
                    else:
                        found = 1
                        break
                    # =============================
                    
                # if line search exceeds max_epochs
                if found == 0:
                    step_size = 1e-6
                    # w -> w0.5
                    ut.try_sgd_update(self.params, step_size, params_current, grad_current)
                    # if line search fails
                    self.zero_grad()
                    with torch.enable_grad():
                        loss_next = closure_deterministic()
                        loss_next.backward()

                    self.state['n_forwards'] += 1
                    self.state['n_backwards'] += 1

                    grad_new = [p.grad for p in self.params]

        # w0.5 -> w
        ut.try_sgd_update(self.params, step_size=step_size,
                       params_current=params_current,
                       grad_current=grad_new)
        
        # save the new step-size
        self.state['step_size'] = step_size
        self.state['step'] += 1

        return loss


def compute_diff_norm(A, B):
    diff_norm = 0.

    zipped = zip(A, B)
    for a, b in zipped:
        # checking only for gradients
        if a is None:
            continue
        diff = a - b
        diff_norm += torch.sum(torch.mul(diff, diff))

    return diff_norm