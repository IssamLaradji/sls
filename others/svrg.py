import math
import copy
import functools

import torch


class SVRG(torch.optim.Optimizer):
    ''' PyTorch Implementation of SVRG
    Parameters:
        full_grad_closure: a PyTorch-style closure function that computes the full loss + gradient.
    '''
    def __init__(self, model, batch_size, lr, n, full_grad_closure, m=0):
        if m == 0:
            m = n

        defaults = dict(batch_size=batch_size, lr=lr, m=m, n=n)
        super().__init__(model.parameters(), defaults)

        self.state['full_grad_closure'] = full_grad_closure
        self.model = model
        self.state['step'] = 0
        self.state['lr'] = lr

        self.x_tilde_model = copy.deepcopy(model)

        self.state['forward_calls'] = 0
        self.state['backward_calls'] = 0

    def step(self, closure):
        full_loss, grad_norm = None, None
        # update x_tilde
        if self.state['step'] % self.defaults['m'] == 0:
            full_loss, grad_norm = self._update_memory()

        # call the closure to get loss and compute gradients
        x_loss = closure(self.model)

        x_tilde_loss = closure(self.x_tilde_model)

        self.state['forward_calls'] = self.state['forward_calls'] + 2
        self.state['backward_calls'] = self.state['backward_calls'] + 2

        # get x_tilde parameters
        x_tilde = self._get_x_tilde()

        with torch.no_grad():
            for i, group in enumerate(self.param_groups):
                for j, p in enumerate(group['params']):
                    # update model parameters using SVRG update:
                    p.data = p.data - self.state['lr'] * (p.grad - x_tilde[i]['params'][j].grad + self.state['full_grad'][i][j])

        # increment the step counter
        self.state['step'] = self.state['step'] + 1
        self.x_tilde_model.zero_grad()

        return x_loss, full_loss, grad_norm

    def _update_memory(self):
        '''
            Updates the SVRG memory (e.g. x_tilde) and the full gradient for the
            model with parameters x_tilde.
        '''
        # call the closure and get the **full** gradient
        full_loss = self.state['full_grad_closure'](self.model)
        # number of calls required to compute full training loss and gradient
        num_calls = math.ceil(self.defaults['n'] / self.defaults['batch_size'])

        self.state['forward_calls'] = self.state['forward_calls'] + num_calls
        self.state['backward_calls'] = self.state['backward_calls'] + num_calls


        full_grad = []
        # accumulators for necessary dot products
        s_norm = 0
        s_dot_y = 0
        grad_norm = 0
        x_tilde = self._get_x_tilde()

        for i, group in enumerate(self.param_groups):
            full_grad_group = []
            for j, p in enumerate(group['params']):

                grad_copy = torch.zeros_like(p.grad.data).copy_(p.grad.data)
                full_grad_group.append(grad_copy)
                grad_norm = grad_norm + torch.sum(torch.mul(grad_copy, grad_copy))

            full_grad.append(full_grad_group)

        self.state['full_grad'] = full_grad
        # save the new x_tilde model.
        self.x_tilde_model.load_state_dict(self.model.state_dict())

        grad_norm = torch.sqrt(grad_norm).cpu()

        return full_loss, grad_norm


    def _get_x_tilde(self):
        x_tilde = list(self.x_tilde_model.parameters())
        if not isinstance(x_tilde[0], dict):
            x_tilde = [{'params': x_tilde}]
        return x_tilde


# =============================================
# helpers
def full_loss_closure_factory(loader, objective,  grad=True):
    return functools.partial(compute_full_loss, loader=loader, objective=objective, grad=grad)

def compute_full_loss(model, loader, objective,  grad=False):

    loss = 0
    n = 0
    if grad:
        model.zero_grad()

    for _, (Ai, yi) in enumerate(loader):
        Ai = Ai.cuda(); yi = yi.cuda()
        ni = yi.size()[0]
        n += ni
        loss_i = (objective(model, Ai, yi) * ni)

        with torch.no_grad():
            loss += loss_i

        # separate backprop for each batch; otherwise computation graph is too large.
        if grad:
            loss_i.backward()

    loss = (loss / n)

    # scale gradients appropriately (assumption is that objective always returns mean loss)
    if grad:
        for i, param in enumerate(model.parameters()):
            param.grad = param.grad / n

    return loss