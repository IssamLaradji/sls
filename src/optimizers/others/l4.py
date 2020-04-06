import math

import torch
import torch.optim as optim

class L4(optim.Optimizer):
  """Implements L4: Practical loss-based stepsize adaptation for deep learning

  Proposed by Michal Rolinek & Georg Martius in
  `paper <https://arxiv.org/abs/1802.05074>`_.

  Arguments:
      params (iterable): iterable of parameters to optimize or dicts defining
          parameter groups
      optimizer: an optimizer to wrap with L4
      alpha (float, optional): scale the step size, recommended value is in range (0.1, 0.3) (default: 0.15)
      gamma (float, optional): scale min Loss (default: 0.9)
      tau (float, optional): min Loss forget rate (default: 1e-3)
      eps (float, optional): term added to the denominator to improve
          numerical stability (default: 1e-12)

  """

  def __init__(self, params, optimizer, alpha=0.15, gamma=0.9, tau=1e-3, eps=1e-12):
    defaults = dict(alpha=alpha, gamma=gamma, tau=tau, eps=eps)
    super(L4, self).__init__(params, defaults)

    self.state['optimizer'] = optimizer
    self.state['alpha'] = alpha
    self.state['gamma'] = gamma
    self.state['tau'] = tau
    self.state['eps'] = eps


  def zero_grad(self):
    self.state['optimizer'].zero_grad()

  def step(self, closure):
    # modified by issam
    loss_tmp = closure()
    loss_tmp.backward()
    #

    loss = float(loss_tmp.detach())

    if loss is None:
      raise RuntimeError('L4: loss is required to step')

    if loss < 0:
      raise RuntimeError('L4: loss must be non negative')

    if math.isnan(loss):
      return

    # copy original data for parameters
    originals = {}
    # grad estimate decay
    decay = 0.9

    state = self.state
    if 'step' not in state:
      state['step'] = 0

    state['step'] += 1
    #correction_term = 1 - math.exp(state['step']  * math.log(decay))
    correction_term = 1 - decay ** state['step']


    for group in self.state['optimizer'].param_groups:
      for p in group['params']:
        if p.grad is None:
          continue

        if p not in state:
          state[p] = torch.zeros_like(p.grad.data)

        # grad running average momentum
        state[p].mul_(decay).add_(1 - decay, p.grad.data)

        if p not in originals:
          originals[p] = torch.zeros_like(p.data)
        originals[p].copy_(p.data)
        p.data.zero_()


    if 'lmin' not in state:
      state['lmin'] = loss * 0.75

    lmin = min(state['lmin'], loss)

    gamma = state['gamma']
    tau = state['tau']
    alpha = state['alpha']
    eps = state['eps']

    self.state['optimizer'].step()

    inner_prod = 0

    for group in self.state['optimizer'].param_groups:
      for p in group['params']:
        if p.grad is None:
          continue
        grad = state[p].div(correction_term)
        v = -p.data.clone()
        inner_prod += torch.dot(grad.view(-1), -p.data.view(-1))

    lr = alpha * (loss - lmin * gamma) / (inner_prod + eps)
    state['lr'] = lr
    for group in self.state['optimizer'].param_groups:
      for p in group['params']:
        if p.grad is None:
          continue
        grad = state[p].div(correction_term)
        v = -p.data.clone()
        p.data.copy_(originals[p])
        p.data.add_(-lr, v)

    state['lmin'] = (1 + tau) * lmin

    return loss