# -*- coding: utf-8 -*-
"""
Implements the Gaussian process functionality needed for the probabilistic
line search algorithm.
"""

import numpy as np
from scipy import linalg

from others.pls_utils.utils import erf

class ProbLSGaussianProcess(object):
  """Gaussian process implementation for probabilistic line searches [1].
  Implements 1D GP regression with observations of function value and
  derivative. Kernel is a once-integrated Wiener process with theta=1.0.

  Public interface:
      - ``gp = ProbLSGaussianProcess()``
      - ``gp.add(t, f, df, sigma2_f, sigma2_df)`` to add a new observation.
      - ``gp.reset()`` to remove all observations.
      - ``gp.update()`` to set up and invert the Gram matrix and make the GP
        ready for inference (i.e. the following methods).
      - ``gp.mu(t)`` returns the posterior mean at ``t``.
      - ``gp.V(t)`` returns the posterior variance ``t``.
      - ``gp.expected_improvement(t)`` returns the expected improvement at
        ``t ``
      - ``gp.cubic_polynomial_coefficients(t)`` to get the coefficients of the
        cubic polynomial ``gp.mu()`` at ``t`` (the posterior mean is
        piece-wise cubic).
      - ``gp.find_cubic_minima()`` to get the minima (if existent) of the cubic
        polynomials in each "cell", i.e. between (sorted) observation at t_i
        and t_i+1.
      - ``gp.find_dmu_equal(val)``, like ``find_cubic_minima()``, but for
        points where the derivative of the posterior mean equals ``val`` (and
        the second derivative is positive).

  [1] M. Mahsereci and P. Hennig. Probabilistic line searches for stochastic
  optimization. In Advances in Neural Information Processing Systems 28, pages
  181-189, 2015"""

  def __init__(self, theta=1.0, offset=10.0):
    """Create a new GP object."""

    # Hyperparamters of the GP
    self.theta = theta
    self.offset = offset

    # Observation counter and arrays to store observations
    self.N = 0
    self.ts = []
    self.fs = []
    self.dfs = []
    self.fvars = []
    self.dfvars = []

    # Kernel matrices
    self.K = None
    self.Kd = None
    self.dKd = None

    # Gram matrix and pre-computed "weights" of the GP
    self.G = None
    self.w = None

    # Switch that remembers whether we are ready for inference (calls to mu,
    # V, etc...). It is set to False when the GP is manipulated (points added,
    # noise level adjusted, reset). After such manipulations, gp.update() has
    # to be called. Remember current best observation of exp. improvement
    self.ready = False
    self.min_obs = None

  def reset(self):
    """Reset the GP, removing all previous observations.

    Automatically adds the observation at t=0 (with f=0 and df=-1)."""

    self.N = 0
    self.ts = []
    self.fs = []
    self.dfs = []
    self.fvars = []
    self.dfvars = []
    self.K = None
    self.Kd = None
    self.dKd = None
    self.G = None
    self.LU = None
    self.LU_piv = None
    self.w = None

    self.min_obs = None
    self.ready = False

  def add(self, t, f, df, fvar=0.0, dfvar=0.0):
    """Add a new observation (t, f, df, simga2_f, sigma2_df) to the GP.

    This stores the observation internally, but does NOT yet set up and invert
    the Gram matrix. Add observations with repeated calls to this method, then
    call ``gp.update()`` to set up and invert the Gram matrix. Only then you
    can perform inference (calls to ``gp.mu(t)``, ``gp.V(t)``, etc...)."""

    assert isinstance(t, (float, np.float32, np.float64))
    assert isinstance(f, (float, np.float32, np.float64))
    assert isinstance(df, (float, np.float32, np.float64))
    assert isinstance(fvar, (float, np.float32, np.float64))
    assert isinstance(dfvar, (float, np.float32, np.float64))

    self.ready = False
    self.min_obs = None

    self.N += 1
    self.ts.append(t)
    self.fs.append(f)
    self.dfs.append(df)
    self.fvars.append(fvar)
    self.dfvars.append(dfvar)

  def update(self):
    """Set up the Gram matrix and compute its LU decomposition to make the GP
    ready for inference (calls to ``.gp.mu(t)``, ``gp.V(t)``, etc...).

    Call this method after you have manipulated the GP by
       - ``gp.reset()`` ing,
       - adding observations with ``gp.add(t, f, df)``, or
       - adjusting the sigmas via ``gp.update_sigmas()``.
    and want to perform inference next."""

    if self.ready:
      return

    # Set up the kernel matrices.
    self.K = np.matrix(np.zeros([self.N, self.N]))
    self.Kd = np.matrix(np.zeros([self.N, self.N]))
    self.dKd = np.matrix(np.zeros([self.N, self.N]))
    for i in range(self.N):
      for j in range(self.N):
        self.K[i, j] = self.k(self.ts[i], self.ts[j])
        self.Kd[i, j] = self.kd(self.ts[i], self.ts[j])
        self.dKd[i, j] = self.dkd(self.ts[i], self.ts[j])

    # Put together the Gram matrix
    S_f = np.matrix(np.diag(self.fvars))
    S_df = np.matrix(np.diag(self.dfvars))
    self.G = np.bmat([[self.K + S_f, self.Kd],
                      [self.Kd.T, self.dKd + S_df]])

    # Compute the LU decomposition of G and store it
    self.LU, self.LU_piv = linalg.lu_factor(self.G, check_finite=True)

    # Set ready switch to True
    self.ready = True

    # Pre-compute the regression weights used in mu
    self.w = self.solve_G(np.array(self.fs + self.dfs))

  def solve_G(self, b):
    """Solve ``Gx=b`` where ``G`` is the Gram matrix of the GP.

    Uses the internally-stored LU decomposition of ``G`` computed in
    ``gp.update()``."""

    assert self.ready
    return linalg.lu_solve((self.LU, self.LU_piv), b, check_finite=True)

  def mu(self, t):
    """Evaluate posterior mean of f at ``t``."""

    assert isinstance(t, (float, np.float32, np.float64))
    assert self.ready

    # Compute kernel vector (k and kd) of the query t and the observations T
    # Then perform inner product with the pre-computed GP weights
    T = np.array(self.ts)
    kvec = np.concatenate([self.k(t, T), self.kd(t, T)])

    return np.dot(self.w, kvec)

  def dmu(self, t):
    """Evaluate first derivative of the posterior mean of df at ``t``."""

    assert isinstance(t, (float, np.float32, np.float64))
    assert self.ready

    # Same is in mu, with the respective "derivative kernel vectors"
    T = np.array(self.ts)
    kvec = np.concatenate([self.kd(T, t), self.dkd(t, T)])

    return np.dot(self.w, kvec)

  def d2mu(self, t):
    """Evaluate 2nd derivative of the posterior mean of f at ``t``."""

    assert isinstance(t, (float, np.float32, np.float64))
    assert self.ready

    # Same is in mu, with the respective "derivative kernel vectors"
    T = np.array(self.ts)
    kvec = np.concatenate([self.d2k(t, T), self.d2kd(t, T)])

    return np.dot(self.w, kvec)

  def d3mu(self, t):
    """Evaluate 3rd derivative of the posterior mean of f at ``t``."""

    assert isinstance(t, (float, np.float32, np.float64))
    assert self.ready

    # Same is in mu, with the respective "derivative kernel vectors"
    T = np.array(self.ts)
    kvec = np.concatenate([self.d3k(t, T), np.zeros(self.N)])

    return np.dot(self.w, kvec)

  def V(self, t):
    """Evaluate posterior variance of f at ``t``."""

    assert isinstance(t, (float, np.float32, np.float64))
    assert self.ready

    # Compute the needed k vector
    T = np.array(self.ts)
    kvec = np.concatenate([self.k(t, T), self.kd(t,T)])
    ktt = self.k(t, t)

    return ktt - np.dot(kvec, self.solve_G(kvec))

  def Vd(self, t):
    """Evaluate posterior co-variance of f and df at ``t``."""

    assert isinstance(t, (float, np.float32, np.float64))
    assert self.ready

    T = np.array(self.ts)
    ktT = self.k(t, T)
    kdtT = self.kd(t, T)
    dktT = self.kd(T, t)
    dkdtT = self.dkd(t, T)
    kdtt = self.kd(t, t)
    kvec_a = np.concatenate([ktT, kdtT])
    kvec_b = np.concatenate([dktT, dkdtT])

    return kdtt - np.dot(kvec_a, self.solve_G(kvec_b))

  def dVd(self, t):
    """Evaluate posterior variance of df at ``t``"""

    assert isinstance(t, (float, np.float32, np.float64))
    assert self.ready

    T = np.array(self.ts)
    dkdtt = self.dkd(t, t)
    dktT = self.kd(T, t)
    dkdtT = self.dkd(t, T)
    kvec = np.concatenate([dktT, dkdtT])

    return dkdtt - np.dot(kvec, self.solve_G(kvec))

  def Cov_0(self, t):
    """Evaluate posterior co-variance of f at 0. and ``t``."""

    assert isinstance(t, (float, np.float32, np.float64))
    assert self.ready

    T = np.array(self.ts)
    k0t = self.k(0., t)
    k0T = self.k(0., T)
    kd0T = self.kd(0., T)
    ktT = self.k(t, T)
    kdtT = self.kd(t, T)
    kvec_a = np.concatenate([k0T, kd0T])
    kvec_b = np.concatenate([ktT, kdtT])

    return k0t - np.dot(kvec_a, self.solve_G(kvec_b))

  def Covd_0(self, t):
    """Evaluate posterior co-variance of f at 0. and df at ``t``."""
    # !!! I changed this in line_search new, Covd_0 <-> dCov_0

    assert isinstance(t, (float, np.float32, np.float64))
    assert self.ready

    T = np.array(self.ts)
    kd0t = self.kd(0., t)
    k0T = self.k(0., T)
    kd0T = self.kd(0., T)
    dktT = self.kd(T, t)
    dkdtT = self.dkd(t, T)
    kvec_a = np.concatenate([k0T, kd0T])
    kvec_b = np.concatenate([dktT, dkdtT])

    return kd0t - np.dot(kvec_a, self.solve_G(kvec_b))

  def dCov_0(self, t):
    """Evaluate posterior co-variance of df at 0. and f at ``t``."""
    # !!! I changed this in line_search new, Covd_0 <-> dCov_0

    assert isinstance(t, (float, np.float32, np.float64))
    assert self.ready

    T = np.array(self.ts)
    dk0t = self.kd(t, 0.)
    dk0T = self.kd(T, 0.)
    dkd0T = self.dkd(0., T)
    ktT = self.k(t, T)
    kdtT = self.kd(t, T)
    kvec_a = np.concatenate([dk0T, dkd0T])
    kvec_b = np.concatenate([ktT, kdtT])

    return dk0t - np.dot(kvec_a, self.solve_G(kvec_b))

  def dCovd_0(self, t):
    """Evaluate posterior co-variance of df at 0. and ``t``."""

    assert isinstance(t, (float, np.float32, np.float64))
    assert self.ready

    T = np.array(self.ts)
    dkd0t = self.dkd(0., t)
    dk0T = self.kd(T, 0.)
    dkd0T = self.dkd(0., T)
    dktT = self.kd(T, t)
    dkdtT = self.dkd(t, T)
    kvec_a = np.concatenate([dk0T, dkd0T])
    kvec_b = np.concatenate([dktT, dkdtT])

    return dkd0t - np.dot(kvec_a, self.solve_G(kvec_b))

  def cubic_polynomial_coefficients(self, t):
    """The posterior mean ``mu`` of this GP is piece-wise cubic. Return the
    coefficients of the cubic polynomial that is ``mu`` at ``t``."""

    assert isinstance(t, (float, np.float32, np.float64))
    assert t not in self.ts # at the observations, polynomial is ambiguous

    d1, d2, d3 = self.dmu(t), self.d2mu(t), self.d3mu(t)
    a = d3/6.0
    b = 0.5*d2-3*a*t
    c = d1-3*a*t**2-2*b*t
    d = self.mu(t)-a*t**3-b*t**2-c*t

    return (a, b, c, d)

  def quadratic_polynomial_coefficients(self, t):
    """The posterior mean ``mu`` of this GP is piece-wise cubic. Return the
    coefficients of the **quadratic** polynomial that is the **derivative** of
    ``mu`` at ``t``.

    This is used to find the minimum of the cubic polynomial in
    ``gp.find_mimima()``."""

    assert isinstance(t, (float, np.float32, np.float64))
    assert t not in self.ts # at the observations, polynomial is ambiguous

    d1, d2, d3 = self.dmu(t), self.d2mu(t), self.d3mu(t)
    a = .5*d3
    b = d2 - d3*t
    c = d1 - d2*t + 0.5*d3*t**2

    return (a, b, c)

  def find_dmu_equal(self, val):
    """Finds points where the derivative of the posterior mean equals ``val``
    and the second derivative is positive.

    The posterior mean is a  cubic polynomial in each of the cells"
    ``[t_i, t_i+1]`` where the t_i are the sorted observed ts. For each of
    these cells, returns points with dmu==val the cubic polynomial if it exists
    and happens to lie in that cell."""

    # We want to go through the observations from smallest to largest t
    ts_sorted = list(self.ts)
    ts_sorted.sort()

    solutions = []

    for t1, t2 in zip(ts_sorted, ts_sorted[1:]):
      # Compute the coefficients of the quadratic polynomial dmu/dt in this
      # cell, then call the function minimize_cubic to find the minimizer.
      # If there is one and it falls into the current cell, store it
      a, b, c = self.quadratic_polynomial_coefficients(t1+0.5*(t2-t1))
      solutions_cell = quadratic_polynomial_solve(a, b, c, val)
      for s in solutions_cell:
        if s>t1 and s<t2:
          solutions.append(s)

    return solutions

  def find_cubic_minima(self):
    """Find the local minimizers of the posterior mean.

    The posterior mean is a  cubic polynomial in each of the cells"
    [t_i, t_i+1] where the t_i are the sorted observed ts. For each of these
    cells, return the minimizer of the cubic polynomial if it exists and
    happens to lie in that cell."""

    return self.find_dmu_equal(0.0)

  def expected_improvement(self, t):
    """Computes the expected improvement at position ``t`` under the current
    GP model.

    Reference "current best" is the observed ``t`` with minimal posterior
    mean."""

    assert isinstance(t, (float, np.float32, np.float64))

    # Find the observation with minimal posterior mean, if it has not yet been
    # computed by a previous call to this method
    if self.min_obs is None:
      self.min_obs = min(self.mu(tt) for tt in self.ts)

    # Compute posterior mean and variance at t
    m, v = self.mu(t), self.V(t)

    # Compute the two terms in the formula for EI and return the sum
    t1 = 0.5 * (self.min_obs-m) * (1 + erf((self.min_obs-m)/np.sqrt(2.*v)))
    t2 = np.sqrt(0.5*v/np.pi) * np.exp(-0.5*(self.min_obs-m)**2/v)

    return t1 + t2

  def k(self, x, y):
    """Kernel function."""
    for arg in [x, y]:
      assert isinstance(arg, (float, np.float32, np.float64)) or \
             (isinstance(arg, np.ndarray) and np.linalg.matrix_rank(arg) == 1)
    mi = self.offset + np.minimum(x, y)
    return self.theta**2 * (mi**3/3.0 + 0.5*np.abs(x-y)*mi**2)

  def kd(self, x, y):
    """Derivative of kernel function, 1st derivative w.r.t. right argument."""
    for arg in [x, y]:
      assert isinstance(arg, (float, np.float32, np.float64)) or \
             (isinstance(arg, np.ndarray) and np.linalg.matrix_rank(arg) == 1)
    xx = x + self.offset
    yy = y + self.offset
    return self.theta**2 * np.where(x<y, 0.5*xx**2, xx*yy-0.5*yy**2)

  def dkd(self, x, y):
    """Derivative of kernel function,  1st derivative w.r.t. both arguments."""
    for arg in [x, y]:
      assert isinstance(arg, (float, np.float32, np.float64)) or \
             (isinstance(arg, np.ndarray) and np.linalg.matrix_rank(arg) == 1)
    xx = x+self.offset
    yy = y+self.offset
    return self.theta**2 * np.minimum(xx, yy)

  def d2k(self, x, y):
    """Derivative of kernel function,  2nd derivative w.r.t. left argument."""
    for arg in [x, y]:
      assert isinstance(arg, (float, np.float32, np.float64)) or \
             (isinstance(arg, np.ndarray) and np.linalg.matrix_rank(arg) == 1)
    return self.theta**2 * np.where(x<y, y-x, 0.)

  def d3k(self, x, y):
    """Derivative of kernel function,  3rd derivative w.r.t. left argument."""
    for arg in [x, y]:
      assert isinstance(arg, (float, np.float32, np.float64)) or \
             (isinstance(arg, np.ndarray) and np.linalg.matrix_rank(arg) == 1)
    return self.theta**2 * np.where(x<y, -1., 0.)

  def d2kd(self, x, y):
    """Derivative of kernel function,  2nd derivative w.r.t. left argument,
    1st derivative w.r.t. right argument."""
    for arg in [x, y]:
      assert isinstance(arg, (float, np.float32, np.float64)) or \
             (isinstance(arg, np.ndarray) and np.linalg.matrix_rank(arg) == 1)
    return self.theta**2 * np.where(x<y, 1., 0.)

  # ToDo: Commenting
  def visualize_f(self, ax):
    """Visualize the GP: function value.

    ``ax`` is a matplotlib axis."""

    a, b = min(self.ts), max(self.ts)
    lo = a - .05*(b-a)
    up = b + (b-a)
    span = up - lo
    w = span/40.
    tt = np.linspace(lo, up, num=1000)
    m = np.array([self.mu(t) for t in tt])
    v = np.array([self.V(t) for t in tt])
    ax.hold(True)
    for t, f, df in zip(self.ts, self.fs, self.dfs):
      ax.plot(t, f, marker='o', markersize=8, color=[0., 0.4717, 0.4604])
      ax.plot([t-w, t+w], [f-w*df, f+w*df], 'black')
    ax.plot(tt, m, color=[1., 0.6, 0.2], linewidth=1.5)
    ax.plot(tt, m + 2*np.sqrt(v), color=[1., 0.6, 0.2], linestyle='--')
    ax.plot(tt, m - 2*np.sqrt(v), color=[1., 0.6, 0.2], linestyle='--')
    ax.plot([lo, up], [0., 0.], color="grey", linestyle=":")
    ax.set_xlim(lo, up)

  # ToDo: Commenting
  def visualize_df(self, ax):
    """Visualize the GP: derivative.

    ``ax`` is a matplotlib axis."""

    a, b = min(self.ts), max(self.ts)
    lo = a - .05*(b-a)
    up = b + (b-a)
    tt = np.linspace(lo, up, num=1000)
    m = np.array([self.dmu(t) for t in tt])
    v = np.array([self.dVd(t) for t in tt])
    ax.hold(True)
    ax.plot(self.ts, self.dfs, 'o', markersize=8, color=[0., 0.4717, 0.4604])
    ax.plot(tt, m, color=[1., 0.6, 0.2], linewidth=1.5)
    ax.plot(tt, m + 2*np.sqrt(v), color=[1., 0.6, 0.2], linestyle='--')
    ax.plot(tt, m - 2*np.sqrt(v), color=[1., 0.6, 0.2], linestyle='--')
    ax.plot([lo, up], [0., 0.], color="grey", linestyle=":")
    ax.set_xlim(lo, up)

  # ToDo: Commenting
  def visualize_ei(self, ax):
    """Visualize expected improvement.

    ``ax`` is a matplotlib axis."""

    a, b = min(self.ts), max(self.ts)
    lo = a - .05*(b-a)
    up = b + (b-a)
    tt = np.linspace(lo, up, num=1000)
    ax.plot(tt, [self.expected_improvement(t) for t in tt])
    ax.set_xlim(lo, up)


def quadratic_polynomial_solve(a, b, c, val):
  """Computes *real* solutions of f(t) = a*t**2 + b*t + c = val with f''(t)>0.

  Returns the *list* of solutions (containg 1 or 0 solutions)."""

  assert isinstance(a, (float, np.float32, np.float64))
  assert isinstance(b, (float, np.float32, np.float64))
  assert isinstance(c, (float, np.float32, np.float64))
  assert isinstance(val, (float, np.float32, np.float64))

  # Check if a is almost zero. If so, solve the remaining linear equation. Note
  # that we return only soultions with f''(t) = b > 0
  if abs(a) < 1e-9:
    if b > 1e-9:
      return [(val-c)/b]
    else:
      return []

  # Compute the term under the square root in pq formula, if it is negative,
  # there is no real solution
  det = b**2-4.*a*(c-val)
  if det < 0:
    return []

  # Otherwise, compute the two roots
  s = np.sqrt(det)
  r1 = (-b - np.sign(a)*s)/(2.*a)
  r2 = (-b + np.sign(a)*s)/(2.*a)

  # Return the one with f''(t) = 2at + b > 0, or []
  if 2*a*r1+b > 0:
    return [r1]
  elif 2*a*r2+b > 0:
    return [r2]
  else:
    return []
