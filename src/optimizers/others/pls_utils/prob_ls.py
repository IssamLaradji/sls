# -*- coding: utf-8 -*-
"""
Implementation of Probabilistic Line Search for Stochastic Optimization [1].

[1] M. Mahsereci and P. Hennig. Probabilistic line searches for stochastic
optimization. In Advances in Neural Information Processing Systems 28, pages
181-189, 2015.

Modified by Aaron Mishkin (amishkin@cs.ubc.ca)
"""

import numpy as np
import others.pls_utils.gaussian_process as gaussian_process
import others.pls_utils.utils as utils

class ProbLSOptimizer(object):
	"""Probabilistic line search optimizer.

	@@__init__
	"""

	def __init__(self, c1=0.05, cW=0.3, fpush=1.0, alpha0=0.01,
							 target_df=0.5, df_lo=-0.1, df_hi=1.1, max_steps=10, max_expl=6,
							 max_dmu0=0.0, max_change_factor=10.0, expl_policy="linear", verbose=False):
		"""Create a new probabilistic line search object.

		Inputs:
			:c1: Scalar parameters for the first Wolfe conditions. Default to 0.05.
			:cW: Acceptance threshold for the Wolfe probability. Defaults to 0.3.
			:fpush: Push factor that is multiplied with the accepted step size to get
					the base step size for the next line search. Defaults to 1.0.
			:alpha0: Initial step size. Defaults to 0.01.
			:target_df: The target value for the relative projected gradient
					df(t)/abs(df(0)). Defaults to 0.5.
			:df_lo, df_hi: Lower and higher threshold for the relative projected
					gradient df(t)/abs(df(0)). Default to -0.1 and 1.1.
			:max_steps: Maximum number of steps (function evaluations) per line
					search. Defaults to 10.
			:max_epl: Maximum number of exploration steps per line search. Defaults
					to 6.
			:max_dmu0: If the posterior derivative at t=0 exceeds ``max_dmu0``, the
					current line search is aborted as a safeguard against bad search
					directions. Defaults to 0.0.
			:max_change_factor: The algorithm usually takes the accepted alpha of the
					current line search as the base ``alpha0`` of the next one (after
					multiplying with ``fpush``). However, if a line search accepts an
					alpha that is more than ``max_change_factor`` times smaller or larger
					than the current ``alpha0``, we instead set the next ``alpha0`` to a
					running average of the accepted alphas (``alpha_stats``). Defaults to
					10.0.
			:expl_policy: String indicating the policy used for exploring points *to
					the right* in the line search. If ``k`` is the number of exploration
					steps already made, then the ``"linear"`` exploration policy chooses
					``2*(k+1)*alpha0`` as the next exploration candidate. The
					``"exponential"`` policy chooses ``2**(k+1)*alpha0``. Defaults to
					``"linear"``."""

		# Note: `func` parameter removed and is now set via the `set_func` method.

		# Store the line search parameters
		self.c1 = c1
		self.cW = cW
		self.fpush = fpush
		self.target_df = target_df
		self.df_lo = df_lo
		self.df_hi = df_hi
		self.max_steps = max_steps
		self.max_expl = max_expl
		self.max_dmu0 = max_dmu0
		self.max_change_factor = max_change_factor
		assert expl_policy in ["linear", "exponential"]
		self.expl_policy = expl_policy

		# Initialize base step size with given value.
		self.alpha0 = alpha0

		# alpha_stats will contain a running average of accepted step sizes
		self.alpha_stats = alpha0

		# Raw function values at the origin of the line search
		self.f0 = None
		self.df0 = None

		# Counting steps in the current line search and, separately, steps that
		# explore "to the right"
		self.num_steps = 0
		self.num_expl = 0

		# Initialize GP object
		self.gp = gaussian_process.ProbLSGaussianProcess()

		# Switch to assert that the prepare method will be called first
		self.prepare_called = False

		# Internal abort status
		self.abort_status = 0

		self.verbose = verbose

	def scale_obs(self, f_raw, df_raw, fvar_raw, dfvar_raw):
		"""Scale an observation of function value and gradient. See section 3.4 of
		[1] for details."""

		f = (f_raw-self.f0)/(self.df0*self.alpha0)
		df = df_raw/(self.df0)
		fvar = fvar_raw/((self.alpha0*self.df0)**2)
		dfvar = dfvar_raw/(self.df0**2)
		return f, df, fvar, dfvar

	def rescale_t(self, t):
		"""Rescale a step size used internally by multiplying with the base step
		size."""

		return t*self.alpha0

	def rescale_obs(self, f, df, fvar, dfvar):
		"""Rescale an observation to real-world scale."""

		f_raw = f*self.df0*self.alpha0 + self.f0
		df_raw = df*self.df0
		fvar_raw = fvar*(self.alpha0*self.df0)**2
		dfvar_raw = dfvar*self.df0**2
		return f_raw, df_raw, fvar_raw, dfvar_raw

	def prepare(self, f_raw, df_raw, fvar_raw, dfvar_raw):
		"""Preparation.

		*pass_to_func_args are arguments that are passed to the function interface,
		e.g. a feed dict."""

		self.f0 = f_raw
		self.df0 = np.abs(df_raw)

		# Add the first observation to the gp
		f, df, fvar, dfvar = self.scale_obs(f_raw, df_raw, fvar_raw, dfvar_raw)
		self.gp.add(0.0, f, df, fvar, dfvar)

		# Set flag that the prepare method has been called
		self.prepare_called = True

	def accept(self, accept_func):
		"""Accept the most recent step size."""

		assert self.abort_status != 1
		assert self.num_steps >= 1

		# Rescale to the "real-world" step size alpha
		alpha = self.rescale_t(self.gp.ts[-1])

		# If this accept was not due to an abort and the step size did not change
		# *too much*, we use the accepted alpha as the new base step size alpha0
		# (and update a running average alpha_stats). Otherwise, we use said
		# running average as the new base step size.
		f = self.max_change_factor
		if self.abort_status == 0 and self.alpha0/f < alpha < self.alpha0*f:
			self.alpha_stats = 0.95*self.alpha_stats + 0.05*alpha
			self.alpha0 = self.fpush*alpha
		else:
			self.alpha0 = self.alpha_stats

		# Reset abort status and counters
		self.abort_status = 0
		self.num_steps = 0
		self.num_expl = 0

		# Run accept op, reset f0 and df0
		f_raw, df_raw, fvar_raw, dfvar_raw = accept_func()
		self.f0 = f_raw
		self.df0 = np.abs(df_raw)

		# Reset the gp and add the first observation to the gp
		self.gp.reset()
		f, df, fvar, dfvar = self.scale_obs(f_raw, df_raw, fvar_raw, dfvar_raw)
		self.gp.add(0.0, f, df, fvar, dfvar)

	def evaluate(self, t, adv_eval_func):
		"""Evaluate at step size ``t``.

		*pass_to_func_args are arguments that are passed to the function interface,
		e.g. a feed dict."""

		assert self.prepare_called

		self.num_steps += 1

		# Call the adv_eval method of the function interface with the increment
		# re-scaled to the "real-world" step size
		dt = t-self.gp.ts[-1]
		dalpha = self.rescale_t(dt)

		f_raw, df_raw, fvar_raw, dfvar_raw = adv_eval_func(dalpha)

		# Safeguard against inf or nan encounters. Trigerring abort.
		if np.isnan(f_raw) or np.isinf(f_raw) or np.isnan(df_raw) or np.isinf(df_raw):
			f_raw = 100.0
			df_raw = 10.0
			self.abort_status = 1

		# Scale the observations, add it to the GP and update the GP
		# We are currently using the variance estimates from t=0 for all
		# observations, but this might change in the future
		f, df, fvar, dfvar = self.scale_obs(f_raw, df_raw, fvar_raw, dfvar_raw)
		fvar = self.gp.fvars[0]
		dfvar = self.gp.dfvars[0]
		self.gp.add(t, f, df, fvar, dfvar)
		self.gp.update()

	def find_next_t(self):
		"""Find the step size for the next evaluation."""

		assert self.num_steps >= 1

		# Generate candidates: the points where the derivative of the posterior
		# mean equals the target value plus one exploration point to the right.
		candidates = self.gp.find_dmu_equal(self.target_df)
		if self.expl_policy == "linear":
			candidates.append(2.*(self.num_expl+1))
		elif self.expl_policy == "exponential":
			candidates.append(2.**(self.num_expl+1))
		else:
			raise Exception("Unknown exploration policy")
		if self.verbose: print( "\t * Computing utilities for candidates %s", candidates)

		# Compute p_Wolfe for candidates
		pws = [self.compute_p_wolfe(t) for t in candidates]
		if self.verbose: print( "\t * p_Wolfe:", pws)
		ind_best = np.argmax(pws)

		# Memorize when we have chosen the exploration point
		if ind_best == len(candidates) - 1:
				self.num_expl += 1

		# Return the candidate t with maximal utility
		if self.verbose: print( "\t * Best candidate is", candidates[ind_best], "(was candidate", ind_best, "/", len(candidates)-1, ")")
		return candidates[ind_best]

	def find_abort_t(self):
		"""Find the step size to use for an abort."""

		return 0.01
# We are currently simply aborting with a very small step, but we might do
# something like this:
#    ts = self.gp.ts
#    pws = [self.compute_p_wolfe(t) for t in ts]
#    if max(pws) > 0.5*self.cW:
#      t = ts[np.argmax(pws)]
#    else:
#      t = 0.0
#    offset = 0.01
#
#    return t + offset

	def compute_p_wolfe(self, t):
		# Already changed dCov and Covd here
		"""Computes the probability that step size ``t`` satisfies the adjusted
		Wolfe conditions under the current GP model."""

		# Compute mean and covariance matrix of the two Wolfe quantities a and b
		# (equations (11) to (13) in [1]).
		mu0 = self.gp.mu(0.)
		dmu0 = self.gp.dmu(0.)
		mu = self.gp.mu(t)
		dmu = self.gp.dmu(t)
		V0 = self.gp.V(0.)
		Vd0 = self.gp.Vd(0.)
		dVd0 = self.gp.dVd(0.)
		dCov0t = self.gp.dCov_0(t)
		Covd0t = self.gp.Covd_0(t)

		ma = mu0 - mu + self.c1*t*dmu0
		Vaa = V0 + dVd0*(self.c1*t)**2 + self.gp.V(t) \
					+ 2.*self.c1*t*(Vd0 - dCov0t) - 2.*self.gp.Cov_0(t)
		mb = dmu
		Vbb = self.gp.dVd(t)

		# Very small variances can cause numerical problems. Safeguard against
		# this with a deterministic evaluation of the Wolfe conditions.
		if Vaa < 1e-9 or Vbb < 1e-9:
			return 1. if ma>=0. and mb>=0. else 0.

		Vab = Covd0t + self.c1*t*self.gp.dCovd_0(t) - self.gp.Vd(t)

		# Compute correlation factor and integration bounds for adjusted p_Wolfe
		# and return the result of the bivariate normal integral.
		rho = Vab/np.sqrt(Vaa*Vbb)
		al = -ma/np.sqrt(Vaa)
		bl = (self.df_lo - mb)/np.sqrt(Vbb)
		bu = (self.df_hi - mb)/np.sqrt(Vbb)
		return utils.bounded_bivariate_normal_integral(rho, al, np.inf, bl, bu)

	def check_for_acceptance(self):
		"""Checks whether the most recent point should be accepted."""

		# Return False when no evaluations t>0 have been made yet
		if self.num_steps == 0:
			return False

		# If an abort has been triggered, return True
		if self.abort_status == 2:
			return True

		# Check Wolfe probability
		pW = self.compute_p_wolfe(self.gp.ts[-1])
		if pW >= self.cW:
			return True
		else:
			return False

	def proceed(self, accept_func, adv_eval_func):
		"""Make one step (function evaluation) in the line search.

		*pass_to_func_args are arguments that are passed to the function interface,
		e.g. a feed dict."""
		# Is the line-search complete?
		complete = False
		assert self.prepare_called

		# Check for acceptance and accept the previous point as the case may be
		if self.check_for_acceptance():
			if self.verbose: print( "-> ACCEPT")
			if self.verbose: print( "\t * alpha = ", self.rescale_t(self.gp.ts[-1]), "[alpha0 was", self.alpha0, "]")
			self.accept(accept_func)
			if self.verbose: print( "\t * f = ", self.f0)
			# iterated accepted; line-search is over.
			complete = True

		# In the first call to proceed in a new line search, evaluate at t=1.
		if self.num_steps == 0:
			if self.verbose: print( "************************************")
			if self.verbose: print( "NEW LINE SEARCH [alpha0 is", self.alpha0, "]")
			if self.verbose: print( "-> First step, evaluating at t = 1.0")
			self.evaluate(1., adv_eval_func)

		# Abort with a very small, safe step size if
		# - Abort triggered in an other method, e.g. evaluate() encountered inf or
		#   nan. (self.abort_status==1)
		# - the maximum number of steps per line search is exceeded
		# - the maximum number of exploration steps is exceeded
		# - the posterior derivative at t=0. is too large (bad search direction)


		elif (self.abort_status == 1
					or self.num_steps >= self.max_steps
					or self.num_expl >= self.max_expl
					or self.gp.dmu(0.) >= self.max_dmu0):

			t_new = self.find_abort_t()
			if self.verbose: print( "-> Aborting with t = ", t_new)
			self.evaluate(t_new, adv_eval_func)
			self.abort_status = 2
			# the line search is over due to abort
			complete = True

		# This is an "ordinary" evaluation. Find the best candidate for the next
		# evaluation and evaluate there.
		else:
			if self.verbose: print( "-> Ordinary step", self.num_steps, ", searching for new t")
			t_new = self.find_next_t()
			if self.verbose: print( "\t * Evaluating at t =", t_new)
			self.evaluate(t_new, adv_eval_func)

		# Return the real-world function value
		f, _, _, _ = self.rescale_obs(self.gp.fs[-1], self.gp.dfs[-1],
																	self.gp.fvars[-1], self.gp.dfvars[-1])
		return f, complete

	def proceed_constant_step(self, alpha, accept_func, adv_eval_func):
		"""Make one step (function evaluation) in the line search.

		*pass_to_func_args are arguments that are passed to the function interface,
		e.g. a feed dict."""

		assert self.prepare_called

		if self.num_steps >= 1:
			self.accept(accept_func)

		if self.verbose: print( "************************************")
		if self.verbose: print( "CONSTANT STEP with alpha =", alpha, "[alpha0 is", self.alpha0, "]")
		t = alpha/self.alpha0
		if self.verbose: print( "-> Evaluating at t =", t)
		self.evaluate(t, adv_eval_func)

		f, _ = self.rescale_obs(self.gp.fs[-1], self.gp.dfs[-1])
		return f

	# ToDo: Commenting
	def visualize_ei_pw(self, ax):
		"""Visualize the current state of the line search: expected improvement
		and p_Wolfe.

		``ax`` is a matplotlib axis."""

		a, b = min(self.gp.ts), max(self.gp.ts)
		lo = a - .05*(b-a)
		up = b + (b-a)
		tt = np.linspace(lo, up, num=1000)
		ei = [self.gp.expected_improvement(t) for t in tt]
		pw = [self.compute_p_wolfe(t) for t in tt]
		prod = [e*p for e, p in zip(ei, pw)]
		ax.hold(True)
		ax.plot(tt, ei, label="EI")
		ax.plot(tt, pw, label="pW")
		ax.plot(tt, prod, label="EI*pW")
		ax.plot([lo, up], [self.cW, self.cW], color="grey")
		ax.text(lo, self.cW, "Acceptance threshold", fontsize=8)
		ax.set_xlim(lo, up)
		ax.legend(fontsize=10)

## LEGACY VERSION OF p_Wolfe #################################################
# Changed dCov and Covd here already!
#  def compute_p_wolfe_original(self, t):
#    """Computes the probability that step size ``t`` satisfies the Wolfe
#    conditions under the current GP model."""
#
#    # Compute mean and covariance matrix of the two Wolfe quantities a and b
#    # (equations (11) to (13) in [1]).
#    mu0 = self.gp.mu(0.)
#    dmu0 = self.gp.dmu(0.)
#    mu = self.gp.mu(t)
#    dmu = self.gp.dmu(t)
#    V0 = self.gp.V(0.)
#    Vd0 = self.gp.Vd(0.)
#    dVd0 = self.gp.dVd(0.)
#    ma = mu0 - mu + self.c1*t*dmu0
#    Vaa = V0 + dVd0*(self.c1*t)**2 + self.gp.V(t) \
#          + 2.*self.c1*t*(Vd0 - self.gp.dCov_0(t)) - 2.*self.gp.Cov_0(t)
#    mb = dmu - self.c2*dmu0
#    Vbb = dVd0*self.c2**2 - 2.*self.c2*self.gp.dCovd_0(t) + self.gp.dVd(t)
#
#    # Very small variances can cause numerical problems. Safeguard against
#    # this with a deterministic evaluation of the Wolfe conditions.
#    if Vaa < 1e-9 or Vbb < 1e-9:
#      return 1. if ma>=0. and mb>=0. else 0.
#
#    Vab = -self.c2*(Vd0 + self.c1*t*dVd0) + self.c2*self.gp.dCov_0(t) \
#          + self.gp.Covd_0(t) + self.c1*t*self.gp.dCovd_0(t) - self.gp.Vd(t)
#
#    # Compute rho and integration bounds for p_Wolfe and return the result of
#    # the bivariate normal integral. Upper limit for b is used when strong
#    # Wolfe conditions are requested (cf. equations (14) to (16)in [1]).
#    rho = Vab/np.sqrt(Vaa*Vbb)
#    al = -ma/np.sqrt(Vaa)
#    bl = -mb/np.sqrt(Vbb)
#    if self.strong_wolfe:
#      bbar = 2.*self.c2*(np.abs(dmu0) + 2.*np.sqrt(dVd0))
#      bu = (bbar - mb)/np.sqrt(Vbb)
#      return utils.bounded_bivariate_normal_integral(rho, al, np.inf, bl, bu)
#    else:
#      return utils.unbounded_bivariate_normal_integral(rho, al, bl)
###############################################################################
