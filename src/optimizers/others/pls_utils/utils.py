# -*- coding: utf-8 -*-
"""
Utility functions for probabilistic line search algorithm.
"""

import numpy as np
from scipy.special import erf

def bounded_bivariate_normal_integral(rho, xl, xu, yl, yu):
  """Computes the bounded bivariate normal integral.
  
  Computes the probability that ``xu >= X >= xl and yu >= Y >= yl`` where X
  and Y are jointly Gaussian random variables, with mean ``[0., 0.]`` and
  covariance matrix ``[[1., rho], [rho, 1.]]``.

  Inputs:
      :rho: Correlation coefficient of the bivariate normal random variable
      :xl, yl: Lower bounds of the integral
      :xu, yu: Upper bounds of the integral
  
  Ported from a Matlab implementation by Alan Genz which, in turn, is based on
  the method described by
      Drezner, Z and G.O. Wesolowsky, (1989),
      On the computation of the bivariate normal inegral,
      Journal of Statist. Comput. Simul. 35, pp. 101-107,
  
  Copyright statement of Alan Genz's version:
  ***************
  Copyright (C) 2013, Alan Genz,  All rights reserved.               

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided the following conditions are met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in 
      the documentation and/or other materials provided with the 
      distribution.
    - The contributor name(s) may not be used to endorse or promote 
      products derived from this software without specific prior 
      written permission.
  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT 
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS 
  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE 
  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, 
  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, 
  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS 
  OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND 
  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR 
  TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."""
  
  bvnu = unbounded_bivariate_normal_integral
  p = bvnu(rho, xl, yl) - bvnu(rho, xu, yl) \
      - bvnu(rho, xl, yu) + bvnu(rho, xu, yu)
  return max(0., min(p, 1.))

def unbounded_bivariate_normal_integral(rho, xl, yl):
  """Computes the unbounded bivariate normal integral.
  
  Computes the probability that ``X>=xl and Y>=yl`` where X and Y are jointly
  Gaussian random variables, with mean ``[0., 0.]`` and covariance matrix
  ``[[1., rho], [rho, 1.]]``.
  
  Note: to compute the probability that ``X < xl and Y < yl``, use
  ``unbounded_bivariate_normal_integral(rho, -xl, -yl)``. 

  Inputs:
      :rho: Correlation coefficient of the bivariate normal random variable
      :xl, yl: Lower bounds of the integral
  
  Ported from a Matlab implementation by Alan Genz which, in turn, is based on
  the method described by
      Drezner, Z and G.O. Wesolowsky, (1989),
      On the computation of the bivariate normal inegral,
      Journal of Statist. Comput. Simul. 35, pp. 101-107,
  
  Copyright statement of Alan Genz's version:
  ***************
  Copyright (C) 2013, Alan Genz,  All rights reserved.               

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided the following conditions are met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in 
      the documentation and/or other materials provided with the 
      distribution.
    - The contributor name(s) may not be used to endorse or promote 
      products derived from this software without specific prior 
      written permission.
  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT 
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS 
  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE 
  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, 
  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, 
  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS 
  OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND 
  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR 
  TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."""
  
  rho = max(-1., min(1., rho))

  if np.isposinf(xl) or np.isposinf(yl):
    return 0.
  elif np.isneginf(xl):
    return 1. if np.isneginf(yl) else _cdf(-yl)
  elif np.isneginf(yl):
    return _cdf(-xl)
  elif rho == 0:
    return _cdf(-xl)*_cdf(-yl)
  
  tp = 2.*np.pi
  h, k = xl, yl
  hk = h*k
  bvn = 0.
  
  if np.abs(rho) < 0.3:
    # Gauss Legendre points and weights, n =  6
    w = np.array([0.1713244923791705, 0.3607615730481384, 0.4679139345726904])
    x = np.array([0.9324695142031522, 0.6612093864662647, 0.2386191860831970])
  elif np.abs(rho) < 0.75:
    # Gauss Legendre points and weights, n = 12
    w = np.array([0.04717533638651177, 0.1069393259953183, 0.1600783285433464,
                  0.2031674267230659, 0.2334925365383547, 0.2491470458134029])
    x = np.array([0.9815606342467191, 0.9041172563704750, 0.7699026741943050,
                  0.5873179542866171, 0.3678314989981802, 0.1252334085114692])
  else:
    # Gauss Legendre points and weights, n = 20
    w = np.array([.01761400713915212, .04060142980038694, .06267204833410906,
                  .08327674157670475, 0.1019301198172404, 0.1181945319615184,
                  0.1316886384491766, 0.1420961093183821, 0.1491729864726037,
                  0.1527533871307259])
    x = np.array([0.9931285991850949, 0.9639719272779138, 0.9122344282513259,
                  0.8391169718222188, 0.7463319064601508, 0.6360536807265150,
                  0.5108670019508271, 0.3737060887154196, 0.2277858511416451,
                  0.07652652113349733])
  
  w = np.tile(w, 2)
  x = np.concatenate([1.-x, 1.+x])
  
  if np.abs(rho) < 0.925:
    hs = .5 * (h*h + k*k)
    asr = .5*np.arcsin(rho)
    sn = np.sin(asr*x)
    bvn = np.dot(w, np.exp((sn*hk-hs)/(1.-sn**2)))
    bvn = bvn*asr/tp + _cdf(-h)*_cdf(-k) 
  else:
    if rho < 0.:
      k = -k
      hk = -hk
    if np.abs(rho) < 1.:
      ass = 1.-rho**2
      a = np.sqrt(ass)
      bs = (h-k)**2
      asr = -.5*(bs/ass + hk)
      c = (4.-hk)/8.
      d = (12.-hk)/80. 
      if asr > -100.:
        bvn = a*np.exp(asr)*(1.-c*(bs-ass)*(1.-d*bs)/3. + c*d*ass**2)
      if hk  > -100.:
        b = np.sqrt(bs)
        sp = np.sqrt(tp)*_cdf(-b/a)
        bvn = bvn - np.exp(-.5*hk)*sp*b*(1. - c*bs*(1.-d*bs)/3.)
      a = .5*a
      xs = (a*x)**2
      asr = -.5*(bs/xs + hk)
      inds = [i for i, asr_elt in enumerate(asr) if asr_elt>-100.]
      xs = xs[inds]
      sp = 1. + c*xs*(1.+5.*d*xs)
      rs = np.sqrt(1.-xs)
      ep = np.exp(-.5*hk*xs / (1.+rs)**2)/rs
      bvn = (a*np.dot(np.exp(asr[inds])*(sp-ep), w[inds]) - bvn)/tp
    if rho > 0:
      bvn +=  _cdf(-max(h, k)) 
    elif h >= k:
      bvn = -bvn
    else:
      if h < 0.:
        L = _cdf(k)-_cdf(h)
      else:
        L = _cdf(-h)-_cdf(-k)
      bvn =  L - bvn
  
  return max(0., min(1., bvn))

def _cdf(z):
  """Cumulative density function (CDF) of the standard normal distribution."""
  return .5 * (1. + erf(z/np.sqrt(2.)))