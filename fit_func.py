#! /usr/bin/env python
# coding: UTF-8
import numpy as np
from numpy import log, exp
from matplotlib import pyplot as plt
from scipy.stats import binom, poisson, hypergeom, nbinom
from scipy.special import gamma, gammainc, erf, erfc, eval_legendre

'''
関数名が小文字の場合は確率（密度）関数，大文字の場合は相補累積分布関数
---
使い方
p = fit(fit_func, x, y, p0=(, ))
'''

# 二項分布
bi = lambda x, n, p: binom.pmf(x, n, p)
BI = lambda x, n, p: binom.cdf(x, n, p)

# ポアソン分布
po = lambda x, a: poisson.pmf(x, a)
PO = lambda x, a: 1-poisson.cdf(x, a)  # (x+1, x+2,..., の上側累積確率)
po2 = lambda x, a, b: b*poisson.pmf(x, a)

# 幾何分布
ge = lambda x, p: p*(1-p)**(x)

# 超幾何分布
hge = lambda x, M, n, N: hypergeom.pmf(x, M, n, N)

# 負の二項分布
nbi = lambda x, n, p: nbinom.pmf(x, n, p)

# 正規分布
nm = lambda x, a, b: (1/np.sqrt(2*np.pi*b**2)) * np.exp(-(x-a)**2/(2*b**2))
NM = lambda x, a, b: (1-erf((x-a)/np.sqrt(2*b**2)))/2

# 対数正規分布
ln = lambda x, a, b: np.exp(-(np.log(x)-a)**2/(2*b**2))/(np.sqrt(2*np.pi)*b*x)
LN = lambda x, a, b: erfc(a*np.log(x/b))/2
LN2 = lambda x, a, b: erfc((np.log(x)-a)/(np.sqrt(2*b**2)))/2

# 指数分布
ep = lambda x, a, b: np.exp(-x/a) / b
ep2 = lambda x, a: np.exp(-x/a) / a
EP = lambda x, a: np.exp(-x/a)
EP2 = lambda x, a, b: b * np.exp(-x/a)

# ガンマ分布
gm = lambda x, a, b: (x)**(a-1)*np.exp(-x/b)/gamma(a)/b**a
GM = lambda x, a, b: 1-gammainc(a, x/b)
