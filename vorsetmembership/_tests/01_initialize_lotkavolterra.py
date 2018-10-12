# -*- coding: utf-8 -*-
"""
@author: Daniel Reyes Lastiri
Set-membership evaluation for Lotka-Volterra
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import odeint
import numpy.random as rnd
from matplotlib.ticker import FormatStrFormatter

from vorsetmembership.initialize import initialize
from vorsetmembership.plots import plot_lines, plot_scatter

# ---- FUNCTION ----
def lotka_volterra(t,y0,u,ut,p, b,d):
    # Parameters. Fixed
    a, c = p['a'], p['c']
    # Differential equation
    def diff(_y0,_t):
        x1,x2 = _y0
        dx1_dt = x1*(a-b*x2)
        dx2_dt = x2*(-c+d*x1)
        return (dx1_dt,dx2_dt)
    # Solve differential equation(s)
    result = odeint(diff,y0,t)
    return result

# ---- ARTIFICIAL DATA ----
p = {'a':1,
     'c':1}
b = 0.01
d = 0.02
y0 = (50,50)
tsim = np.linspace(0,7,num=7*24+1)
ysim = lotka_volterra(tsim,y0,None,None,p,b,d)
idata = np.arange(1*24,7*24,1*24)
tdata = tsim[idata]
ydata = ysim[idata]
# -- Artificial bounds --
errmin1 = np.average(ydata[:,0])/4 * (1 + rnd.random_sample(ydata.shape[0]))
errmax1 = np.average(ydata[:,0])/4 * (1 + rnd.random_sample(ydata.shape[0]))
errmin2 = np.average(ydata[:,1])/4 * (1 + rnd.random_sample(ydata.shape[0]))
errmax2 = np.average(ydata[:,1])/4 * (1 + rnd.random_sample(ydata.shape[0]))

ymin = np.stack(( ydata[:,0] - errmin1, ydata[:,1] - errmin2 ),axis=-1)
ymax = np.stack(( ydata[:,0] + errmax1, ydata[:,1] + errmax2 ),axis=-1)

errmin = np.stack((errmin1,errmin2),axis=-1)
errmax = np.stack((errmax1,errmax2),axis=-1)
errbars = np.stack((errmin,errmax))

# ---- SET MEMBERSHIP ----
b_arr = np.linspace(0.5*b,1.5*b,num=50)
d_arr = np.linspace(0.5*d,1.5*d,num=50)

# Initialize parameter space
p_arr = np.stack((b_arr,d_arr),axis=-1)
init_results = initialize(
    lotka_volterra,tsim,y0,None,None,p,p_arr,
    tdata,ymin,ymax,n_pf_desired=20,max_iter=10,
    vicinity=True)
pf = init_results['pf'] # feasible points
pu = init_results['pu'] # unfeasible points
pfn = init_results['pfn'] # normalized feasible points
pun = init_results['pun'] # normalized unfeasible points
minmax = init_results['norm_minmax'] # min and max used for normalization

# Figure. Bounds, feasible example and original simulation
fig1 = plt.figure(1)
x1 = [tsim,]
y1a = [ysim[:,0],]
y1b = [ysim[:,1],]
ax1a = plt.subplot2grid((8,1), (0, 0), rowspan=3)
ax1a = plot_lines(
    ax1a,x1,y1a, xlabel=r'$t$', ylabel=r'$x_{1}(\theta)$',
    linestyle_list=['-',],
    label_list = [r'$x_{1}$',]
    )
ax1a.errorbar(tdata,ydata[:,0], yerr=errbars[:,:,0], linestyle='',
              capthick=1, elinewidth=1, ecolor='#2d8ce5')
ax1a.set_xlim([0,7])
ax1a.set_ylim([0,100])

ax1b = plt.subplot2grid((8,1), (4, 0), rowspan=3)
ax1b = plot_lines(
    ax1b,x1,y1b, xlabel=r'$t$', ylabel=r'$x_{2}(\theta)$',
    linestyle_list=['-',],
    label_list = [r'$x_{2}$',]
    )
ax1b.errorbar(tdata,ydata[:,1], yerr=errbars[:,:,1], linestyle='', capthick=1,
    elinewidth=1, ecolor='#2d8ce5'
    )

ax1b.set_xlim([0,7])
ax1b.set_ylim([0,200])
fig1.set_size_inches(8.5/2.54 , 7.5/2.54) # Required 1 column width

# Figure. Parameter space
fig2 = plt.figure(2)
ax2 = plt.subplot2grid((4,1), (0, 0), rowspan=3)
ax2 = plot_scatter(
    ax2,[pun,pfn],
    xlabel=r'$b$ (normalized) [{0:.4f}, {1:.4f}]'.format(minmax[0,0],minmax[1,0]),
    ylabel=r'$d$ (normalized) [{0:.4f}, {1:.4f}]'.format(minmax[0,1],minmax[1,1]),
    marker_list=['x','o'],
    label_list = ['Unfeasible','Feasible']
    )

ax2.locator_params(axis='both',nbins=4)
ax2.set_aspect('equal')
#ax2.set_xlim(-2.0,3.0)
#ax2.set_ylim(-2.0,3.0)
ax2.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
fig2.set_size_inches(7.5/2.54 , 7.5/2.54) # Required 1 column width





