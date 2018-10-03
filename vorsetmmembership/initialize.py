# -*- coding: utf-8 -*-
"""
@author: Daniel Reyes Lastiri
Module to initialize the parameter space, finding the first set of feasible
points pf_ini.
"""
import numpy as np
import logging

from vorsetmembership import timing

# Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(name)s:%(levelname)s \t%(message)s')
file_handler = logging.FileHandler('report.log', mode='w')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def initialize(function,tsim,x0,u,ut,p,p_arr,tdata,ydata_min,ydata_max,
               n_pf_desired=10,max_iter=10,vicinity=True):
    '''Initialize the parameter space by finding an initial set of 
    fesible points using Latin Hypercube Sampling,
    and normalize to interval [0-1] based on min-max coordinates
    
    Parameters
    ----------
    
    Returns
    -------
    
    '''
    logger.info('---- Calling initialize function ----')
    logger.info('Initializing parameter space by LHCS')
    tstart = timing.start()
    # Since cube, all arrays must have same size
    n_points = p_arr.shape[0]
    n_dim = p_arr.shape[1]
    # Initialize arrays (removing this first row at the end of iterations)
    pf = np.zeros((1,n_dim))
    pu = np.zeros((1,n_dim))
    npf_log = []
    it_log = []
    for it in range(max_iter):
        npf_log.append(pf.shape[0])
        it_log.append(it)
        # -- Latin hypercube sampling --
        # Generate array of indices
        idx_template = np.arange(0,n_points)
        idx_arr = idx_template.copy().reshape(n_points,1)
        for j in range(n_dim)[1:]:
            idx_j = np.random.choice(idx_template,size=n_points,replace=False)
            idx_j = idx_j.reshape(n_points,1)
            idx_arr = np.concatenate((idx_arr,idx_j),axis=1)
        # Generate array of sampled points
        ps_arr = np.zeros((n_points,n_dim))
        for i in range(n_points):
            for j in range(n_dim):
                ps_arr[i,j] = p_arr[idx_arr[i,j],j]
        # -- Evaluate sample --
        # Indexes of tsim matching tdata
        # (does not return the closest value, but the one preceding)
        indexes = np.searchsorted(tsim,tdata)
        # Generate list of feasible and unfeasible indexes
        if_list, iu_list= [], []
        for i,pi in enumerate(ps_arr):
            ysim = function(tsim,x0,u,ut,p,*pi)
            y = ysim[indexes]
            if np.all(ydata_min<=y) and np.all(y<=ydata_max):
                if_list.append(i)
            else:
                iu_list.append(i)
        # Leave iteration if no pf found
        if len(if_list) == 0: continue
        # Arrays of feasible and unfeasible points
        pf = np.vstack((pf,ps_arr[if_list]))
        pu = np.vstack((pu,ps_arr[iu_list]))
        # Keep only unique points
        pf = np.unique(pf,axis=0)
        pu = np.unique(pu,axis=0)
        # Leave iterations if amount of pf desired is reached
        if len(pf) > n_pf_desired:
            # Limit number of pu to prevent memory fill-up
            if len(pu) >= 5000:
                pu = pu[0:5000]
            break
        
        # -- Evaluate vicinity of found pf --
        # define vicinity indices
        if vicinity:
            if len(if_list)==0: continue
            idx_vic_base = idx_arr[if_list]
            idx_vic_arr = np.zeros((1,n_dim))
            for j in range(n_dim):
                idx_vic_plus = idx_vic_base.copy()
                idx_vic_plus[:,j] = idx_vic_plus[:,j]+1
                idx_vic_minus = idx_vic_base.copy()
                idx_vic_minus[:,j] = idx_vic_minus[:,j]-1
                # Leave iteration if beyond hypercube
                if np.any(idx_vic_plus[:,j] >= n_points) or \
                   np.any(idx_vic_minus[:,j] < 0):
                    continue
                idx_vic_arr = np.vstack((idx_vic_arr,idx_vic_plus))
                idx_vic_arr = np.vstack((idx_vic_arr,idx_vic_minus))
            # Remove first (dummy) row and make sure they are integers
            idx_vic_arr = idx_vic_arr[1:].astype(int)
            # Array of points in vicinity
            pvic_arr = np.zeros_like(idx_vic_arr)
            for j in range(n_dim):
                pvic_arr[:,j] = p_arr[idx_vic_arr[:,j],j]
            # Evaluate vicinity
            if_vic_list, iu_vic_list= [], []
            for i,pi in enumerate(pvic_arr):
                ysim = function(tsim,x0,u,ut,p,*pi)
                y = ysim[indexes]
                if np.all(ydata_min<=y) and np.all(y<=ydata_max):
                    if_vic_list.append(i)
                else:
                    iu_vic_list.append(i)
            # Leave iteration if no pf found in vicinity
            if len(if_vic_list): continue
            # Stack vicinity array on general array
            pf = np.vstack((pf,pvic_arr[if_vic_list]))
            pu = np.vstack((pu,pvic_arr[iu_vic_list]))
            # Keep only unique points
            pf = np.unique(pf,axis=0)
            pu = np.unique(pu,axis=0)
            # Leave iterations if amount of pf desired is reached
            if len(pf) >= n_pf_desired:
                if len(pu) >= 5000:
                    pu = pu[0:5000]
                break
        
        logger.debug('Iteration {}: {} feasible points'.format(it,pf.shape[0]))
        print('Iteration {}: {} feasible points'.format(it,pf.shape[0]))
            
    # Removing first row of initialized arrays
    pf = pf[1:]
    pu = pu[1:]
    if len(pu)>5000:
        pu = pu[0:5000]
    # Normalize data points with respect to maximum & minimum of each
    # coordinate for feasible points
    nmin = np.min(pf,axis=0)
    nmax = np.max(pf,axis=0)
    minmax = np.array([nmin,nmax])
    pfn = np.multiply(pf-nmin, 1/(nmax-nmin))
    pun = np.multiply(pu-nmin, 1/(nmax-nmin))
    
    
    t_elapsed = timing.end(tstart)
    logger.info('t_elapsed: {}'.format(timing.secondsToStr(t_elapsed)))
    
    
    return {
        'norm_minmax':minmax, 'pf':pf, 'pu':pu, 'pfn':pfn, 'pun':pun,
        'ydata_min':ydata_min, 'ydata_max':ydata_max #Used in refine routine
        }
