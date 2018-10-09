# -*- coding: utf-8 -*-
"""
@author: Daniel Reyes Lastiri
Functions to evaluate sphere set
"""

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import logging

from chilafufu import timing

###TODO: make arrays for ysim_nb_in_list (and out) instead of lists to arrays
###TODO: try np.vectorize to evaluate feasible/unfeasible (instead of for loop)
###TODO yf_included_list is recalculating the function,
# after it has been calculated for ps_nb_out of previous iterations

# Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(name)s:%(levelname)s \t%(message)s')
file_handler = logging.FileHandler('report.log', mode='a')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def evaluate(function,tsim,x0,u,ut,p,sphere_dict,ps_body,ps_nb_in,ps_nb_out,
             norm_minmax,tdata,ydata_min,ydata_max,weights=False):
    '''Evaluates sampled points from feasible parameter
    Parameters
    ----------
    
    Returns
    -------
    
    '''
    logger.info('---- Calling evaluate output function ----')
    
    # Extract variables
    centers = sphere_dict['centers']
    ndim = centers[0].size
    nmin, nmax = norm_minmax
    pfn_included = sphere_dict['pfn_included']
    pf_included = np.multiply(pfn_included, nmax-nmin) + nmin
    pfn_excluded = sphere_dict['pfn_excluded']
    ny = ydata_min.shape[1]
    if weights == False:
        weights = np.array([1/ny,]*ny)
    
    # Indexes of tsim matching tdata
    # (does not return the closest value, but the one preceding)
    indexes = np.searchsorted(tsim,tdata)

    # ---- EVALUATE SAMPLES ----
    # ---- Inside body of spheres ----
    logger.info('Evaluating inside body of spheres')
    tstart = timing.start()
    # Unfeasible points provide new points for Voronoi tesselation
    # This sample has no impact on overestimation or underestimation
    # (it is just a control measure to cure hernias)
    isu_body_list= []
    for i,pi in enumerate(ps_body):
        ysim = function(tsim,x0,u,ut,p,*pi)
        y_body = ysim[indexes]
        if np.all(ydata_min<=y_body) and np.all(y_body<=ydata_max):
            continue
        else:
            isu_body_list.append(i)
    # Convert lists to arrays, return nan if empty
    if len(isu_body_list) > 0:
        psu_body_arr = ps_body[isu_body_list]
    else:
        psu_body_arr = np.array([ [np.nan,]*ndim ])
    logger.debug('{} unfeasible points'.format(len(isu_body_list)))
    t_elapsed = timing.end(tstart)
    logger.info('t_elapsed: {}'.format(timing.secondsToStr(t_elapsed)))
    
    # ---- Inside borders ----
    logger.info('Evaluating near borders, inside')
    tstart = timing.start()
    # Initialize lists of unfeasible points and outputs.
    # Unfeasible points provide new points for Voronoi tesselation.
    # Unfeasible outputs are used to measure overestimation
    isu_nb_in_list = []
    ysim_nb_in_list = []
    # Iterate over sample to evaluate output as feasible or unfeasible
    for i,pi in enumerate(ps_nb_in):
        ysim = function(tsim,x0,u,ut,p,*pi)
        ysim_nb_in_list.append(ysim)
        y_nb_in = ysim[indexes]
        if np.all(ydata_min<=y_nb_in) and np.all(y_nb_in<=ydata_max):
            continue
        else:
            isu_nb_in_list.append(i)
    # Convert lists to arrays, return nan if empty
    ysim_nb_in_arr = np.asarray(ysim_nb_in_list)
    if len(isu_nb_in_list) > 0:
        psu_nb_in_arr = ps_nb_in[isu_nb_in_list]
        ysu_nb_in_arr = ysim_nb_in_arr[isu_nb_in_list]
    else:
        psu_nb_in_arr = np.array([ [np.nan,]*ndim ])
        ysu_nb_in_arr = np.nan*np.zeros((1,ysim.shape[0],ysim.shape[1]))
    print('{} unfeasible points'.format(len(isu_nb_in_list)))
    t_elapsed = timing.end(tstart)
    logger.info('t_elapsed: {}'.format(timing.secondsToStr(t_elapsed)))
    
    # ---- Outside borders ----
    logger.info('Evaluating near borders, outside')
    tstart = timing.start()
    # Initialize lists of feasible points
    # Feasible points provide new points for Voronoi tesselation.
    # Feasible outputs are used to measure underestimation.
    isf_nb_out_list = []
    ysim_nb_out_list = []
    # Iterate over sample to evaluate output as feasible or unfeasible
    for i,pi in enumerate(ps_nb_out):
        ysim = function(tsim,x0,u,ut,p,*pi)
        ysim_nb_out_list.append(ysim)
        y_nb_out = ysim[indexes]
        if np.all(ydata_min<=y_nb_out) and np.all(y_nb_out<=ydata_max):
            isf_nb_out_list.append(i)
    # Convert lists to arrays, return nan if empty
    ysim_nb_out_arr = np.asarray(ysim_nb_out_list)
    if len(isf_nb_out_list) > 0:
        psf_nb_out_arr = ps_nb_out[isf_nb_out_list]
        ysf_nb_out_arr = ysim_nb_out_arr[isf_nb_out_list]
    else:
        psf_nb_out_arr = np.array([ [np.nan,]*ndim ])
        ysf_nb_out_arr = np.nan*np.zeros((1,ysim.shape[0],ysim.shape[1]))
    logger.debug('{} feasible points'.format(len(isf_nb_out_list)))
    t_elapsed = timing.end(tstart)
    logger.info('t_elapsed: {}'.format(timing.secondsToStr(t_elapsed)))
    
    logger.info('Compiling results')
    tstart = timing.start()
    
    # ---- OVERESTIMATION ----
    # Based on last sample inside set of spheres
    # (Does not account for psn_body, because that's a control measure
    # outside of the overall random sample, very focused on the hernias)
    errabovemax_arr = ysu_nb_in_arr[:,indexes,:] - ydata_max
    errabovemax_arr = errabovemax_arr.clip(min=0)
    errbelowmin_arr = ydata_min - ysu_nb_in_arr[:,indexes,:]
    errbelowmin_arr = errbelowmin_arr.clip(min=0)   
    # Retrieve largest error, zeros if there is no ysu_nb_out
    # (above upper bound)
    if not np.all(np.isnan(errabovemax_arr)):
        errabovemax = errabovemax_arr.max(axis=0)
    else:
        errabovemax = np.zeros_like(ydata_max)
    # (below lower bound)
    if not np.all(np.isnan(errbelowmin_arr)):
        errbelowmin = errbelowmin_arr.max(axis=0)
    else:
        errbelowmin = np.zeros_like(ydata_min)
    # Overestimation. Sum of deviations with respect to sum of bounds
    overest_arr = np.sum((errabovemax+errbelowmin)/(ydata_max-ydata_min),axis=0)
    overest = np.sum(weights * overest_arr)
    
    # ---- UNDERESTIMATION ----
    # (Based on total amount of pf_included, not only on last sample)     
    # Initialize list of deviations
    yf_included_list = [function(tsim,x0,u,ut,p,*pi) for pi in pf_included]
    yf_included_arr = np.asarray(yf_included_list)
    shortabovemin_arr = yf_included_arr[:,indexes,:] - ydata_min
    shortabovemin_arr = shortabovemin_arr.clip(min=0)
    shortbelowmax_arr = ydata_max - yf_included_arr[:,indexes,:]
    shortbelowmax_arr = shortbelowmax_arr.clip(min=0)    
    # Retrieve smallest distance from upper and lower bounds
    if not np.all(np.isnan(shortabovemin_arr)):
        shortabovemin = shortabovemin_arr.min(axis=0)
    else:
        shortabovemin = np.zeros_like(ydata_min)
    if not np.all(np.isnan(shortbelowmax_arr)):
        shortbelowmax = shortbelowmax_arr.min(axis=0)
    else:
        shortbelowmax = np.zeros_like(ydata_max)
    # Underestimation. Sum of deviations with respect bounds
    underest_arr = np.sum((shortabovemin+shortbelowmax)/(ydata_max-ydata_min),axis=0)
    underest = np.sum(weights * underest_arr)
    
    # ---- WEIGHTED DEVIATION ----
    td = overest + underest
    # Sample ratios for underestimation and overestimation
    ratio_oe = len(isu_nb_in_list)/len(ps_nb_in)
    ratio_ue = len(isf_nb_out_list)/len(ps_nb_out)
    # Exclusion ratio
    ratio_excl = len(pfn_excluded)/(len(pfn_excluded)+len(pfn_included))
    # Weighted deviation
    wd = ratio_oe*overest + ratio_ue*underest + ratio_ue*ratio_excl
    
    # ---- PLOT VARIABLES ----
    # For plotting, the error bars are masked to show only if > 0
    # The array for plots is constructed as (min,max)
    # (the error bars are assymetrical)
    # ---- Overestimation ----
    errabovemax_masked = np.ma.masked_equal(errabovemax,0)
    errabovemax_lo = np.zeros_like(errabovemax)
    if type(errabovemax_masked.mask)==np.ndarray:
        errabovemax_lo[errabovemax_masked.mask] = np.nan
    
    errbelowmin_masked = np.ma.masked_equal(errbelowmin,0)
    errbelowmin_hi = np.zeros_like(errbelowmin)
    if type(errbelowmin_masked.mask)==np.ndarray:
        errbelowmin_hi[errbelowmin_masked.mask] = np.nan
    
    errabovemax_plt = np.array( [errabovemax_lo, errabovemax_masked] )
    errbelowmin_plt = np.array( [errbelowmin_masked, errbelowmin_hi] )
    
    # ---- Underestimation ----
    shortbelowmax_masked = np.ma.masked_equal(shortbelowmax,0)
    shortbelowmax_hi = np.zeros_like(shortbelowmax)
    if type(shortbelowmax_masked.mask)==np.ndarray:
        shortbelowmax_hi[shortbelowmax_masked.mask] = np.nan
    
    shortabovemin_masked = np.ma.masked_equal(shortabovemin,0)
    shortabovemin_lo = np.zeros_like(shortabovemin)
    if type(shortabovemin_masked.mask)==np.ndarray:
        shortabovemin_lo[shortabovemin_masked.mask] = np.nan
    
    shortbelowmax_plt = np.array( [shortbelowmax_masked, shortbelowmax_hi] )
    shortabovemin_plt = np.array( [shortabovemin_lo, shortabovemin_masked] )
    
    # ---- REPORTS ----
    t_elapsed = timing.end(tstart)
    logger.info('t_elapsed: {}'.format(timing.secondsToStr(t_elapsed)))
    print('Underestimation = {0:2.2g}'.format(underest))
    print('Overestimation = {0:2.2g}'.format(overest))
    print('Weighted deviation = {0:2.2g}'.format(wd))
        
    results = {
        'psu_body':psu_body_arr, 'isu_body_list':isu_body_list,
        'psu_nb_in':psu_nb_in_arr, 'isu_nb_in_list':isu_nb_in_list,
        'psf_nb_out':psf_nb_out_arr, 'isf_nb_out_list':isf_nb_out_list,
        'ysu_nb_in':ysu_nb_in_arr, 'ysf_nb_out':ysf_nb_out_arr,
        'yf_included':yf_included_arr,
        'overestimation':overest, 'underestimation':underest,
        'total_deviation':td,
        'weighted_deviation':wd,
        'errabovemax_plt':errabovemax_plt,
        'errbelowmin_plt':errbelowmin_plt ,
        'shortabovemin_plt':shortabovemin_plt,
        'shortbelowmax_plt':shortbelowmax_plt
        }
    
    return results
