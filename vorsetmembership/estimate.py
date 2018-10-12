# -*- coding: utf-8 -*-
"""
@author: Daniel Reyes Lastiri
Funtions to evaluate and refine a sphere set
"""

import numpy as np
import logging

from vorsetmembership.spheres import spheres
from vorsetmembership.sample import sample_body, sample_near_borders
from vorsetmembership.evaluate import evaluate
from vorsetmembership import timing


# Logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(message)s')
    file_handler = logging.FileHandler('report.log', mode='w')
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

def slice_all(slice_obj,arrays_list):
    self = [arr[slice_obj] for arr in arrays_list]
    return self
        
def chilafufu(initialize_dict,function,tsim,x0,u,ut,p,tdata,
              wdtol=0.05, iter_max=10, ns_nb=100,
              rmin=0.025, rmax = 1.0, constraints=False, filter_spheres=True):
    logger.info('\n---- Starting FPS identification ----')
    # Retrieve variables from results of Initialize
    ydata_min = initialize_dict['ydata_min']
    ydata_max = initialize_dict['ydata_max']
    pfn = initialize_dict['pfn']
    pun = initialize_dict['pun']
    pun_ini = pun.copy()
    norm_minmax = initialize_dict['norm_minmax']
    nmin, nmax = norm_minmax
    # Initialize arrays of results per iteration
    it_arr = np.arange(iter_max)
    td_arr = np.ones(iter_max)
    oe_arr = np.ones(iter_max) 
    ue_arr = np.ones(iter_max)
    wd_arr = np.ones(iter_max)
    t_elapsed_arr = np.ones(iter_max)
    # Other parameters
    ndim = pfn.shape[1]
    # Normalize constraints.
    # Tuple of lists, e.g. ([p0min,p1min],[p0max,p1max])
    if constraints:
        cmin,cmax = constraints[0], constraints[1]
        cmin_n, cmax_n = (cmin-nmin)/(nmax-nmin), (cmax-nmin)/(nmax-nmin)
        cnstr_n = (cmin_n,cmax_n)
    else:
        cnstr_n = False
    # -- Initial sample and evaluation (Iteration 0) --
    logger.info('-- Iteration 0 --')
    tstart_0 = timing.start()
    
    # Initialize spheres
    sphere_dict = spheres(pfn,pun,rmin=rmin,rmax=rmax,
                          cnstr=cnstr_n,filtering=filter_spheres)
    pun_hull = sphere_dict['pun_hull']
    pun_hulls = pun_hull.copy()
    # Sample in body of spheres
    sample_body_res = sample_body(sphere_dict,norm_minmax)
    ps_body, psn_body = sample_body_res['ps'], sample_body_res['psn']
    # Sample near borders of spheres
    sample_nb_res = sample_near_borders(sphere_dict,norm_minmax,ns=ns_nb,
                                        rmin=rmin)
    ps_nb_in, psn_nb_in = sample_nb_res['ps_in'], sample_nb_res['psn_in']
    ps_nb_out, psn_nb_out = sample_nb_res['ps_out'], sample_nb_res['psn_out']
    # Evaluate sampled points
    # (to determine deviations from bounds and new points for Voronoi)
    eval_out_dict = evaluate(function,tsim,x0,u,ut,p,
                             sphere_dict,ps_body,ps_nb_in,ps_nb_out,
                             norm_minmax,tdata,ydata_min,ydata_max)
    # Deviations
    td_arr[0] = eval_out_dict['total_deviation']
    oe_arr[0] = eval_out_dict['overestimation']
    ue_arr[0] = eval_out_dict['underestimation']
    wd_arr[0] = eval_out_dict['weighted_deviation']
    # New points for Voronoi tesselation (remove if nan)
    isu_body = eval_out_dict['isu_body_list']
    if not np.all(np.isnan(isu_body)): psun_body = psn_body[isu_body]
    else: psun_body = np.array([np.nan,]*ndim)
    isu_nb_in = eval_out_dict['isu_nb_in_list']
    if not np.all(np.isnan(isu_nb_in)): psun_nb_in = psn_nb_in[isu_nb_in]
    else: psun_nb_in = np.array([np.nan,]*ndim)
    isf_nb_out = eval_out_dict['isf_nb_out_list']
    if not np.all(np.isnan(isf_nb_out)): psfn_nb_out = psn_nb_out[isf_nb_out]
    else: psfn_nb_out = np.array([np.nan,]*ndim)
    psfn_nb_in = psn_nb_in[eval_out_dict['isf_nb_in_list']]
    #XXX: Testing idea of not only pun_hull on 1st iteration
    if constraints:
        pun = np.vstack((pun,psun_body,psun_nb_in)) # Need all pun when constrained
    else:
        pun = np.vstack((pun,psun_body,psun_nb_in))
    mask_pun = np.all(np.isnan(pun), axis=1)
    pun = pun[~mask_pun]
    pfn = np.vstack((pfn,psfn_nb_out))
    mask_pfn = np.all(np.isnan(pfn), axis=1)
    pfn = pfn[~mask_pfn]
    
    t_elapsed = timing.end(tstart_0)
    t_elapsed_arr[0] = t_elapsed
    logger.info('t_elapsed: {}'.format(timing.secondsToStr(t_elapsed)))
    
    # Iterate until error tolerance or iteration limit are met
    for i in range(iter_max)[1:]:        
        logger.info('-- Iteration {} --'.format(i))
        tstart_it = timing.start()
        # ---- SPHERE SET ----
        # Generate new sphere dictionary
#        rmin = rmin*min(pfn.max(axis=0)-pfn.min(axis=0))
        sphere_dict_new = spheres(pfn,pun,rmin=rmin,rmax=rmax,
                                  cnstr=cnstr_n,filtering=filter_spheres)
        # Store pun_hull for soft reset
        if i%2 == 0:
            pun_hulls = np.vstack((pun_hulls,sphere_dict_new['pun_hull']))
        
        # ---- SAMPLE ----
        # Sample in body of spheres
        sample_body_dict = sample_body(sphere_dict_new,norm_minmax)
        ps_body, psn_body = sample_body_dict['ps'], sample_body_dict['psn']
        # Sample near borders of spheres
        sample_nb_dict = sample_near_borders(sphere_dict_new,norm_minmax,
                                             ns=ns_nb,rmin=rmin)
        ps_nb_in,psn_nb_in = sample_nb_dict['ps_in'],sample_nb_dict['psn_in']
        ps_nb_out,psn_nb_out = sample_nb_dict['ps_out'],sample_nb_dict['psn_out']

        # ---- EVALUATE ----
        # ---- Evaluate sampled points ----
        eval_out_dict = evaluate(function,tsim,x0,u,ut,p,
                                 sphere_dict_new,ps_body,ps_nb_in,ps_nb_out,
                                 norm_minmax,tdata,ydata_min,ydata_max)
        # Deviations
        td_arr[i] = eval_out_dict['total_deviation']
        oe_arr[i] = eval_out_dict['overestimation']
        ue_arr[i] = eval_out_dict['underestimation']
        wd_arr[i] = eval_out_dict['weighted_deviation']
        # New points for Voronoi tesselation
        isu_body = eval_out_dict['isu_body_list']
        if not np.all(np.isnan(isu_body)): psun_body = psn_body[isu_body]
        else: psun_body = np.array([np.nan,]*ndim)
        isu_nb_in = eval_out_dict['isu_nb_in_list']
        if not np.all(np.isnan(isu_nb_in)): psun_nb_in = psn_nb_in[isu_nb_in]
        else: psun_nb_in = np.array([np.nan,]*ndim)
        isf_nb_out = eval_out_dict['isf_nb_out_list']
        if not np.all(np.isnan(isf_nb_out)): psfn_nb_out = psn_nb_out[isf_nb_out]
        else: psfn_nb_out = np.array([np.nan,]*ndim)
         
        # ---- END ----
        # Stop iterations if weighted deviation tolerance is met
        if wd_arr[i] <= wdtol:
            t_elapsed_it = timing.end(tstart_it)
            t_elapsed_arr[i] = t_elapsed_it
            logger.info('t_elapsed iteration{}: {}'.format(
                    i,timing.secondsToStr(t_elapsed_it)))
            sl = slice(0,i+1)
            it_arr, td_arr, wd_arr, oe_arr, ue_arr, t_elapsed_arr = \
                slice_all(sl,[it_arr,td_arr,wd_arr,oe_arr,ue_arr,t_elapsed_arr])
            logger.info('WD tol {} met at iteration {}'.format(wdtol,i))
            print('\nSuccess:')
            print(
            'WD tolerance {} met at iteration {}'.format(wdtol,i))
            break
        # Stop iterations if maximum iteration is reached
        if i == iter_max-1:
            t_elapsed_it = timing.end(tstart_it)
            t_elapsed_arr[i] = t_elapsed_it
            logger.info('t_elapsed iteration {}: {}'.format(
                    i,timing.secondsToStr(t_elapsed_it)))
            print('')
            print('Deviation tolerance was not met')
            print('Total elapsed time = {0:.2f} s'.format(np.sum(t_elapsed_arr)))
            break
        
        # ---- NEW POINTS ----
        # Stack new points and remove nan
        pun = np.vstack((pun,psun_body,psun_nb_in))
        mask_pun = np.all(np.isnan(pun), axis=1)
        pun = pun[~mask_pun]
        
        keeppfin = int(0.1*psfn_nb_in.shape[0])
        pfn = np.vstack((pfn,psfn_nb_out,psfn_nb_in[0:keeppfin,:]))
        mask_pfn = np.all(np.isnan(pfn), axis=1)
        pfn = pfn[~mask_pfn]
        
        # Soft reset if excess pun
#        if i%5==0:# and pun.shape[0] >= 100:
#            print('SOFT RESET')
#            print('TADA!')
#            pun = np.vstack((pun_ini,pun_hulls))
            
        t_elapsed_it = timing.end(tstart_it)
        t_elapsed_arr[i] = t_elapsed_it
        logger.info('t_elapsed iteration {}: {}'.format(
                i,timing.secondsToStr(t_elapsed_it)))
    
    
    results = {
        'pfn':pfn, 'pun':pun,
        'spheres':sphere_dict_new,
        'errabovemax_plt':eval_out_dict['errabovemax_plt'],
        'errbelowmin_plt':eval_out_dict['errbelowmin_plt'] ,
        'shortabovemin_plt':eval_out_dict['shortabovemin_plt'],
        'shortbelowmax_plt':eval_out_dict['shortbelowmax_plt']
        }
    
    logs = {
        'iterations':it_arr,
        'total_deviation':td_arr,
        'overestimation':oe_arr,
        'underestimation':ue_arr,
        'weighted_deviation':wd_arr,
        't_elapsed':t_elapsed_arr}
    
    return results, logs
  

