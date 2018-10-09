# -*- coding: utf-8 -*-
"""
@author: Daniel Reyes Lastiri
Module to characterise a set of feasible solutions for parameter estimates
as a set of circles contained within unfeasible points,
by means of Voronoi vertices as the centers of the circles
"""
import numpy as np
from scipy.spatial import Voronoi
from matplotlib import pyplot as plt
import numpy.linalg as LA
from scipy.spatial.distance import cdist
import logging

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

def spheres(p_feasible, p_unfeasible,rmin=0.025,rmax=1.0,
            filtering=True,cnstr=False): 
    '''Generates spheres that contain feasible points and exclude
    unfeasible points, using a Voronoi diagram of unfeasible points as seeds.
    Feasible and unfeasible points must be provided in normalized coordinates.
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    logger.info('---- Calling spheres function ----')    
    pfn, pun = p_feasible, p_unfeasible
    ndim = pfn.shape[1]
    # Generate Voronoi diagram and retrieve its information
    logger.info('Generating spheres from Voronoi vertices') 
    tstart = timing.start()
    vor = Voronoi(pun)
    # Coordinates of input points
    points = vor.points
    # Coordinates of Voronoi vertices
    vertices = vor.vertices
    # Dictionary of indices of point-vertices per ridge
    pv_ridge_dict = vor.ridge_dict
    # Constraints (normalized). Tuple of lists ([p0min,p1min],[p0max,p1max])
    if cnstr:
        cnstr_min, cnstr_max = cnstr[0], cnstr[1]
    else:
        cnstr_min, cnstr_max = [-1E4,]*ndim, [1E4,]*ndim
    t_elapsed = timing.end(tstart)
    logger.debug('{} pu and {} pf to {} spheres'.format(
            pun.shape[0],pfn.shape[0],vertices.shape[0]))
    logger.info('t_elapsed: \t{}'.format(timing.secondsToStr(t_elapsed)))
    
    # Calculate radii
    logger.info('Calculating radii')
    tstart = timing.start()
    r_all_list = [-1,]*len(vertices)
    iph_tup_all_list = [-1,]*len(vertices)
    for ip_tuple,iv_list in pv_ridge_dict.items():
        for iv in iv_list:
            # Calculate radius if not done yet (when r==-1),
            # for vertices that don't go to infinity (v!=-1)
            # (each vertex can be in several ridges, so take it only once)
            if iv!=-1 and r_all_list[iv] == -1:
                r_all_list[iv] = LA.norm(vertices[iv]-points[ip_tuple[0]])
                iph_tup_all_list[iv] = ip_tuple
    r_all_arr = np.asarray(r_all_list)
    t_elapsed = timing.end(tstart)
    logger.info('t_elapsed: {}'.format(timing.secondsToStr(t_elapsed)))
    
    # Remove meaningless spheres
    # (spheres with very large radius, very small radius, or beyond constraints)
    logger.info('Filtering out spheres')
    tstart = timing.start()
    pfnmin, pfnmax = pfn.min(axis=0), pfn.max(axis=0)
    if filtering:

        print('num r all',r_all_arr.shape[0])
        ibig = [i for i,r in enumerate (r_all_list) if r>=rmax]
        ismall = [i for i,r in enumerate (r_all_list) if r<rmin]
        ilowmin = [i for i,r in enumerate (r_all_list) if np.any(vertices[i,:]<pfnmin)]
        ihimax = [i for i,r in enumerate (r_all_list) if np.any(vertices[i,:]>pfnmax)]
        icnstrmin = [i for i,r in enumerate (r_all_list) if np.any(vertices[i,:]<cnstr_min+r)]
        icnstrmax = [i for i,r in enumerate (r_all_list) if np.any(vertices[i,:]>cnstr_max-r)]
        print('too large',len(ibig))
        print('too small',len(ismall))
        print('too far low pfn',len(ilowmin))
        print('too far hi pfn',len(ihimax))
        print('too far low cnstr', len(icnstrmin))
        print('too far hi cnstr', len(icnstrmax))
    
        i_meaningful = [
            i for i,r in enumerate(r_all_list)
            if (r!=-1 and r<rmax and r>rmin
            and np.all(vertices[i,:] > pfnmin)
            and np.all(vertices[i,:] < pfnmax)
            and np.all(vertices[i,:] > cnstr_min+r)
            and np.all(vertices[i,:] < cnstr_max-r))
            ]
        c_meaningful = vertices[i_meaningful]
        r_meaningful = r_all_arr[i_meaningful]
        iph_tup_meaningful_list = [iph_tup_all_list[i] for i in i_meaningful]
        t_elapsed = timing.end(tstart)
    else:
        c_meaningful, r_meaningful = vertices, r_all_arr
        iph_tup_meaningful_list = iph_tup_all_list
        
    print('Spheres: \t{}'.format(r_meaningful.shape[0]))
    logger.info('t_elapsed: {}'.format(timing.secondsToStr(t_elapsed)))
    
    # Generate array feasible points associated to each sphere center
    # as 0 for not belonging and 1 for belonging to a sphere
    logger.info('Assigning feasible points to spheres')
    tstart = timing.start()
    dist_arr = cdist(c_meaningful, pfn, 'euclidean')
    center_pf = dist_arr/r_meaningful[:,np.newaxis]
    center_pf[center_pf<=1] = 1
    center_pf[center_pf>1] = 0
    t_elapsed = timing.end(tstart)
    logger.info('t_elapsed: {}'.format(timing.secondsToStr(t_elapsed)))
    
    # Delete empty spheres
    logger.info('Deleting empty spheres')
    tstart = timing.start()
    center_pf_sum = np.sum(center_pf,axis=1)
    i_nonempty = list(np.where(center_pf_sum>0)[0])
    c_nonempty = c_meaningful[i_nonempty]
    r_nonempty = r_meaningful[i_nonempty]
    center_pf_nonempty = center_pf[i_nonempty]
    iph_tup_nonempty_list = [iph_tup_meaningful_list[i] for i in i_nonempty]
    t_elapsed = timing.end(tstart)
    logger.debug('Spheres: \t{}'.format(len(i_nonempty)))
    logger.info('t_elapsed: {}'.format(timing.secondsToStr(t_elapsed)))
    
    # Remove redundant spheres
    logger.info('Deleting redundant spheres')
    # First, sort in ascending radius
    tstart = timing.start()
    i_sorted = np.argsort(r_nonempty)
    r_sorted = r_nonempty[i_sorted]
    c_sorted = c_nonempty[i_sorted]
    center_pf_sorted = center_pf_nonempty[i_sorted]
    iph_tup_sorted = [iph_tup_nonempty_list[i] for i in i_sorted]        
    # Then, remove redundants, leaving at least 1 sphere
    i_redundant, i_nonredundant = [], []
    sum_center_pf_sorted = np.sum(center_pf_sorted,axis=0)
    sum_redundant = np.zeros_like(sum_center_pf_sorted)
    for i,v in enumerate(center_pf_sorted):
        test_redundancy = v * (sum_center_pf_sorted - sum_redundant)
        if np.any(test_redundancy==1):
            i_nonredundant.append(i)
        else:
            i_redundant.append(i)
            sum_redundant += v
    if len(i_nonredundant) == 0:
        i_nonredundant = [0,]
    c_final = c_sorted[i_nonredundant]
    r_final = r_sorted[i_nonredundant]
    center_pf_final = center_pf_sorted[i_nonredundant]
    iph_final = [iph_tup_sorted[i] for i in i_nonredundant]
    t_elapsed = timing.end(tstart)
    print('Spheres: \t{}'.format(len(i_nonredundant)))
    logger.info('t_elapsed: {}'.format(timing.secondsToStr(t_elapsed)))

    # Included and excluded feasible points
    logger.info('Indentifying excluded feasible points')
    tstart = timing.start()
    sum_center_pf_final = np.sum(center_pf_final,axis=0)
    ipf_included = list(np.where(sum_center_pf_final>=1)[0])
    ipf_excluded = list(np.where(sum_center_pf_final==0)[0])
    pfn_included = pfn[ipf_included]
    pfn_excluded = pfn[ipf_excluded]
    t_elapsed = timing.end(tstart)
    print('pf included/excluded: \t{}/{}'.format(
            len(ipf_included),len(ipf_excluded)))
    logger.info('t_elapsed: {}'.format(timing.secondsToStr(t_elapsed)))

    # Hull of unfeasible points, including redundant spheres
    # (used only for first iteration)
    iph_set = {tup[i] for tup in iph_final for i in range(len(tup))}
    pun_hull_list = [points[i] for i in iph_set]
    pun_hull = np.asarray(pun_hull_list)

    # Dictionary of results
    results = {'centers':c_final, 'radii':r_final,
               'center_pf':center_pf_final,
               'pfn_included':pfn_included, 'pfn_excluded':pfn_excluded,
               'ipf_included':ipf_included, 'ipf_excluded':ipf_excluded,
               'pun_hull':pun_hull
               }
    
    return results

# -------------------------------------
# ------------- TEST RUN --------------      
if __name__=='__main__':
    plt.close('all')
    pfn = np.load('_tests/pfeasmcnorm.npy')
    pun = np.load('_tests/punfeasmcnorm.npy')
    vor = Voronoi(pun)

    # Test function
    sphere_set_results = spheres(pfn, pun)
    centers = sphere_set_results['centers']
    radii = sphere_set_results['radii']
    
    # Plot figure
    fig1 = plt.figure(1)
    ax1 = plt.gca()
    ax1.scatter(pfn[:,0],pfn[:,1])
    ax1.scatter(pun[:,0],pun[:,1],c='r')
    for i,center in enumerate(centers):
        radius = radii[i]
        circleplt = plt.Circle(center,radius,fill=False,color='0.5')
        ax1.add_artist(circleplt)
    ax1.set_xlim(0,1.5)
    ax1.set_ylim(0,1.5)
    ax1.set_aspect('equal')
