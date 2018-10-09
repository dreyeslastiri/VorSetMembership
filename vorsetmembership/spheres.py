# -*- coding: utf-8 -*-
"""
@author: Daniel Reyes Lastiri
Module to generate spheres that characterise the FPS,
contained within unfeasible points, with Voronoi vertices as centers.
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

def _plot_spheres(fig_name,p_list,c,r):
    fig = plt.figure(fig_name)
    ax = plt.gca()
    ax = plot_scatter(ax,[pun,pfn],
        xlabel=r'$p_1$', ylabel=r'$p_2$', marker_list=['x','o'],
        label_list = ['Unfeasible','Feasible']
        )
    for i,ci in enumerate(c):
        circleplt = plt.Circle(ci,r[i],fill=False,color='0.5')
        ax.add_artist(circleplt)
    ax.set_aspect('equal')
    return fig
    
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
    logger.info('\n---- Calling spheres function ----')    
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
        cnstr_min, cnstr_max = [-1E3,]*ndim, [1E3,]*ndim
    t_elapsed = timing.end(tstart)
    logger.info('{} pu and {} pf'.format(pun.shape[0],pfn.shape[0]))
    logger.info('Spheres: \t{}'.format(vertices.shape[0]))
    logger.info('t_elapsed: \t{}'.format(timing.secondsToStr(t_elapsed)))
    
    # Calculate radii
    logger.info('Calculating radii')
    tstart = timing.start()
    r_list = [-1,]*len(vertices)
    iph_tup_all_list = [-1,]*len(vertices)
    for ip_tuple,iv_list in pv_ridge_dict.items():
        for iv in iv_list:
            # Calculate radius if not done yet (when r==-1),
            # for vertices that don't go to infinity (v!=-1)
            # (each vertex can be in several ridges, so take it only once)
            if iv!=-1 and r_list[iv] == -1:
                r_list[iv] = LA.norm(vertices[iv]-points[ip_tuple[0]])
                iph_tup_all_list[iv] = ip_tuple
    r_all_arr = np.asarray(r_list)
    t_elapsed = timing.end(tstart)
    logger.info('t_elapsed: {}'.format(timing.secondsToStr(t_elapsed)))
#    _plot_spheres('initial',[pfn,pun],vertices,r_all_arr)
    
    # Remove meaningless spheres
    # (spheres with very large radius, very small radius, or beyond constraints)
    # To show details: Uncomment commented lines and set logger level to debug
    if filtering:
        logger.info('Filtering out spheres')
        tstart = timing.start()
        pfnmin, pfnmax = pfn.min(axis=0), pfn.max(axis=0)
#        ibig = [i for i,r in enumerate (r_list) if r>=rmax]
#        ismall = [i for i,r in enumerate (r_list) if r<rmin]
#        ilo = [i for i,r in enumerate (r_list) if np.any(vertices[i,:]<pfnmin)]
#        ihi = [i for i,r in enumerate (r_list) if np.any(vertices[i,:]>pfnmax)]
#        ibelowcnstr = [i for i,r in enumerate (r_list)
#                        if np.any(vertices[i,:]<cnstr_min+r)]
#        iabovecnstr = [i for i,r in enumerate (r_list)
#                        if np.any(vertices[i,:]>cnstr_max-r)]
#        logger.debug('{0} large, {1} small'.format(
#                len(ibig), len(ismall)))
#        logger.debug('{0} below nmin, {1} above nmax'.format(
#                len(ilo),len(ihi)))
#        logger.debug('{0} below, {1} above constraints'.format(
#                len(ibelowcnstr),len(iabovecnstr)))
        i_meaningful = [
            i for i,r in enumerate(r_list)
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
        logger.info('Spheres: \t{}'.format(len(i_meaningful)))
        logger.info('t_elapsed: {}'.format(timing.secondsToStr(t_elapsed)))
#        _plot_spheres('meaningful',[pfn,pun],c_meaningful,r_meaningful)
    else:
        c_meaningful, r_meaningful = vertices, r_all_arr
        iph_tup_meaningful_list = iph_tup_all_list
    
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
    logger.info('Spheres: \t{}'.format(len(i_nonempty)))
    logger.info('t_elapsed: {}'.format(timing.secondsToStr(t_elapsed)))
#    _plot_spheres('non-empty',[pfn,pun],c_nonempty,r_nonempty)
    
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
    logger.info('Spheres: \t{}'.format(len(i_nonredundant)))
    logger.info('t_elapsed: {}'.format(timing.secondsToStr(t_elapsed)))
#    _plot_spheres('non-redundant',[pfn,pun],c_final,r_final)
    
    # Included and excluded feasible points
    logger.info('Indentifying excluded feasible points')
    tstart = timing.start()
    sum_center_pf_final = np.sum(center_pf_final,axis=0)
    ipf_included = list(np.where(sum_center_pf_final>=1)[0])
    ipf_excluded = list(np.where(sum_center_pf_final==0)[0])
    pfn_included = pfn[ipf_included]
    pfn_excluded = pfn[ipf_excluded]
    t_elapsed = timing.end(tstart)
    logger.info('pf included/excluded: \t{}/{}'.format(
            len(ipf_included),len(ipf_excluded)))
    logger.info('t_elapsed: {}'.format(timing.secondsToStr(t_elapsed)))

    # Hull of unfeasible points, including redundant spheres
    # (used to reset iterations to a region closer to the FPS)
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

# ---- TEST RUN ----
if __name__=='__main__':
    from vorsetmembership.plots import plot_scatter
    pfn = np.load('_tests/pfn_test.npy')
    pun = np.load('_tests/pun_test.npy')

    # Test function
    spheres_dict = spheres(pfn, pun)
    centers = spheres_dict['centers']
    radii = spheres_dict['radii']
    
    # Plot figure
    fig1 = plt.figure('Test')
    ax1 = plt.subplot2grid((4,1), (0, 0), rowspan=3)
    ax1 = plot_scatter(ax1,[pun,pfn],
        xlabel=r'$p_1$', ylabel=r'$p_2$', marker_list=['x','o'],
        label_list = ['Unfeasible','Feasible'],spheres=spheres_dict
        )
    ax1.set_xlim(0,1.5)
    ax1.set_ylim(0,1.8)
    ax1.set_aspect('equal')
