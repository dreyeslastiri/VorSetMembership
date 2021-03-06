# -*- coding: utf-8 -*-
"""
@author: Daniel Reyes Lastiri
Functions to sample random points from spheres
"""
# http://stackoverflow.com/questions/5408276/sampling-uniformly-distributed-random-points-inside-a-spherical-volume#5408843
# http://nojhan.free.fr/metah/

# For sample in ring, see answer by Hennadii Madan
#https://stackoverflow.com/questions/47472123/sample-uniformly-in-a-multidimensional-ring-without-rejection/47492146#47492146

# matlab code for gammainc sample:  
#X = randn(m,n);
#s2 = sum(X.^2,2);
#X = X.*repmat(r*(gammainc(s2/2,n/2).^(1/n))./sqrt(s2),1,n);

import numpy as np
import numpy.linalg as LA
from matplotlib import pyplot as plt
from scipy.special import gammainc
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
        
def _sample_unit(center,n_per_sphere):
    ndim = len(center)
    x = np.random.normal(size=(n_per_sphere, ndim)) 
    x /= np.linalg.norm(x, axis=1)[:, np.newaxis]
    punit_arr = x
    return punit_arr

def _sample_body_one(center,radius,n_per_sphere):
    r = radius
    ndim = center.size
    x = np.random.normal(size=(n_per_sphere, ndim))
    ssq = np.sum(x**2,axis=1)
    fr = r*gammainc(ndim/2,ssq/2)**(1/ndim)/np.sqrt(ssq)
    frtiled = np.tile(fr.reshape(n_per_sphere,1),(1,ndim))
    p = center + np.multiply(x,frtiled)
    return p

def _sample_ring(center,r1,r2,n_points):
    nd = center.size # Number of dimensions
    x = np.random.normal(size = (n_points,nd))
    # Generate on unit sphere
    x /= np.linalg.norm(x,axis=1)[:,np.newaxis] 
    # Using the inverse cdf method
    u = np.random.uniform(size=(n_points)) 
    # Inverse of cdf of ring volume as a function of radius
    sc = (u*(r2**nd-r1**nd) + r1**nd)**(1/nd) 
    return center + x*sc[:,None]
        
def sample_body(sphere_dict,norm_minmax,ns=100):
    ''' Find overestimated areas (hernias) and holes in the set
    '''
    logger.info('\n---- Calling sample body function ----')
    tstart = timing.start()
    
    # Retrieve variables from sphere dictionary
    centers = sphere_dict['centers']
    radii = sphere_dict['radii']
    center_pf = sphere_dict['center_pf']
    center_pf_sum = np.sum(center_pf,axis=1)
    n_pf = np.sum(center_pf_sum)
    r_mean = np.mean(radii)
    ndim = centers[0].size
    # Sample size per sphere
    # Target large spheres with low amount of pf (hernias),
    # sampling 5 times number of dimensions
    ns_list = [5*ndim if (ri>2*r_mean and center_pf_sum[i]/n_pf<0.05)
               else 0 for i,ri in enumerate(radii)]
    # Initialize array with first sphere
    pn_arr = _sample_body_one(centers[0],radii[0],ns_list[0])
    # Iterate over the rest of spheres
    for i,c in enumerate(centers[1:]):
        r = radii[i]
        pni_arr = _sample_body_one(c,r,ns_list[i])
        pn_arr = np.vstack((pn_arr,pni_arr))
    # Generate nominal values from normalized values
    nmin, nmax = norm_minmax
    p_arr = np.multiply(pn_arr, nmax-nmin) + nmin
    
    t_elapsed = timing.end(tstart)
    logger.info('t_elapsed: {}'.format(timing.secondsToStr(t_elapsed)))
    logger.debug('{} points sampled inside {} spheres'.format(
            sum(ns_list),radii.shape[0]))
    
    results = {'ps':p_arr, 'psn':pn_arr}
    
    return results

def sample_near_borders(sphere_dict, norm_minmax, ns=50, rmin=0.01):
    logger.info('\n---- Calling sample near borders function ----')
    tstart = timing.start()
    
    # Retrieve variables from sphere dictionary
    centers = sphere_dict['centers']
    radii = sphere_dict['radii']
    ndim = centers[0].size
    # Sample size per sphere, based on radii
    #ns_list = [int(ri/sum(radii)*ns)+1 for ri in radii]
    idx = np.arange(0,radii.shape[0])
    p_idx = radii/np.sum(radii)
    idx_sample = np.random.choice(idx,size=ns,p=p_idx,replace=True)
    idx_unique, idx_counts = np.unique(idx_sample, return_counts=True)
    idx_dict = dict(zip(idx_unique, idx_counts))
    # Initialize arrays with dummy point (to be deleted at the end)
    psn_in_arr = np.zeros((1,ndim))
    psn_out_arr = np.zeros((1,ndim))
    r_in_list, r_out_list =  [], []
    # Iterate over spheres
    for i,nSi in idx_dict.items():
        ri = radii[i]
#        r_in = 0.95*rmin
#        r_out = 2.0*ri
        # XXX: will try with very near borders only
        r_in = 0.5*ri
        r_out = 2.0*ri
        r_in_list.append(r_in)
        r_out_list.append(r_out)
        # Sample and remove redundant until desired nS
        psi_in_arr = np.zeros((1,ndim))
        while psi_in_arr.shape[0] < nSi:
            # Sample in ring r_in-r
            psi_in = _sample_ring(centers[i],r_in,ri,10*nSi)
            for j,cj in enumerate(centers):
                if i==j: continue # Ignore current sphere
                rj = radii[j]
                psi_in = psi_in[LA.norm(psi_in-cj,axis=1)>rj]
            psi_in_arr = np.vstack((psi_in_arr,psi_in))
            if psi_in_arr.shape[0] > nSi: psi_in_arr = psi_in_arr[1:nSi+1]
        # Stack pi in total arrays
        psn_in_arr = np.vstack((psn_in_arr,psi_in_arr))
        psi_out_arr = np.zeros((1,ndim))
        while psi_out_arr.shape[0] < nSi:
            # Sample in ring and r-r_out
            psi_out = _sample_ring(centers[i],ri,r_out,10*nSi)
            for j,cj in enumerate(centers):
                if i==j: continue # Ignore current sphere
                rj = radii[j]
                psi_out = psi_out[LA.norm(psi_out-cj,axis=1)>rj]
            psi_out_arr = np.vstack((psi_out_arr,psi_out))
            if psi_out_arr.shape[0] > nSi: psi_out_arr = psi_out_arr[1:nSi+1]
        # Stack pi in total arrays
        psn_out_arr = np.vstack((psn_out_arr,psi_out_arr))
    # Remove dummy point (used for initialization of array)
    psn_in_arr = psn_in_arr[1:ns+1,:]
    psn_out_arr = psn_out_arr[1:ns+1,:]
    # Generate nominal values from normalized values
    nmin, nmax = norm_minmax
    ps_in_arr = np.multiply(psn_in_arr, nmax-nmin) + nmin
    ps_out_arr = np.multiply(psn_out_arr, nmax-nmin) + nmin
    
    t_elapsed = timing.end(tstart)
    logger.info('t_elapsed: {}'.format(timing.secondsToStr(t_elapsed)))
    logger.info('{} points inside and {} points outside {} spheres'.format(
            psn_in_arr.shape[0],psn_out_arr.shape[0],centers.shape[0]))

    results = {'ps_in':ps_in_arr, 'psn_in':psn_in_arr,
        'ps_out':ps_out_arr, 'psn_out':psn_out_arr,
        'r_in':r_in_list, 'r_out':r_out_list}
    
    return results

# ------ TEST RUN ------
if __name__=='__main__':
    from vorsetmembership.spheres import spheres
    from mpl_toolkits.mplot3d import axes3d
    from vorsetmembership.plots import plot_scatter
    
    # Sample inside body of one sphere 2D
    fig1 = plt.figure('Sample one 2D')
    center1 = np.array([0,0])
    punit_arr1 = _sample_body_one(center1,1,10000)
    x1 = [punit_arr1[:,0],]
    y1 = [punit_arr1[:,1],]
    ax1 = fig1.gca()
    ax1.scatter(x1,y1,s=0.5)
    ax1.add_artist(plt.Circle(center1,1,fill=False,color='0.5'))
    ax1.set_xlim(-1.5,1.5)
    ax1.set_ylim(-1.5,1.5)
    ax1.set_aspect('equal')
    
    # Sample inisde body of one sphere 3D
    center2 = np.array([0,0,0])
    p3d= _sample_body_one(center2,1,300)
    theta, phi = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    fig2 = plt.figure('Sample one 3D')
    ax2 = plt.gca(projection='3d')
    ax2.scatter(p3d[:,0],p3d[:,1],p3d[:,2], s=20, c='b', marker='o')
    ax2.set_xlabel('\n x')
    ax2.set_ylabel('\n y')
    ax2.set_zlabel('\n z')
    x0,y0,z0 = center2
    r = 1
    x = x0 + r*np.cos(theta)*np.sin(phi)
    y = y0 + r*np.sin(theta)*np.sin(phi)
    z = z0 + r*np.cos(phi)
    ax2.plot_wireframe(x, y, z, color="r")
    #ax1.plot_surface(x, y, z, rstride=1, cstride=1, color='m')
    ax2.set_aspect("equal")

    # Sample inside body of dictionary of 2D spheres
    pfn = np.load('_tests/pfn_test.npy')
    pun = np.load('_tests/pun_test.npy')
    pfmin = np.min(pfn,axis=0)
    pfmax = np.max(pfn,axis=0)
    norm_minmax = np.vstack((pfmin,pfmax))
    sphere_dict = spheres(pfn, pun)
    centers = sphere_dict['centers']
    radii = sphere_dict['radii']
    p_body = sample_body(sphere_dict,norm_minmax)['psn']
    fig3 = plt.figure('Sample body')
    ax3 = plt.gca()
    ax3 = plot_scatter(ax3,[p_body,],'x','y',
        title='Sample in body of spheres', marker_list=['o'],
        )
    for i,ci in enumerate(centers):
        ri = radii[i]
        ax3.add_artist(plt.Circle(ci,ri,fill=False,color='0.5'))
    ax3.set_xlim(0.2,1.2)
    ax3.set_ylim(0.2,1.2)
    ax3.set_aspect('equal')
    
    # Sample near borders in 2D sphere dict
    p_nb = sample_near_borders(sphere_dict, norm_minmax,
                        ns=500, rmin=0.025)
    psn_nb_in = p_nb['psn_in']
    psn_nb_out = p_nb['psn_out']
    r_in = p_nb['r_in']
    r_out = p_nb['r_out']
    fig4 = plt.figure('Sample near borders')
    ax4 = plt.gca()
    ax4 = plot_scatter(ax4,[psn_nb_in,psn_nb_out],'x','y',
        title='Sample near borders of spheres',
        marker_list=['o','o'],
        label_list = ['Inside','Outside']
        )
    for i,ci in enumerate(centers):
        ri = radii[i]
        ax4.add_artist(plt.Circle(ci,ri,fill=False,color='0.5'))
        ax4.add_artist(plt.Circle(ci,r_in[i],fill=False,color='b',ls='--'))
        ax4.add_artist(plt.Circle(ci,r_out[i],fill=False,color='b',ls='--'))
    ax4.set_xlim(0.2,1.2)
    ax4.set_ylim(0.2,1.2)
    ax4.set_aspect('equal')
