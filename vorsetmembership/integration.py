#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 16:15:20 2017

@author: Daniel
"""

# Routines for integration of ode
import numpy as np
from scipy.interpolate import interp1d
from uncertainties import ufloat, unumpy

def euler_forward(diff,y0,t,h=300,
                  interp_kind='linear',interp_fill_value='extrapolate'):
    tint = np.linspace(t[0],t[-1],int(t[-1]/h)+1)
    yint = np.zeros((tint.size,y0.size))
    yint[0] = y0
    for i,ti in enumerate(tint[1:],start=1):
        yint[i] = y0 + diff(y0,ti)*h
        y0 = yint[i]
    f_interp_y = interp1d(tint,yint,axis=0,kind=interp_kind,
                          fill_value=interp_fill_value)
    y = f_interp_y(t)  
    return y

def create_f_interp(t,ut,fill_value='extrapolate'):
    f_interp_dic = {}
    for k,v in ut.items():
        tu = np.linspace(0,t[-1],ut[k].shape[0])
        if '_pulse' in k: ikind = 'zero'
        else: ikind = 'linear'
        if isinstance(ut[k][0,0],float): #!!!
            f_interp_dic[k] = interp1d(tu,ut[k],
                        axis=0,kind=ikind,fill_value=fill_value)
        else:
            nom = unumpy.nominal_values(ut[k])
            std = unumpy.std_devs(ut[k])
            f_nom_interp_dic = interp1d(tu,nom,axis=0,kind=ikind,
                                        fill_value=fill_value)
            f_std_interp_dic = interp1d(tu,std,axis=0,kind=ikind,
                                        fill_value=fill_value)
            f_interp_dic[k] = (f_nom_interp_dic,f_std_interp_dic)
    return f_interp_dic

def call_f_interp(tnew,f_interp_dic):
    utnew = {}
    for k,f_interp in f_interp_dic.items():
        if isinstance(f_interp,tuple): #!!!
            f_nom_interp,f_std_interp = f_interp
            nom_interp = f_nom_interp(tnew)
            std_interp = f_std_interp(tnew)
            utnew[k] = np.full(tnew.size,ufloat(0,0))
            for i in range(tnew.size):
                utnew[k][i] = ufloat(nom_interp[i],std_interp[i])
        else:
            utnew[k] = f_interp(tnew)
    return utnew
             
if __name__ == '__main__':
    # Test function: dx_dt = fv/V * (x_in-x)
    from matplotlib import pyplot as plt
    t = np.arange(0,31) #[day]
    V = 5
    x0 = 4
    fv = 1
#    x_in = 10
    t_in = np.arange(0,31,1)
    x_in = 10*np.ones((len(t_in),1))
    x_in[int(len(x_in)/2):] = 20
    
    # Analytical
    f_interp_an = interp1d(t_in,x_in,axis=0)
    x_in_an = f_interp_an(t)[:,0]
    x_anal = x_in_an - (x_in_an-x0)*np.exp(-fv/V*t)
    
    # Numerical
    ut = {'x_in':x_in}
    f_interp_dic = create_f_interp(t,ut)
    def diff(_x0,_t): #,_ut if EF
        # States
        x = _x0
        # Time-varying inputs
        ut_int = call_f_interp(_t,f_interp_dic)
        x_in = ut_int['x_in']
        # d_dt
        dx_dt = (fv/V)*(x_in-x)
        return dx_dt
    # Integrate
    x = euler_forward(diff,x0,t,h=1/24) # h = 1 hr
#    x = odeint(diff,x0,t)

    # Plot
    fig1 = plt.figure(1)
    plt.plot(t,x_anal,label='Analytical')
    plt.plot(t,x,'--',label='Numerical')
    plt.legend()
    