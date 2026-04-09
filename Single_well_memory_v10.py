# -*- coding: utf-8 -*-
"""
Single_well_memory_v10      08/04/2026

Computation code for quasistatic and dynamic evolution 
of a single well memory

Units : energy in kBT, entropy in kB, others MKSA 

Computation time about 40 mn (with AMD Ryzen 7 processor)

© 2026 Jean Argouarc'h
This script is part of the project 
"Limits of Landauer Principle and of the Second Law of Thermodynamics".
Licensed under MIT License. See LICENSE file for details.

jr.argouarch@gmail.com

"""

# ==========================
# Import libraries
# =====================

import time
from datetime import datetime
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import random
import sys
import os
import pickle
     
# ==========================
# Figure parameters
# =========================
fig_size      = 2    
fig_fontsize  = 8
mpl.rcParams.update({"figure.dpi": 100, "savefig.dpi": 300,}) 

# ==========================
# Utility functions
# ==========================
    
def display_parameters():
    print ('\nLandscape parameters')
    print ('fim   = %6.1f  point M (0,fim)' % fim)
    print ('xa    = %6.3f  half well width ' % xa)
    print ('a0    = %6.3f  coef well parabola' % a0)
    
    print ('\nProcess parameters')
    print ('tc    = %6i  cycle time (s)'% tc)
    print ('dt    = %6.2f  time increment (s)'% dt)
    print ('cd    = %6.1f  diffusion coefficient'% cd)
    print ('cb    = %6.1f  brownian movement coefficient' % cb)
                
            
def plot_quasistatic_reset_to_zero(tc):
    print('\nFigure of quasistatic reset-to-zero, starting from bit =', bit)    
    print ('\nSqs max = %6.3f' % max(Sqs-np.log(2)))    
    fig_ls       = 0.8
    fig_ls2      = 1.0
    tdata        = np.arange(tc+1)            
    
    fig,ax = plt.subplots(figsize=(3*fig_size,1.5*fig_size))
    ax.plot(tdata,Sqs0-np.log(2),'r-',linewidth=fig_ls,label='Sqs')
    ax.plot(tdata,Hqs0,'g-',linewidth=fig_ls,label='Hqs')
    ax.plot(tdata,Pqs1,'b-',linewidth=fig_ls2,label='Pqs1')
    ax.plot(tdata,Pqs0,'b--',linewidth=fig_ls2,label='Pqs0')
    plt.xticks([0,tc/2,tc],['0','tc/2','tc'],fontsize=fig_fontsize)
    plt.yticks([0.5,1, 1.5,1.8],[0.5,1.0, 1.5,1.8],fontsize=fig_fontsize)
    plt.ylim(-0.1,2.)
    plt.text(int(0.2*tc),-0.3,'phase 1',fontsize=fig_fontsize)
    plt.text(int(0.7*tc),-0.3,'phase 2',fontsize=fig_fontsize)
    ax.legend(bbox_to_anchor=(1.17,0.8),fontsize=fig_fontsize)                    
    ax.grid()
    plt.show() 
 
    
def plot_both_reset_to_zeros(tc):
    print('\nGlobal figure of dynamic vs quasistatic evolution')    
    print('\nfinal W0  = %6.3f  W1 = %6.3f' % (Wf0[-1], Wf1[-1]))
        
    fig_ls = 1   # linewidth
    fig_ms = 0.25  # markersize
    fig_fontsize = 10
    
    tdata  = np.arange(tc+1)            
    
    Hmv0, Hmv1 = np.zeros((tc+1)), np.zeros((tc+1))
    for ti in range(tc+1):
        x = max(1e-9, Pmv0[ti])
        x = min(x, 1-1e-9)
        Hmv0[ti] = - x * np.log(x) - (1-x) * np.log(1-x)
        x = max(1e-9, Pmv1[ti])
        x = min(x, 1-1e-9)
        Hmv1[ti] = - x * np.log(x) - (1-x) * np.log(1-x)
    
    fig,ax = plt.subplots(figsize=(4*fig_size,2*fig_size))
    
    ax.plot(tdata,Qqs0,'r-',linewidth=fig_ls,label='Sqs')    
    ax.plot(tdata,Qmv0,'ro',linewidth=fig_ls,markersize=fig_ms,label='Ssim')
    ax.plot(tdata,Qmv1,'ro',linewidth=fig_ls,markersize=fig_ms)
    
    ax.plot(tdata,Hqs0,'g-',linewidth=fig_ls,label='Hqs')    
    ax.plot(tdata,Hmv0,'go',linewidth=fig_ls,markersize=fig_ms,label='Hsim')
    ax.plot(tdata,Hmv1,'go',linewidth=fig_ls,markersize=fig_ms)
    
    ax.plot(tdata,Pqs1,'b-', linewidth=fig_ls,label='Pqs1')
    ax.plot(tdata,Pqs0,'b--',linewidth=fig_ls,label='Pqs0')
    
    ax.plot(tdata[1:],Pmv0[1:],'bo',linewidth=fig_ls,markersize=fig_ms,label='Psim')
    ax.plot(tdata[1:],Pmv1[1:],'bo',linewidth=fig_ls,markersize=fig_ms)

    plt.xticks([0,tc/2,tc],['0','tc/2','tc'],fontsize=fig_fontsize)
    plt.yticks([0,0.5,np.log(2),1,1.5,1.8],[0,0.5,'log2',1.0, 1.5,1.8],fontsize=fig_fontsize)
    plt.text(int(0.2*tc),-0.4,'phase 1',fontsize=fig_fontsize)
    plt.text(int(0.7*tc),-0.4,'phase 2',fontsize=fig_fontsize)
    plt.ylim(-0.2,2.)
    ax.legend(bbox_to_anchor=(1.15,0.8),fontsize=fig_fontsize)                    
    ax.grid()
    plt.show() 
    
def plot_evolution_of_Wf(ns0,ns):
    print('\nEvolution of W = f(ns)')            
    xdata  = np.arange(ns0,ns+1)
    Wfm = (Wf0+Wf1)/2   # mean value         
    fig,ax = plt.subplots(figsize=(3*fig_size,1.5*fig_size))    
    ax.plot(xdata,Wf1[ns0:],'r-',linewidth=0.8,label='Wf1')
    ax.plot(xdata,Wfm[ns0:],'b-',linewidth=1,label='Wfm')
    ax.plot(xdata,Wf0[ns0:],'g-',linewidth=0.8,label='Wf0')
    ax.legend(bbox_to_anchor=(1.2,0.8),fontsize=fig_fontsize)                    
    ax.grid()
    plt.show()                    
    
# ==============================
# Potential lansdcape parameters
# ==============================

fim  = 13                 # point M (0,fim, in kBT)
xa   = 2                  # point A well bottom (1e-6 m)
a0   = fim/xa**2          # coef well parabola  
xmin, xmax = -3*xa, 3*xa # limits of computing

# ==============================
# Potential lansdcape functions
# ==============================

# t : time (s)
# x : abcissa (nm)

def potential(x,u,v,fim): 
    # u varies from 0 to 1 when well from left to right 
    # v varies from 0 to 1 for well depth = v*fim
    xp = (2*u-1)*xa     # abcissa of well bottom
    if     x < -2*xa :  y = a0*(x+xa)**2 - fim
    elif   x < xp - xa: y = 0
    elif   x < xp + xa: y = v * (a0*(x-xp)**2 - fim)
    elif   x < 2*xa:    y = 0       
    else:               y = a0*(x-xa)**2 -fim
    return y

# ===========================================
# Potential derivate dy/dx
# ===========================================

def potential_derivate(x,u,v): # potential derivate
    xp = (2*u-1)*xa
    if     x < -2*xa :  dy = 2*a0*(x+xa)
    elif   x < xp - xa: dy = 0
    elif   x < xp + xa: dy = 2 * v * a0*(x-xp)
    elif   x < 2*xa:    dy = 0       
    else:               dy = 2*a0*(x-xa)
    return dy

# ==========================
# Physical parameters
# =========================
"""
diffusion coefficient cd: Jun-2014 proposes 1.7 
but direct calculation gives 2.5 for water
"""
dt  = 0.01              # period of landscape update (s) 
cd  = 2.5               # diffusion coefficient
cb = np.sqrt(2*cd/dt)   # brownian movement coefficient
           
# ==========================
# Simulation parameters
# ========================= 

tc  = 900          # cycle time (s)
nit = int(tc/dt)   # nb of iterations per simulation
dne = int(1/dt)    # to store results every second   
ne  = int(nit/dne) # for sampling every second

# ==========================
# Process functions
# =========================

def state_variables(u,v,fim, potential): 
    """
    u and v vary from 0 to 1
    returns :               
    Ut : memory potential
    St : statistical entropy 
    Zt : Boltzmann repartition function 
    Pt : probability of state 1    
    """    
    dx = 0.001
    Ut, Zt, Z0, Z1 = 0,0,0,0     
    nxi = int((xmax-xmin)/dx)
    for ii in range(nxi):
        xi = xmin + ii*dx
        fi = potential(xi,u,v,fim)
        expt = np.exp(-fi)
        if   ii < int(nxi/2):  Z0 += expt
        elif ii > int(nxi/2):  Z1 += expt
        elif ii == int(nxi/2): Z0 += expt/2; Z1 += expt/2
        Ut += fi * expt
    Zt = Z0 + Z1
    Ut = Ut/Zt
    Pt = Z1/Zt       
    St = np.log(Zt)+ Ut
    return Ut,St,Zt,Pt 

def quasistatic_reset_to_zero(bit,tc): 
    """
    starting from bit = 0 or 1
    returns vectors of energy U, work W, statistical entropy S, heat Q, 
    proba state one P, information entropy H
    """    
    Uqs,Wqs,Sqs,Qqs,Pqs,Hqs = np.zeros((tc+1)), np.zeros((tc+1)),np.zeros((tc+1)),np.zeros((tc+1)),np.zeros((tc+1)),np.zeros((tc+1))

    # initial state for t = 0
    Ui,Si,Zi,Pi = state_variables(u=bit,v=1,fim=fim,potential=potential)
    
    print ('\nComputing quasistatic reset_to_zero for bit = ',bit)

    for ti in range(tc+1):
        
        if ti < tc/2: # flattening well
            ui = bit
            vi = 1 - 2 * ti/tc # vi varies from 1 to 0             
            Ut, St, Zt, Pt = state_variables(ui,vi,fim,potential)
            Wt = -np.log(Zt) + np.log(Zi) 
            Qt = St - Si
            St = St - Si + np.log(2)
            Ptbis = min(1-1e-9, Pt)
            Ht = -Pt*np.log(max(Ptbis,1e-9)) - (1-Pt)*np.log(max(1-Ptbis,1e-9))
            Uqs[ti],Wqs[ti],Sqs[ti],Qqs[ti],Pqs[ti],Hqs[ti] = Ut,Wt,St,Qt,Pt,Ht
            
        else:   # creating bit = 0
            ui = 0
            vi =  2 * (ti-tc/2) / tc # vi varies from 0 to 1             
            Ut, St, Zt, Pt = state_variables(ui,vi,fim,potential)
            Wt = -np.log(Zt) + np.log(Zi) 
            Qt = St - Si
            St = St - Si + np.log(2)
            Ptbis = min(1-1e-9, Pt)
            Ht = -Pt*np.log(max(Ptbis,1e-9)) - (1-Pt)*np.log(max(1-Ptbis,1e-9))
            Uqs[ti],Wqs[ti],Sqs[ti],Qqs[ti],Pqs[ti],Hqs[ti] = Ut,Wt,St,Qt,Pt,Ht
                    
    return Uqs,Wqs,Sqs,Qqs,Pqs,Hqs

                                                      
def one_dynamic_reset_to_zero(tc,bit): #  
    """
    one run for starting bit = 0 or 1 
    returns vector of U, Q, W, P for every second
    """                           
    Uv = np.zeros((ne+1))
    Qv = np.zeros((ne+1))
    Wv = np.zeros((ne+1)) 
    Pv = np.zeros((ne+1))

    # starting from - xa or xa => equilibrium after 2000 steps (20 s)
    xi = pre_start_phase(2000,bit) # ùù
             
    if xi < 0: Pv[0]  = 0
    else:      Pv[0]  = 1                   
    
    ti = 0
    Wi, Qi = 0, 0
    
    dxp,dxb,dQi,dUi = 0,0,0,0  # variations during every dt increment

    loc_disp = False           # option to display intermediate results
 
    if loc_disp : 
        print ('\none_dynamic_reset_to_zero')
        print ('%8s  %8s  %8s  %8s  %8s  %8s  %8s  %8s  %8s' 
               % ('ti','xi','dxp','dxb','dQi','dUi','Qi','Wi','U'))
    
    for ni in range(1,nit+1): # nit+1
        if ni%dne == 0:  # sampling every second
            nj = int(ni/dne)
            Uv[nj] = potential(xi,ui,vi,fim) # ui not = bit ùù
            Qv[nj] = Qi
            Wv[nj] = Wi
            if xi < 0: Pv[nj]  = 0
            else:      Pv[nj]  = 1                   
            if loc_disp and False: 
                print ('%8.1f  %8.3f  %8.4f  %8.4f  %8.3f  %8.3f  %8.3f  %8.3f  %8.3f' 
                       % (ti,xi,dxp,dxb,dQi,dUi,Qi,Wi,Uv[nj]))
        if ni <= int(nit/2):    # first phase    
            ui = bit
            vi = 1 - 2*ni/nit
            dv = - 2/nit
        else:                   # second phase 
            ui = 0
            vi = 2*(ni-nit/2)/nit
            dv = 2/nit
            
        pu  = potential_derivate(xi,ui,vi)                 
        ti += dt
        dxp = - cd*pu *dt              # potential part
        dxb = cb*np.random.randn() *dt # thermal part
        dxi = dxp+dxb
        xi += dxi            
        dQi = potential(xi,ui,vi-dv,fim) - potential(xi-dxi,ui,vi-dv,fim)
        Qi += dQi
        dUi = potential(xi,ui,vi,fim) - potential(xi-dxi,ui,vi-dv,fim)
        dWi = dUi-dQi
        Wi += dWi

        if loc_disp and ni%10000 == 0:
            print ('%8.2f  %8.3f  %8.3f  %8.3f  %8.3f  %8.3f  %8.3f  %8.3f  %8.3f' 
                   % (ti,xi,dxp,dxb,dQi,dUi,Qi,Wi,potential(xi,ui,vi,fim)))
            if ni == int(nit/2): print()
            
    if loc_disp and False:
        print()
        print ('%8s  %8s  %8s  %8s  %8s  %8s  %8s  %8s  %8s' 
               % ('ti','xi','dxp','dxb','dQi','dUi','Qi','Wi','U'))
        print ('%8.1f  %8.3f  %8.4f  %8.4f  %8.3f  %8.3f  %8.3f  %8.3f  %8.3f'
               % (ti,xi,dxp,dxb,dQi,dUi,Qi,Wi,Uv[nj]))
        print()
                
    return Uv,Qv,Wv,Pv


def pre_start_phase(nt,bit): 
    """
    initial x after waiting for equilibrium of memory for nt iterations
    """
    # starting point
    if bit == 0: xi = -xa
    else :       xi = xa        
    ti = 0
    for ni in range(nt):                         
        pu  = potential_derivate(x=xi,u=bit,v=1)                 
        ti += dt
        dxp = - cd*pu *dt              # potential part
        dxb = cb*np.random.randn() *dt # thermal part
        dxi = dxp+dxb
        xi += dxi            
    return xi


def batch_of_dynamic_reset_to_zeros(tc,ns):
    """
    returns average of U, Q, W, P at every second 
    for ns simulations
    """
                  
    print ('\nComputing dynamic_evolution for ',ns,' simulations')
    
    tv  = np.arange(ne+1)
    
    # mean values = f(t) for ns simulations for starting bit = 0 and 1           
    Umv0, Qmv0, Wmv0, Pmv0 = np.zeros((ne+1)),np.zeros((ne+1)),np.zeros((ne+1)),np.zeros((ne+1))
    Umv1, Qmv1, Wmv1, Pmv1 = np.zeros((ne+1)),np.zeros((ne+1)),np.zeros((ne+1)),np.zeros((ne+1))
    
    # mean W after n simulations
    Wf0 = np.zeros((ns+1)) # for si = 1 to ns
    Wf1 = np.zeros((ns+1)) # for si = 1 to ns
    
    # values at end of phases 
    U1, Q1, W1, P1 = np.zeros((ns)),np.zeros((ns)),np.zeros((ns)),np.zeros((ns))
    U2, Q2, W2, P2 = np.zeros((ns)),np.zeros((ns)),np.zeros((ns)),np.zeros((ns))
    
    print ('\n%4s  %8s  %8s  %8s  %8s  %8s  %8s  %8s' % 
           ('si', 'Umf0','Umf1','Qmf0','Qmf1','Wmf0','Wmf1','time')) 
    xx, yy = 0,0
    
    # MAIN COMPUTING W = f(ns)
    
    for si in range(ns):
        
        # one run per starting bit       
        Uv0,Qv0,Wv0,Pv0 = one_dynamic_reset_to_zero(tc=tc,bit=0) 
        Uv1,Qv1,Wv1,Pv1 = one_dynamic_reset_to_zero(tc=tc,bit=1) 
                
        Umv0 += Uv0; Qmv0 += Qv0; Wmv0 += Wv0; Pmv0 += Pv0
        Wf0[si+1] = Wmv0[-1] / (si+1)

        Umv1 += Uv1; Qmv1 += Qv1; Wmv1 += Wv1; Pmv1 += Pv1
        Wf1[si+1] = Wmv1[-1] / (si+1)
        
        if si%50 == 0 or si == ns-1:
            timex = time.perf_counter() - start
            print ('%4i  %8.3f  %8.3f  %8.3f  %8.3f  %8.3f  %8.3f  %8.1f'
                   % (si, Umv0[-1]/(si+1), Umv1[-1]/(si+1),
                          Qmv0[-1]/(si+1), Qmv1[-1]/(si+1),
                          Wmv0[-1]/(si+1), Wmv1[-1]/(si+1), timex))
    
    # END OF MAIN COMPUTING
    
    # means
    Umv0 = Umv0/ns; Qmv0 = Qmv0/ns; Wmv0 = Wmv0/ns; Pmv0 = Pmv0/ns
    Umv1 = Umv1/ns; Qmv1 = Qmv1/ns; Wmv1 = Wmv1/ns; Pmv1 = Pmv1/ns
    
    return Umv0, Qmv0, Wmv0, Pmv0, Wf0, Umv1, Qmv1, Wmv1, Pmv1, Wf1


if __name__ == "__main__":
    
    start = time.perf_counter()
    
    print()
    print ('Single_well_memory_v10',datetime.now().strftime("%d/%m/%Y at %H:%M:%S"))
   
    todo_list = [0,1,2]
    
    # 0 parameters
    if 0 in todo_list:
        display_parameters()
           
    #  quasistatic_reset_to_zero                            
    if 1 in todo_list:
        bit = 0
        Uqs0,Wqs0,Sqs0,Qqs0,Pqs0,Hqs0 = quasistatic_reset_to_zero(bit,tc) 
        bit = 1
        Uqs1,Wqs1,Sqs1,Qqs1,Pqs1,Hqs1 = quasistatic_reset_to_zero(bit,tc) 
                
        plot_quasistatic_reset_to_zero(tc)
                    
    # dynamic reset-to-zero  COMPUTING TIME = about 20 mn for ns = 500
    if 2 in todo_list:
        ns = 500
        Umv0, Qmv0, Wmv0, Pmv0, Wf0, Umv1, Qmv1, Wmv1, Pmv1, Wf1 = \
                                          batch_of_dynamic_reset_to_zeros(tc,ns)    
        plot_both_reset_to_zeros(tc)
        plot_evolution_of_Wf(ns0=50,ns=ns) 
            
                               
    print ("\nend of Single_well_memory_v10 after %6.3f s" % (time.perf_counter() - start),'\n')




