# -*- coding: utf-8 -*-
"""
Tilting_bistable_memory_v10    8/04/2026

Computation code for quasistatic and dynamic evolution 
of a tilting bistable memory (experiments of Jun-2014)

Units : energy in kBT, entropy in kB, others MKSA 

computation time about 40 mn (with AMD Ryzen 7 processor)

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
fig_lw        = 0.8
mpl.rcParams.update({"figure.dpi": 100, "savefig.dpi": 300,}) 

# ==========================
# Utility functions
# ==========================
                
def display_parameters():
    print ('\nLandscape parameters')
    print ('fim   = %6i  maximum potential phi (kBT)'% fim)
    print ('xm    = %6.1f  position of right well bottom (nm)' % xm)
    print ('omega = %6.4f angular frequency (rd/s)  '% omega)
   
    print ('\nProcess parameters')
    print ('tc    = %6i  cycle time (s)'% tc)
    print ('ns    = %6i  number of simulations'% ns)    
    print ('dt    = %6.2f  time increment (s)'% dt)
    print ('cd    = %6.1f  diffusion coefficient'% cd)
    print ('cb    = %6.1f  brownian movement coefficient' % cb)

def plot_ff_and_gg_functions(): 
    print('\nFigure functions (ff(t) and gg(t)')
    tdata = np.arange(0,tc)
    fig,ax = plt.subplots(figsize=(2*fig_size,0.6*fig_size))
    ax.plot(tdata,[ff(t) for t in tdata],'b-',linewidth=fig_lw,label='f(t)')
    ax.plot(tdata,[gg(t) for t in tdata],'r--',linewidth=fig_lw,label='g(t)')
    plt.legend(bbox_to_anchor=(1.25,0.5),fontsize=fig_fontsize)
    plt.xticks(np.arange(0,950,235),['0','tc/4','tc/2','3tc/4','tc'],fontsize=fig_fontsize)
    plt.yticks(fontsize=fig_fontsize)
    ax.grid()
    plt.show()

def plot_potential_landscape(): 
    print('\nFigure potential(x,t)')
    fig,ax = plt.subplots(figsize=(2*fig_size,1.2*fig_size))                       
    xdata = np.arange(-6,6,0.01)
    for ii in range(4):
        xii, yii = [],[]                
        for x in xdata:
            y = potential(x,ii*tc/4)
            if y < 45: xii.append(x); yii.append(y)
        ax.plot(xii,yii,['b-','r--','g-.','y:'][ii],linewidth=[1,1,1,2][ii],
                label=['t=0','tc/4','tc/2','3tc/4'][ii])
    plt.legend(bbox_to_anchor=(1,0.6),fontsize=fig_fontsize)
    plt.xticks(fontsize=fig_fontsize)
    plt.yticks(fontsize=fig_fontsize)
    plt.xlabel(r'x ($\mu$m)',fontsize=fig_fontsize)
    plt.ylabel("U (kB T)",fontsize=fig_fontsize)
    plt.ylim(-50,50)
    ax.grid()
    plt.show() 

def plot_work_for_ns_simulations():
    print ('\nFigure W = f(si)')
    fig,ax = plt.subplots(figsize=(3*fig_size,1.5*fig_size))
    ax.plot(np.arange(10,ns),[np.sum(Wf[:si])/si for si in range(10,ns)],'b--',linewidth=fig_lw,label='W')
    plt.ylim(0.6,0.8)
    plt.xticks(fontsize=fig_fontsize)
    plt.yticks(fontsize=fig_fontsize)
    plt.xlabel('number of runs',fontsize=fig_fontsize)
    plt.ylabel('kB T',fontsize=fig_fontsize)
    ax.legend(bbox_to_anchor=(1.0,0.6),fontsize=fig_fontsize)
    ax.grid()
    plt.show() 
               
def plot_evolution_of_S_H_P_Q():     
    print ('\nFigure evolution of S,H,P,Q')
    fig,ax = plt.subplots(figsize=(3*fig_size,1.5*fig_size))
    tdata = np.arange(tc+1)
    ax.hlines(np.log(2),0,tc+1,'k',linestyle='dashed',linewidth=0.8,label='+/- log2')
    ax.hlines(-np.log(2),0,tc+1,'k',linestyle='dashed',linewidth=0.8)
    ax.plot(tdata,Sqs,'b-', linewidth=fig_lw, label='  Sqs')
    ax.plot(tdata,Hqs,'k-.',linewidth=fig_lw, label='  Hqs')
    ax.plot(tdata,Pqs,'g--', linewidth=fig_lw, label='   Pqs')
    ax.plot(tdata,Pmv,'g-', linewidth=fig_lw, label='  Psim')    
    ax.plot(tdata,Qqs,'r--', linewidth=fig_lw, label='   Qqs')
    ax.plot(tdata,Qmv,'r-',linewidth=0.5,markersize=0.5,label=' Qsim')
    
    for ii in range(5):
        plt.vlines(235*ii,-1,1.5,color='k', linestyle='--', lw=0.8)
    plt.xticks(np.arange(0,950,235),fontsize=fig_fontsize)
    plt.xticks(np.arange(0,950,235),[0,'tc/4','tc/2','3tc/4','tc'],fontsize=fig_fontsize)
    plt.ylim(-1.2,1.6)
    plt.xlabel("t",fontsize=fig_fontsize)
    plt.ylabel("kB T",fontsize=fig_fontsize) 
    ax.legend(bbox_to_anchor=(1.05,0.8),fontsize=fig_fontsize)
    plt.grid() 
    plt.show()    
      
def plot_evolution_of_W_and_U(): 
    tdata = np.arange(tc+1)
    fig,ax = plt.subplots(figsize=(3*fig_size,1.5*fig_size))
    print ('\nFigure evolution of W and U')
    ax.plot(tdata,Wqs,'b-',linewidth=fig_lw,label='Wqs')
    ax.plot(tdata,Wmv,'b--',linewidth=fig_lw,label='Wmv')
    ax.plot(tdata,Uqs,'r-',linewidth=fig_lw,label='Uqs')
    ax.plot(tdata,Umv,'r--',linewidth=fig_lw,label='Umv')
    plt.xticks(np.arange(0,950,235),[0,'tc/4','tc/2','3tc/4','tc'],fontsize=fig_fontsize)
    for ii in range(5):
        plt.vlines(235*ii,-40,15,color='k', linestyle='--', lw=0.8)
    plt.xticks(np.arange(0,950,235),fontsize=fig_fontsize)
    plt.yticks(fontsize=fig_fontsize)
    plt.xlabel("t",fontsize=fig_fontsize)
    plt.ylabel("kB T",fontsize=fig_fontsize) 
    ax.legend(bbox_to_anchor=(1.2,0.6),fontsize=fig_fontsize)
    ax.grid()
    plt.show()     

# ==============================
# Potential lansdcape parameters
# ==============================

fim = 13                 # maximum potential phi (kBT)
xm  = 2.5                # position of right well bottom (nm)
xmin, xmax = -2*xm, 2*xm # limits of computing
tc    = 940              # cycle time (s)
omega = 2*np.pi/tc       # angular frequency (rad/s)
    
# ==============================
# Potential lansdcape functions
# ==============================

# t : time (s)
# x : abcissa (nm)

def ff(t): 
    # Jun-2014 function f(t)
    if   t <= tc/4:    y = 0 
    elif t <= tc/2:    y = 4/tc*(t-tc/4)
    elif t <= 3*tc/4:  y = 1
    else:              y = 1- 4/tc*(t-3*tc/4)
    return y

def gg(t): 
    # Jun-2014 function g(t)
    if   t <= tc/4:    y = 1 - np.sin(omega*t)
    elif t <= tc/2:    y = 0
    elif t <= 3*tc/4:  y = 1 - np.sin(omega*(t-tc/4))
    else:              y = 1
    return y
        
def potential(x,t): # 
    # Jun-2014 potential function   
    xr = x/xm
    return fim*xr*(xr*(xr**2 -2*gg(t)) -2*ff(t))

def potential_derivate(x,t): 
    # partial derivate over x
    xr = x/xm
    return 4*fim/xm*(xr*(xr**2 -gg(t)) -ff(t)/2)


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
 
ns  = 1440          # nb of simulations
                    # COMPUTING TIME = about 40 mn for ns = 1440

nit = int(tc/dt)    # # nb of iterations per simulation
dne = int(1/dt)     # to store results every second        

# ==========================
# Process functions
# ========================= 

def state_variables(t, potential): 
    """
    computes state variables of memory for a given potential
    t         : time
    potential : potential function (x,t)
    x         : abscissa between xmin and xmax
    dx        : x increment
   
    return :               
    Ut : memory potential
    St : statistical entropy 
    Zt : Boltzmann repartition function 
    Pt : probability of state 1
    
    """
    dx = 0.01
    Ut, Zt = 0,0                # Zt = repartition function
    nxi = int((xmax-xmin)/dx)
    for ii in range(nxi):
        xi = xmin + ii*dx
        fi = potential(xi,t)
        expt = np.exp(-fi)
        Zt += expt
        Ut += fi * expt
        if ii == int(nxi/2): # proba state 1
            Pt = Zt
    Ut = Ut/Zt
    Pt = 1-Pt/Zt
    St = np.log(Zt)+ Ut
    return Ut,St,Zt,Pt 

def one_dynamic_simulation(x0):
    """                              
    one run for starting point = x0 
    returns vector of U, Q, W, P for every second
    """
    Uv = np.zeros((tc+1))   # memory potential
    Qv = np.zeros((tc+1))   # heat received from thermostat
    Wv = np.zeros((tc+1))   # work provided by actuator
    Pv = np.zeros((tc+1))   # probability of state 1
    
    xi = x0
    ti = 0
    Wi, Qi = 0, 0
    dxp,dxb,dQi,dUi = 0,0,0,0 # variations during every dt increment

    loc_disp = False          # option to display intermediate results
 
    if loc_disp : 
        print ('%8s  %8s  %8s  %8s  %8s  %8s  %8s  %8s  %8s' 
               % ('ti','xi','dxp','dxb','dQi','dUi','Qi','Wi','U'))
    
    for ni in range(1,nit+1): # 
        if ni%dne == 0: # to store results every second
            nj = int(ni/dne)
            Uv[nj] = potential(xi,ti)
            Qv[nj] = Qi
            Wv[nj] = Wi
            if xi < 0: Pv[nj]  = 0
            else:      Pv[nj]  = 1                   
            if loc_disp and (nj%100 == 0 or ni == nit): 
                print ('%8.1f  %8.3f  %8.4f  %8.4f  %8.3f  %8.3f  %8.3f  %8.3f  %8.3f' 
                       % (ti,xi,dxp,dxb,dQi,dUi,Qi,Wi,Uv[nj]))
                
        # MAIN LOOP        
        ti += dt
        dxp = - cd*potential_derivate(xi,ti) *dt  # potential part
        dxb = cb*np.random.randn() *dt            # thermal part
        dxi = dxp+dxb
        xi += dxi            
        dQi = potential(xi,ti-dt) - potential(xi-dxi,ti-dt)
        Qi += dQi
        dUi = potential(xi,ti) - potential(xi-dxi,ti-dt)
        dWi = dUi-dQi
        Wi += dWi
       # END OF MAIN LOOP

       # end of for ni
         
    return Uv,Qv,Wv,Pv
    
def batch_of_dynamic_simulations():
    """
    returns average values of variables Umv,Qmv,Wmv,Pmv,
    every second for ns simulations
    """
    def pre_start_phase(nt,x0): 
        """
        returns position x after free evolution from x0 for nt iterations 
        """
        xi = x0
        ti = 0
        for ni in range(nt):                         
            ti += dt
            dxp = - cd * potential_derivate(xi,0) * dt # potential part
            dxb = cb * np.random.randn() * dt          # thermal part
            dxi = dxp + dxb
            xi += dxi            
        return xi
    
    print ('\nComputing dynamic_evolution for ',ns,' simulations')
           
    # computing starting points for all simulations                                               
    x0v = np.zeros((ns)) # vector of starting positions for ns simulations
    for ni in range(ns): 
        # starting alternatively from left or right bottom
        if ni%2 == 0:   x0 = -xm
        else:           x0 =  xm 
        # free evolution for 20 seconds                                                          
        x0v[ni] = pre_start_phase(2000,x0)
        
    print ('\nmean value of x0 = %6.4f' % np.mean(x0v))
                
    # preparing vectors of mean values for ns simulations                         
    tv  = np.arange(tc+1)
    Umv, Qmv, Wmv, Pmv = np.zeros((tc+1)),np.zeros((tc+1)),np.zeros((tc+1)),np.zeros((tc+1))
    
    # final value of Wm for every si
    Wf = np.zeros((ns))
    
    print ('\nmultirun ns = ',ns)
    print ('\n%4s  %8s  %8s  %8s  %8s' % ('si', 'Umf','Qmf','Wmf','time')) 
    
    # MAIN COMPUTING 40 mn
    for si in range(ns):
        Uv,Qv,Wv,Pv = one_dynamic_simulation(x0v[si]) 
        Umv += Uv
        Qmv += Qv
        Wmv += Wv
        Pmv += Pv
        if si%100 == 0:
            timex = time.perf_counter() - start
            print ('%4i  %8.3f  %8.3f  %8.3f  %8.1f'
                   % (si, Umv[-1]/(si+1),Qmv[-1]/(si+1), Wmv[-1]/(si+1), timex))
        Wf[si] = Wv[-1]
        # end of for si
     # END OF MAIN COMPUTING 
     
    print ('%4i  %8.3f  %8.3f  %8.3f  %8.1f'
           % (si, Umv[-1]/(si+1),Qmv[-1]/(si+1), Wmv[-1]/(si+1), timex))
        
    # mean values
    Umv = Umv/ns
    Qmv = Qmv/ns
    Wmv = Wmv/ns
    Pmv = Pmv/ns
    
    return Umv,Qmv,Wmv,Pmv,Wf  
    
def quasistatic_evolution():
    """
    returns 6 arrays of variable values at every second 
    U, W, Q, P : momory potential, work, heat, probability of state 1
    S : statistical entropy
    H : information entropy
    """                              
    Uqs, Wqs, Sqs, Qqs, Pqs, Hqs = [np.zeros(tc + 1) for _ in range(6)]

    # initial state for t = 0
    Ui,Si,Zi,Pi = state_variables(0, potential)

    for ti in range(tc+1):
        Ut, St, Zt, Pt = state_variables(ti, potential)
        Wt = -np.log(Zt) + np.log(Zi) 
        Qt = St - Si
        St = St - Si + np.log(2)
        Ptbis = min(1-1e-9, Pt)
        Ht = -Ptbis*np.log(max(Ptbis,1e-9)) - (1-Ptbis)*np.log(max(1-Ptbis,1e-9))
        Uqs[ti],Wqs[ti],Sqs[ti],Qqs[ti],Pqs[ti],Hqs[ti] = Ut,Wt,St,Qt,Pt,Ht
        
    return Uqs, Wqs, Sqs, Qqs, Pqs, Hqs
 
    
if __name__ == "__main__":

    start = time.perf_counter()
    
    print ('\nTilting_bistable_memory_v10',datetime.now().strftime("%d/%m/%Y at %H:%M:%S"))
                       
    # parameters
    display_parameters() 
    
    # plot functions defining the potential landscape
    plot_ff_and_gg_functions()
    plot_potential_landscape()
    
    # dynamic simulation               
    Umv,Qmv,Wmv,Pmv,Wf = batch_of_dynamic_simulations()        
    plot_work_for_ns_simulations()
    
    # quasistatic evolution                             
    Uqs, Wqs, Sqs, Qqs, Pqs, Hqs = quasistatic_evolution()
    
    # plot results            
    plot_evolution_of_S_H_P_Q()
    plot_evolution_of_W_and_U() 
            
    
    print ("\nend of Tilting_bistable_memory_v10 after %6.3f s" % (time.perf_counter() - start),'\n')

    