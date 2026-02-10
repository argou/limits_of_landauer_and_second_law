# -*- coding: utf-8 -*-
"""
Shifting_bistable_memory_v10    10/02/2026

Computation code for quasistatic and dynamic evolution 
of a shifting bistable memory

Units : energy in kBT, entropy in kB, others MKSA 

computation time about 30 mn (with AMD Ryzen 7 processor)

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
    print ('xa    = %6.3f  half width of large parabola bottom' % xa)
    print ('ka    = %6.3f  coef small parabola' % ka)
    print ('xn    = %6.3f  half width small parabola' % xn)
    print ('a0    = %6.3f  coef well parabola' % a0)
    print ('b0    = %6.3f  coef small parabola' % b0)
    print ('xb    = %6.3f  point B well bottom' % xb)
    print ('xc    = %6.3f  point C such as yc = fim/(1+ka)' % xc)
    print ('xd    = %6.3f  point D such as yd = fim' % xd)
    print ('xbd   = %6.3f  half width well without barrier' % xbd)
    print ('ulim  = %6.3f  right limit of small parabola' % ulim)
    
    print ('\nProcess parameters')
    print ('tc    = %6i  cycle time (s)'% tc)
    print ('ns    = %6i  number of simulations'% ns)    
    print ('dt    = %6.2f  time increment (s)'% dt)
    print ('cd    = %6.1f  diffusion coefficient'% cd)
    print ('cb    = %6.1f  brownian movement coefficient' % cb)
        
    
def plot_landscape_of_shifting_bistable_memory(): 
    print ('\nFigure shifting bistable memory')
    fig,ax = plt.subplots(figsize=(4*fig_size,0.6*fig_size))

    # to remove edges and grid
    fig.patch.set_visible(False)
    ax.axis('off')
    
    loc_lw = 0.8
    xh     = 8
    xmax   = xh+1
    dx     = 17
    xdata = np.arange(-xmax,xmax,0.01)                    
    ax.plot(xdata+0*dx,[phi_flatten(x,u=0)   for x in xdata],'b-',linewidth=loc_lw,label='phi3')            
    ax.plot(xdata+0*dx,[phi_flatten(x,u=0.5) for x in xdata],color='blue',linestyle='dotted',linewidth=loc_lw,label='phi3')            
    ax.plot(xdata+0*dx,[phi_flatten(x,u=1)   for x in xdata],'b--',linewidth=loc_lw,label='phi3')            

    ax.plot(xdata+1*dx,[phi_shift(x,u=0)   for x in xdata],'b-',linewidth=loc_lw,label='phi3')            
    ax.plot(xdata+1*dx,[phi_shift(x,u=0.5) for x in xdata],color='blue',linestyle='dotted',linewidth=loc_lw,label='phi3')            
    ax.plot(xdata+1*dx,[phi_shift(x,u=1)   for x in xdata],'b--',linewidth=loc_lw,label='phi3')            
    
    ax.plot(xdata+2*dx,[phi_restore(x,u=0)   for x in xdata],'b-',linewidth=loc_lw,label='phi3')            
    ax.plot(xdata+2*dx,[phi_restore(x,u=0.5) for x in xdata],color='blue',linestyle='dotted',linewidth=loc_lw,label='phi3')            
    ax.plot(xdata+2*dx,[phi_restore(x,u=1)   for x in xdata],'b--',linewidth=loc_lw,label='phi3')            
    plt.xlim(-0.8*xmax, 5.5*xmax)
    plt.ylim(-1,25)    
    plt.xticks(fontsize=fig_fontsize)
    plt.yticks(fontsize=fig_fontsize)
    plt.xlabel('x',fontsize=fig_fontsize)
    ax.grid()
    plt.show()

def plot_evolution_of_W_P_Q():
    print('\nGlobal figure of dynamic vs quasistatic evolution')    
    print('\nfinal W  = %6.3f' % (Wmv[-1]))
        
    fig_ls       = 0.8
    fig_ls2      = 1.0
    tdata        = np.arange(tc+1)            
    
    fig,ax = plt.subplots(figsize=(3*fig_size,1.5*fig_size))
    ax.plot(tdata,Wqs,'g--',linewidth=fig_ls,label='Wqs')
    ax.plot(tdata,Wmv,'g-',linewidth=fig_ls2,markersize=0.5,label='Wsim')                                
    ax.plot(tdata,Pqs,'b-.',linewidth=fig_ls2,label='Pqs')
    ax.plot(tdata[1:],Pmv[1:],'b-',linewidth=fig_ls2,markersize=0.3,label='Psim')
    ax.plot(tdata,Qqs,'r:',linewidth=fig_ls2,label='Qqs')
    ax.plot(tdata,Qmv,'r-',linewidth=fig_ls2,markersize=0.5,label='Qsim')
    plt.xticks(np.arange(0,tc+10,tc/3),[0,'tc/3','2tc/3','tc'],fontsize=fig_fontsize)
    plt.yticks([-1,-np.log(2),0,0.5,np.log(2),1, 1.5],[-1.0,'-log2',0,0.5,'log2',1.0, 1.5],fontsize=fig_fontsize)
    plt.text(120,-1.6,'phase 1',fontsize=fig_fontsize)
    plt.text(520,-1.6,'phase 2',fontsize=fig_fontsize)
    plt.text(920,-1.6,'phase 3',fontsize=fig_fontsize)
    plt.ylim(-1.3,1.)
    ax.legend(bbox_to_anchor=(1.02,0.8),fontsize=fig_fontsize)                    
    ax.grid()
    plt.show() 
    
def plot_W_during_phase3_for_different_durations(Wtab):               
    print ('\nFigure W during phase 3 for different cycle times tc')              
    
    language = 0
    if language == 0: # english version
        tc_names = ['3 years','1 year','1 month','1 week','1 day','1 hour']
    else:             # french version
        tc_names = ['3 ans','1 an','1 mois','1 semaine','1 jour','1 heure']

    fig,ax = plt.subplots(figsize=(3*fig_size,1.5*fig_size))

    plt.hlines(np.log(2), 0, tc, color='black',lw=1,
               linestyle='--',label='log2')
    colors = ['b','b','b','r','r','r']
    linestyles = ['-','-','--','-','--',':']
    tcn, loc_tc = Wtab.shape[0], Wtab.shape[1]
    for tci in range(1, tcn):
        ax.plot(np.arange(loc_tc),Wtab[tci,:],lw=1,
            color=colors[tci],linestyle=linestyles[tci],label=tc_names[tci])
        
    plt.legend(bbox_to_anchor=(1,0.8),fontsize=fig_fontsize)
    plt.xticks(np.arange(0,loc_tc+100,200),[0,0.2,0.4,0.6,0.8,1],fontsize=fig_fontsize)
    plt.yticks(np.arange(0,0.9,0.2),[0,0.2,0.4,0.6,0.8],fontsize=fig_fontsize)
    
    plt.xlabel('t/tc',fontsize=fig_fontsize)
    plt.xlim(0,1050)
    plt.ylabel('W/kBT',fontsize=fig_fontsize)
    plt.ylim(0,0.8)
    ax.grid(lw=0.5)
    plt.show()  
        
def save_data(data, file_name):
    print ("\nSaving data in ", file_name) 
    pickle.dump(data, open(file_name,"wb"))        

def load_data(file_name):
    print ("\nRestoring data from ", file_name) 
    new_data = pickle.load(open(file_name,"rb"))
    return new_data 
    
# ==============================
# Potential lansdcape parameters
# ==============================

fim  = 13                 # point M (0,fim)
xa   = 2                  # half width of large parabola bottom
ka   = 0.625              # coef small parabola
xn   = ka*xa              # half width small parabola
a0   = fim/((1+ka)*xa**2) # coef well parabola  
b0   = a0/ka              # coef small parabola
xb   = (1+ka)*xa          # point B well bottom
xbd  = np.sqrt(fim/a0)    # half width well without barrier 
xc   = xb+xa              # point C such as yc = fim/(1+ka)
yc   = fim/(1+ka)
xd   = xb + xbd           # point D such as yd = fim
ulim = ka/2/(1+ka)        # right limit of small parabola 
xmin, xmax = -3*xa, 3*xa # limits of computing

# ==============================
# Potential lansdcape functions
# ==============================

# t : time (s)
# x : abcissa (nm)

def phi_double_well(x):
    # double well landscape
    if x < -ka*xa:  y = a0*(x+xb)**2
    elif x < ka*xa: y = fim - b0*x**2
    else:           y = a0*(x-xb)**2
    return y

def phi_flatten(x,u): # f
    # phase 1 - from double well u = 0 to flat bottom u = 1
    if x < -xb: y = a0*(x+xb)**2   # left parabola
    else:       y = (1-u)*phi_double_well(x) + u*phi_shift(x,u=0)
    return y
        
def phi_shift(x,u): # 
    # phase 2 - flat bottom u = 0 to single left well u = 1
    xp = xb*(1-2*u)        
    if x < -xb:      y = a0*(x+xb)**2   # left parabola 
    elif x < xp:     y = 0              # flat bottom
    elif x < xp+xa:  y = a0*(x-xp)**2   # right parabola
    else:                        
        if u < ulim: # 
            b_coef = a0*xp+b0*xd
            c_coef = a0*xp**2+b0*xd**2-fim
            delta = max(0,b_coef**2 - (a0+b0)*c_coef)
            xr = (b_coef+np.sqrt(delta))/(a0+b0)
            if   x < xr:    y = a0*(x-xp)**2   # beginning right parabola
            elif x < xd:    y =  fim - b0*(x-xd)**2
            else:           y = a0*(x-xb)**2 
        else:  # 
            if x < xp+xa:     y = a0*(x-xp)**2     # right small parabola
            elif x < xp+xb:   y = fim - b0*(x-xp-xb)**2 
            elif x < xd:      y = fim              # right plateau
            else:             y = a0*(x-xb)**2     # right large parabola                     
    return y 
    
def phi_restore(x,u): 
    # phase 3 - from single left well to double well (u = 0 to 1)
    if x < 0:           y = phi_double_well(x)
    elif x < (2+ka)*xb: y = (1-u)*phi_shift(x,u=1) + u*phi_double_well(x)
    else:               y = a0*(x-xb)**2
    return y

def potential(x,u): 
    # potential function for 3 phases
    # u varies from 0 to 3 
    if   u < 1:    y = phi_flatten( x, u)
    elif u < 2:    y = phi_shift(  x, u-1)
    else:          y = phi_restore(x, u-2)
    return y

# ===========================================
# Potential derivate functions in x
# ===========================================

def dphi_double_well(x):
    if x < -ka*xa:  dy = 2*a0*(x+xb)
    elif x < ka*xa: dy = - 2*b0*x
    else:           dy = 2*a0*(x-xb)
    return dy

def dphi_flatten(x,u): # phase 1
    if x < -xb: dy = 2*a0*(x+xb)   # left parabola
    else:       dy = (1-u)*dphi_double_well(x) + u*dphi_shift(x,u=0)
    return dy
        
def dphi_shift(x,u): # phase 2 
    xp = xb*(1-2*u)        
    if x < -xb:      dy = 2*a0*(x+xb)   # left parabola 
    elif x < xp:     dy = 0             # flat bottom
    elif x < xp+xa:  dy = 2*a0*(x-xp)   # beginning right parabola 
    else:                        
        if u < ulim: # 
            b_coef = a0*xp+b0*xd
            c_coef = a0*xp**2+b0*xd**2-fim
            delta = max(0, b_coef**2 - (a0+b0)*c_coef)
            xr = (b_coef+np.sqrt(delta))/(a0+b0)
            if   x < xr:    dy = 2*a0*(x-xp)   # beginning right parabola 
            elif x < xd:    dy = - 2*b0*(x-xd)
            else:           dy = 2*a0*(x-xb) 
        else:  # 
            if x < xp+xa:     dy = 2*a0*(x-xp)     # small right parabola 
            elif x < xp+xb:   dy = - 2*b0*(x-xp-xb) 
            elif x < xd:      dy = 0               # right plateau
            else:             dy = 2*a0*(x-xb)     # large right parabola                   
    return dy    

def dphi_restore(x,u): # phase 3 
    if x < 0:           dy = dphi_double_well(x)
    elif x < (2+ka)*xb: dy = (1-u)*dphi_shift(x,u=1) + u*dphi_double_well(x)
    else:               dy = 2*a0*(x-xb)
    return dy

def potential_derivate(x,u): # potential derivate for 3 phases
    if u < 1:   dy = dphi_flatten(x,u)
    elif u < 2: dy = dphi_shift(x,u-1)
    else:       dy = dphi_restore(x,u-2)
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

ns  = 1000  # nb of simulations
            # COMPUTING TIME = about 15 mn for ns = 1000
tc  = 1200   # cycle time (s)

nit = int(tc/dt)   # nb of iterations per simulation
dne = int(1/dt)    # to store results every second   
ne  = int(nit/dne) # for sampling every second

# ==========================
# Process functions
# =========================

def state_variables(u, potential): 
    """
    u varies from 0 to 3
    returns :               
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
        fi = potential(xi,u)
        expt = np.exp(-fi)
        Zt += expt
        Ut += fi * expt
        if ii == int(nxi/2): 
            Pt = Zt      # Z value for x < 0
    Ut = Ut/Zt
    Pt = 1-Pt/Zt       
    St = np.log(Zt)+ Ut
    return Ut,St,Zt,Pt 

def quasistatic_evolution(): 
    """
    returns vectors of energy U, work W, statistical entropy S, heat Q, 
    proba state one P, information entropy H
    initial state x = 0
    """
    
    Uqs,Wqs,Sqs,Qqs,Pqs,Hqs = np.zeros((tc+1)), np.zeros((tc+1)),np.zeros((tc+1)),np.zeros((tc+1)),np.zeros((tc+1)),np.zeros((tc+1))

    # initial state for t = 0
    Ui,Si,Zi,Pi = state_variables(0, potential)
    print ('\nComputing quasistatic evolution')
    print ('\n%4s  %8s  %8s  %8s  %8s  %8s  %8s  %8s' % ('pi','ti','Ut','St','Wt','Qt','Pt','Ht'))

    for ti in range(tc+1):
        ui = 3 * ti/tc # ici u varie de 0 à 3
        Ut, St, Zt, Pt = state_variables(ui, potential)
        Wt = -np.log(Zt) + np.log(Zi) 
        Qt = St - Si
        St = St - Si + np.log(2)
        Ptbis = min(1-1e-9, Pt)
        Ht = -Pt*np.log(max(Ptbis,1e-9)) - (1-Pt)*np.log(max(1-Ptbis,1e-9))
        Uqs[ti],Wqs[ti],Sqs[ti],Qqs[ti],Pqs[ti],Hqs[ti] = Ut,Wt,St,Qt,Pt,Ht
        if (ti)%(tc/3) == 0:
            pi = int(ti/(tc/3))
            print ('%4i  %8i  %8.3f  %8.3f  %8.3f  %8.3f  %8.3f  %8.3f' % (pi,ti,Ut,St,Wt,Qt,Pt,Ht)) 
        
    return Uqs,Wqs,Sqs,Qqs,Pqs,Hqs
  
                            
def one_dynamic_simulation(tc,x0): #  
    """
    one run for starting point = x0 
    returns vector of U, Q, W, P for every second
    """                           
    Uv = np.zeros((ne+1))
    Qv = np.zeros((ne+1))
    Wv = np.zeros((ne+1)) 
    Pv = np.zeros((ne+1))

    xi = x0 # init(2000,x0)
    ti = 0
    ui = 0
    Wi, Qi = 0, 0
    
    du  = 3*dt/tc      # u increment (u = 0 to 3) 
    dxp,dxb,dQi,dUi = 0,0,0,0  # variations during every dt increment

    loc_disp = False           # option to display intermediate results
 
    if loc_disp : 
        print ('%8s  %8s  %8s  %8s  %8s  %8s  %8s  %8s  %8s' 
               % ('ti','xi','dxp','dxb','dQi','dUi','Qi','Wi','U'))
    
    for ni in range(1,nit+1): # nit+1
        if ni%dne == 0:
            nj = int(ni/dne)
            Uv[nj] = potential(xi,ui)
            Qv[nj] = Qi
            Wv[nj] = Wi
            if xi < 0: Pv[nj]  = 0
            else:      Pv[nj]  = 1                   
            if loc_disp and False: 
                print ('%8.1f  %8.3f  %8.4f  %8.4f  %8.3f  %8.3f  %8.3f  %8.3f  %8.3f' 
                       % (ti,xi,dxp,dxb,dQi,dUi,Qi,Wi,Uv[nj]))
                
        ui = 3*ti/tc
        pu  = potential_derivate(xi,ui)                 
        ti += dt
        dxp = - cd*pu *dt              # potential part
        dxb = cb*np.random.randn() *dt # thermal part
        dxi = dxp+dxb
        xi += dxi            
        dQi = potential(xi,ui-du) - potential(xi-dxi,ui-du)
        Qi += dQi
        dUi = potential(xi,ui) - potential(xi-dxi,ui-du)
        dWi = dUi-dQi
        Wi += dWi

        if loc_disp and ni%10000 == 0: 
            print ('%8.3f  %8.3f  %8.3f  %8.3f  %8.3f  %8.3f  %8.3f  %8.3f  %8.3f' 
                   % (ti,xi,dxp,dxb,dQi,dUi,Qi,Wi,potential(xi,ti)))

    if loc_disp: 
        print ('%8.1f  %8.3f  %8.4f  %8.4f  %8.3f  %8.3f  %8.3f  %8.3f  %8.3f' 
               % (ti,xi,dxp,dxb,dQi,dUi,Qi,Wi,Uv[nj]))
                        
    return Uv,Qv,Wv,Pv


def batch_of_dynamic_simulations():
    """
    returns average of U, Q, W, P at every second 
    for ns simulations
    """
          
    def pre_start_phase(nt,x0): # initial phi for nt*dt starting from x0
        xi = x0
        ti = 0
        for ni in range(nt):                         
            pu  = potential_derivate(xi,0)                 
            ti += dt
            dxp = - cd*pu *dt              # potential part
            dxb = cb*np.random.randn() *dt # thermal part
            dxi = dxp+dxb
            xi += dxi            
        return xi
        
    print ('\nComputing dynamic_evolution for ',ns,' simulations')

    # du  = 3*dt/tc      # u increment (u = 0 to 3) 
    
    tv  = np.arange(ne+1)
    
    # mean values for ns simulations              
    Umv, Qmv, Wmv, Pmv = np.zeros((ne+1)),np.zeros((ne+1)),np.zeros((ne+1)),np.zeros((ne+1))

    # values at end of  phases 
    U1, Q1, W1, P1 = np.zeros((ns)),np.zeros((ns)),np.zeros((ns)),np.zeros((ns))
    U2, Q2, W2, P2 = np.zeros((ns)),np.zeros((ns)),np.zeros((ns)),np.zeros((ns))
    
    print ('\n%4s  %8s  %8s  %8s  %8s' % ('si', 'Umf','Qmf','Wmf','time')) 
    xx, yy = 0,0
    
    # MAIN COMPUTING 25 mn
    
    for si in range(ns):
        
        # starting alternatively from left or right bottom
        if si%2 == 0:   x0 = -xb
        else:           x0 =  xb 
        x0 = pre_start_phase(2000,x0) # equilibrium after 2000 steps (20 s)len(
        
        Uv,Qv,Wv,Pv = one_dynamic_simulation(tc, x0) 
        
        nii = int(ne/2)
        U1[si], Q1[si], W1[si], P1[si] = Uv[nii],Qv[nii],Wv[nii],Pv[nii]
        U2[si], Q2[si], W2[si], P2[si] = Uv[-1],Qv[-1],Wv[-1],Pv[-1]
                                       
        Umv += Uv
        Qmv += Qv
        Wmv += Wv
        Pmv += Pv
        if si%50 == 0 or si == ns-1:
            timex = time.perf_counter() - start
            print ('%4i  %8.3f  %8.3f  %8.3f  %8.1f'
                   % (si, Umv[-1]/(si+1),Qmv[-1]/(si+1), Wmv[-1]/(si+1), timex))
    
    # END OF MAIN COMPUTING

    print ('\n%6s  %8s  %8s  %8s  %8s' % ('phase', 'Um','Qm','Wm','Pm')) 
    print ('%6i  %8.3f  %8.3f  %8.3f  %8.3f' 
           % (1, np.mean(U1), np.mean(Q1), np.mean(W1), np.mean(P1)))
    print ('%6i  %8.3f  %8.3f  %8.3f  %8.3f'  
           % (2, np.mean(U2)-np.mean(U1), np.mean(Q2)-np.mean(Q1), np.mean(W2)-np.mean(W1), np.mean(P2)-np.mean(P1)))
    print ('%6s  %8.3f  %8.3f  %8.3f  %8.3f'  
           % ('total', np.mean(U2), np.mean(Q2), np.mean(W2), np.mean(P2)))
    
    # means
    Umv = Umv/ns
    Qmv = Qmv/ns
    Wmv = Wmv/ns
    Pmv = Pmv/ns

    return Umv, Qmv, Wmv, Pmv


def dynamic_evolution_of_last_phase():    
    """
    computing recovered work W for dynamic process
    for several durations of phase 3
    based on Kramers formula 
   
    second derivate of potential curve at well bottom Ub''= 2*a0, 
    at top of barrier Ut''= 2*b0
    """
       
    print ("\nDynamic evolution of phase 3 for shifting bistable memory") # 
    
    def state_energy3(u): 
        # computes energy of one state as a function of the geometric parameter u
        dx = 0.01
        Ut, Zt = 0,0                # Zt = repartition function
        nxi = int((xmax)/dx)
        for ii in range(nxi):
            xi = ii*dx
            fi = phi_restore(xi, u)
            expt = np.exp(-fi)
            Zt += expt
            Ut += fi * expt
        Ut = Ut/Zt
        return Ut 
      
    Umax = state_energy3(u=0) # beginning of phase 3
    U0   = state_energy3(u=1) # end of phase 3
    
    print()
    print ('Umax   = %8.3f' % (Umax))
    print ('U0     = %8.3f' % U0)

    # Kramers coefficient for F01
    f0     = np.sqrt(2*a0*2*b0) / (2*np.pi/cd)
    F01    = f0 * np.exp(-Umax)
    F10max = F01 * np.exp(Umax - U0)    
    print ('f0     = %8.3f' % f0)
    print ('F01    = %8.2e' % F01)
    print ('F10max = %8.3f' % F10max)

    un  = 100000   # nb of iterations
    du  = 1/un     # step value
    print ('un     = %8i     nb of iterations' % (un)) 

    # pre-computing energy of state 1 and its derivate for u between 0 and 1 
    # TAKES ABOUT 4  MN for un = 100000
    
    print ('\nPre-computing state energy and derivate')
    Usv, dUsv = np.zeros((un + 1)), np.zeros((un + 1))
    
    for ni in range(un+1):
        ui = ni/un
        Usv[ni] = state_energy3(ui)
        if ui > 0:
            dUsv[ni] = Usv[ni] - Usv[ni-1]
        if ni%(un/5) == 0: print ('ni = ', ni)

    # Computing W for several durations
    print ('\nComputing W for several durations')

    # durations to compute 
    tc_names = ['3 years','1 year','1 month','1 week','1 day','1 hour']

    th = 3600; td = 24*th; tw = 7*td; tm = 4*tw; ty = 12*tm
    
    tc_list = [3*ty,ty,tm,tw,td,th]
    
    tcn = len(tc_list)
    
    # initial state              
    Pi = 0 # 
    points_nb = 1000                # number of points for the figure
    nit_per_point = un / points_nb  # number of iterations per point
    
    Wtab = np.zeros((tcn,points_nb + 1))
    
    print ('\n%-10s  %8s  %8s' % ('time', 'P', 'W'))                 
    
    for tci in range(tcn):
        
        tc = tc_list[tci]
                                                                            
        # vectors
        Pv  = np.zeros((un+1))  #  probability of state 1    
        Wv  = np.zeros((un+1))  # work from actuator  
        
        # initial state
        ui  = 0            
        dPi = 0
        
        # condition of calculation, to prevent P < 0 at beginning of phase
        
        F10_min = 1/tc/du - F01 # no computation when F10 < F10_min
                
        # numerical integration
            
        for ni in range(1,un+1):                                        
            F10 = F01 * np.exp(Usv[ni] - U0)
            
            if F10 < F10_min: # good condition
                dPi = (F01 - (F01 + F10) * Pv[ni-1]) * tc * du
            else:             # no evolution
                dP1 = 0                    

            Pv[ni] = Pv[ni-1] + dPi
            Wv[ni] = Wv[ni-1] - Pv[ni] * dUsv[ni]
                            
            if ni%100 == 0:
                Wtab[tci,int(ni/nit_per_point)] = Wv[ni]
                                                                
        print ('%-10s  %8.3f  %8.3f' % (tc_names[tci], Pv[-1], Wv[-1]))

    return Wtab


if __name__ == "__main__":
    
    start = time.perf_counter()
    
    print()
    print ('Shifting_bistable_memory_v10',datetime.now().strftime("%d/%m/%Y at %H:%M:%S"))
   
    # parameters
    display_parameters()
    
    # plot the potential landscape
    plot_landscape_of_shifting_bistable_memory()
    
    # quasistatic evolution                             
    Uqs,Wqs,Sqs,Qqs,Pqs,Hqs = quasistatic_evolution()
    
    # dynamic evolution                                           
    Umv, Qmv, Wmv, Pmv = batch_of_dynamic_simulations()
    
    # plot results        
    plot_evolution_of_W_P_Q()
    
    # dynamic evolution of last phase from adiabatic to quasistatic (4 mn)    
    Wtab = dynamic_evolution_of_last_phase() 
    
    plot_W_during_phase3_for_different_durations(Wtab) 
    
    if False:# optional 
        # save results
        data = [Uqs,Wqs,Sqs,Qqs,Pqs,Hqs,Umv, Qmv, Wmv, Pmv, Wtab] 
        save_data(data, 'entropy_4B_data_backup.p') 
        
    if False:# optional    
        # load results    
        Umv,Qmv,Wmv,Pmv,Wf,Uqs,Wqs,Sqs,Qqs,Pqs,Hqs = load_data('entropy_4B_data_backup.p')                                 
                   
    print ("\nend of Shifting_bistable_memory_v10 after %6.3f s" % (time.perf_counter() - start),'\n')




