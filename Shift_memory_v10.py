
# -*- coding: utf-8 -*-
"""
Shift_memory_v10      10/02/2026

Computation code for quasistatic and dynamic evolution of a shift memory

Units : energy in kBT, entropy in kB, others MKSA 

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
import sys
import os
import pickle

# ==========================
# Figure parameters
# =========================
fig_size      = 2    
fig_fontsize  = 8
fig_linewidth = 0.8 
mpl.rcParams.update({"figure.dpi": 100, "savefig.dpi": 300,}) 

# ==========================
# Utility functions
# ==========================

def display_parameters():
    print ('\nLandscape parameters')
    print ('fim  = %5.1f  plateau potential' % fim)
    print ('xa   = %5.3f  half width of parabola' % xa)
    print ('a0   = %5.3f  coef parabola' % a0) 
    print ('xmax = %5.1f  limits of computing' % xmax)                   

    print ('\nProcess parameters')
    print ('tc    = %6i  cycle time (s)'% tc)
    print ('ns    = %6i  number of simulations'% ns)    
    print ('dt    = %6.2f  time increment (s)'% dt)
    print ('cd    = %6.1f  diffusion coefficient'% cd)
    print ('cb    = %6.1f  brownian movement coefficient' % cb)

def plot_potential_landscape():
    xh = 5
    print ('\nfunction potential(x,u)')
    fig,ax = plt.subplots(figsize=(3*fig_size,1.5*fig_size))
    xdata = np.arange(-xh,xh,0.01)            
    colors = ['b-','b:','b--','r--','r-']
    u_values = [0, 0.3, 1]
    for ii in range(3):
        uu = u_values[ii]
        ax.plot(xdata,[potential(x,u=uu) for x in xdata],colors[ii],
                linewidth=fig_linewidth, label='u='+str(uu)) 
    plt.xticks(fontsize=fig_fontsize)
    plt.yticks(fontsize=fig_fontsize)
    plt.xlim(-xh-1, xh+1) 
    plt.ylim(-1,35)  
    ax.legend(bbox_to_anchor=(1,0.6),fontsize=fig_fontsize)
    plt.grid() 
    plt.show()

def plot_potential_derivate():
    print ('\nfunction potential_derivate(x,u)')
    fig,ax = plt.subplots(figsize=(3*fig_size,1.5*fig_size))
    xh = 5
    xdata = np.arange(-xh,xh,0.01)            
    colors = ['b-','b:','b--','r--','r-']
    u_values = [0, 0.5, 1]
    for ii in range(3):
        uu = u_values[ii]
        ax.plot(xdata,[potential_derivate(x,u=uu) for x in xdata],
                colors[ii],linewidth=fig_linewidth, label='u='+str(uu)) 
    plt.xticks(fontsize=fig_fontsize)
    plt.yticks(fontsize=fig_fontsize)
    plt.xlim(-xh-1, xh+1) 
    ax.legend(bbox_to_anchor=(1,0.6),fontsize=fig_fontsize)
    plt.grid() 
    plt.show()
    
def plot_shift_memory():    
    print ('\nFigure shift memory')
    fig,ax = plt.subplots(figsize=(4*fig_size,1.*fig_size))
    # to remove edges and grid
    fig.patch.set_visible(False)
    ax.axis('off')
    xh = 5
    dx = 12    
    xdata = np.arange(-xh,xh,0.01)                    
    loc_lw = 1
    ax.plot(xdata,[potential(x,u=0)   for x in xdata],'b-',linewidth=loc_lw,label='u = 0')            
    ax.plot(xdata+dx,[potential(x,u=0.3) for x in xdata],'b-',linewidth=loc_lw,label='u = 0.3')            
    ax.plot(xdata+2*dx,[potential(x,u=0.8) for x in xdata],'b',linewidth=loc_lw,label='u = 0.8')            
    ax.plot(xdata+3*dx,[potential(x,u=1)   for x in xdata],'b',linewidth=loc_lw,label='u = 1')
    for ii in range(4):
        ax.text((dx+1.5)*ii-3, -5,'u = '+str([0,0.3,0.8,1][ii]), fontsize=12)
    plt.xlim(-xmax, xmax+3*dx)
    plt.ylim(-5,30)    
    plt.xticks(fontsize=fig_fontsize)
    plt.yticks(fontsize=fig_fontsize)
    plt.xlabel('x',fontsize=fig_fontsize)
    ax.grid()
    plt.show()

def plot_quasistatic_evolution_of_P_U():
    print ('\nFigure evolution of P, U')
    lw = 1.2
    fig_fontsize = 12
    fig,ax = plt.subplots(figsize=(3*fig_size,1.5*fig_size))
    tdata = np.arange(tc+1)
    ax.plot(tdata,Pqs,'g-', linewidth=lw, label='   P')    
    ax.plot(tdata,Uqs,'b-',  linewidth=lw, label='   U')
    for ii in range(3):
        plt.vlines(int(tc/2)*ii,0,1,color='k', linestyle='--', lw=0.8)
    plt.ylim(-0.1,1.1)
    plt.ylabel("kB T",fontsize=fig_fontsize) 
    plt.yticks(fontsize=fig_fontsize)
    ax.legend(bbox_to_anchor=(1.17,0.6),fontsize=fig_fontsize)
    plt.grid() 
    plt.show()

def plot_quasistatic_evolution_of_S_H():
    print ('\nFigure quasistatic evolution of S,H')
    fig,ax = plt.subplots(figsize=(3*fig_size,1.5*fig_size))
    tdata = np.arange(tc+1)
    lw = fig_linewidth
    ax.hlines(np.log(2),0,tc+1,'k',linestyle='--',linewidth=lw,label='log2')
    ax.plot(tdata,Sqs,'b-', linewidth=lw, label=' Sqs')
    ax.plot(tdata,Hqs,'r-',linewidth=lw, label=' Hqs')            
    for ii in range(3):
        plt.vlines(int(tc/2)*ii,0,0.8,color='k', linestyle='--', lw=0.8)
    plt.ylim(-0.05,0.8)
    ax.legend(bbox_to_anchor=(1.17,0.6),fontsize=fig_fontsize)
    plt.grid() 
    plt.show()            

def plot_work_for_ns_simulations(Wf):
    print ('\nFigure W = f(si)')
    fig,ax = plt.subplots(figsize=(3*fig_size,1.5*fig_size))
    ns0 = 100 # to avoid large variations of first values
    ax.plot(np.arange(ns0,ns),  [np.sum(Wf[:si])/si for si in range(ns0,ns)],
            'b-',linewidth=1,label='Wf')
        
    plt.xticks(fontsize=fig_fontsize)
    plt.yticks(fontsize=fig_fontsize)
    plt.xlabel('number of runs for tc = '+str(tc),fontsize=fig_fontsize)
    plt.ylabel('kB T',fontsize=fig_fontsize)
    ax.legend(bbox_to_anchor=(1.0,0.6),fontsize=fig_fontsize)
    ax.grid()
    plt.show() 
    
def plot_evolution_of_W_and_Q():
    print('\nGlobal figure of dynamic vs quasistatic evolution')    
    print('\nfinal W  = %6.3f' % (Wmv[-1]))
        
    fig_fontsize = 12  
    fig_ls       = 1
    fig_ls2      = 1.3
    tdata        = np.arange(tc+1)            
    Hmv = [-Pmv[ti]*np.log(max(1e-12,Pmv[ti])) - (1-Pmv[ti])*np.log(max(1e-12, 1-Pmv[ti])) for ti in tdata]
    
    fig,ax = plt.subplots(figsize=(3*fig_size,1.5*fig_size))
    ax.plot(tdata,Wqs,'g--',linewidth=fig_ls,label='Wqs') 
    ax.plot(tdata,Wmv,'g-',linewidth=fig_ls2,markersize=1,label='Wsim')                                
    ax.plot(tdata,Pqs,'b--',linewidth=fig_ls2,label='Pqs')
    ax.plot(tdata,Pmv,'b-',linewidth=fig_ls2,markersize=1,label='Psim')
    ax.plot(tdata,Qqs,'r--',linewidth=fig_ls2,label='Qqs')
    ax.plot(tdata,Qmv,'r-',linewidth=fig_ls2,markersize=1,label='Qsim')
    plt.ylim(-0.2,1.1)
    ax.legend(bbox_to_anchor=(1.02,0.8),fontsize=fig_fontsize)                    
    ax.grid()
    plt.show() 
       
# ==============================
# Potential lansdcape parameters
# ==============================

fim  = 13         # plateau potential
xa   = 2          # half width of parabola 
a0   = fim/xa**2  # coef well parabola 

# ==============================
# Potential lansdcape functions
# ==============================

# t : time (s)
# x : abcissa (nm)

def potential(x,u): 
    """ 
    potential fonction
    from left u = 0 to right left u = 1
    """
    xp = xa*(2*u-1)                   # well bottom  
    if x < -2*xa:   y = a0*(x+xa)**2  # left parabola 
    elif x < xp-xa: y = fim           # left plateau
    elif x < xp+xa: y = a0*(x-xp)**2  # parabola
    elif x < 2*xa:  y = fim           # right plateau
    else:           y = a0*(x-xa)**2  # right parabola 
    return y 
                
def potential_derivate(x,u): 
    """ 
    potential derivate
    from left u = 0 to right left u = 1
    """
    xp = xa*(2*u-1)                   # well bottom  
    if x < -2*xa:   dy = 2*a0*(x+xa)  # left parabola 
    elif x < xp-xa: dy = 0            # left plateau
    elif x < xp+xa: dy = 2*a0*(x-xp)  # parabola
    elif x < 2*xa:  dy = 0            # right plateau
    
    else:           dy = 2*a0*(x-xa)  # right parabola 
    return dy             

# ==========================
# Physical parameters
# =========================
"""
# diffusion coefficient cd: Jun-2014 proposes 1.7 
# but direct calculation gives 2.5 for water
"""
dt  = 0.01              # period of landscape update (s) 
cd  = 2.5               # diffusion coefficient
cb = np.sqrt(2*cd/dt)   # brownian movement coefficient
           
# ==========================
# Simulation parameters
# ========================= 

ns  = 1000  # nb of simulations
            # COMPUTING TIME = about 15 mn for ns = 1000
half_ns = int(ns/2) # to process starting point from left and right
                    # for dynamic simulation           
tc      = 100   # cycle time (s)

nit = int(tc/dt)   # nb of iterations per simulation
dne = int(1/dt)    # to store results every second   
ne  = int(nit/dne) # for sampling every second
  
xmax = 4 * xa  # x limit for computing

# ==========================
# Process functions
# =========================

def state_variables(u): 
    """
    returns values of energy Ut, entropy St, Zt and proba Pt
    Zt = repartition function
    """
    dx = 0.01
    Ut, Zt = 0,0                
    nxi = int(2*xmax/dx)
    for ii in range(nxi):
        xi = -xmax + (ii+0.5)*dx
        fi = potential(xi,u)
        
        if abs(fi) < 0.01: fi = 0.01*np.sign(fi)
        if abs(fi) > 100:  fi = 100*np.sign(fi)
            
        expt = np.exp(-fi)
        Zt += expt
        Ut += fi * expt
        if ii == int(nxi/2-1): # proba state 1
            Pt = Zt
    Ut = Ut/Zt
    Pt = 1-Pt/Zt
    St = np.log(Zt)+ Ut
    return Ut,St,Zt,Pt

def compute_quasistatic_evolution():
    """
    Computing quasistatic evolution ')
    returns vectors of U,W,S,Q,P,H 
    """             
    Uqs,Wqs,Sqs,Qqs,Pqs,Hqs = np.zeros((tc+1)), np.zeros((tc+1)),np.zeros((tc+1)),np.zeros((tc+1)),np.zeros((tc+1)),np.zeros((tc+1))    

    Ui,Si,Zi,Pi = state_variables(0)

    for ti in range(tc+1):
        ui = ti/tc
        Ut, St, Zt, Pt = state_variables(ui)
        Wt = -np.log(Zt) + np.log(Zi) 
        Qt = St - Si
        St = St - Si + np.log(2)
        Ptbis = min(1-1e-9, Pt)
        Ht = -Pt*np.log(max(Pt,1e-9)) - (1-Pt)*np.log(max(1-Pt,1e-9))
        Uqs[ti],Wqs[ti],Sqs[ti],Qqs[ti],Pqs[ti],Hqs[ti] = Ut,Wt,St,Qt,Pt,Ht
    
    print ('\nmin and max values of variables')
    variables = ['probability state 1','information entropy','statistical entropy S',
                 'energy U','work W','heat Q']
    print ()                 
    for ii in range(len(variables)):
        tab =  [Pqs,Hqs,Sqs,Uqs,Wqs,Qqs][ii]
        print ('%-25s  varies from %6.3f  to  %6.3f' % (variables[ii],np.min(tab), np.max(tab)))   

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
    
    du  = dt/tc      # u increment (u = 0 to 3) 
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
                
        ui = ti/tc
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


def batch_of_dynamic_simulations(tc):
    """
    returns average of U, Q, W, P at every second 
    for ns simulations
    """
          
    def pre_start_phase(nt,x0): # phi initial pendant nt*dt à partir de x0
        xi = x0
        ti = 0
        for ni in range(nt):                         
            pu  = potential_derivate(xi,0)                 
            ti += dt
            dxp = - cd*pu *dt              # potential part
            dxb = cb*np.random.randn() *dt # thermal part
            dxi = dxp+dxb
            xi += dxi
        # to avoid rare cases when starting point is out of well
        xi = min(xi,0.01)          
        return xi
        
    print ('\nComputing dynamic_evolution for ',ns,' simulations')
                                                          
    # preparing vectors of mean values for ns 
    tv  = np.arange(ne+1)                         
    Umv, Qmv, Wmv, Pmv = np.zeros((ne+1)),np.zeros((ne+1)),np.zeros((ne+1)),np.zeros((ne+1))

    # final value of Wm for every si
    Wf = np.zeros((ns))
    
    print ('\ntc = ',tc)
    print ('\n%4s  %8s  %8s  %8s  %8s' % ('si', 'Umf','Qmf','Wmf','time')) 
        
    # MAIN COMPUTING 2 mn
    
    for si in range(ns):
        
        # initial position at well bottom
        x0 = pre_start_phase(2000,-xa)
        
        Uv,Qv,Wv,Pv = one_dynamic_simulation(tc, x0) 
                                               
        Umv += Uv
        Qmv += Qv
        Wmv += Wv
        Pmv += Pv
        if si%100 == 0 or si == ns-1:
            timex = time.perf_counter() - start
            print ('%4i  %8.3f  %8.3f  %8.3f  %8.1f'
                   % (si, Umv[-1]/(si+1),Qmv[-1]/(si+1), Wmv[-1]/(si+1), timex))
        Wf[si] = Wv[-1]

    # END OF MAIN COMPUTING
    
    # means
    Umv = Umv/ns
    Qmv = Qmv/ns
    Wmv = Wmv/ns
    Pmv = Pmv/ns

    return Umv, Qmv, Wmv, Pmv,Wf


if __name__ == "__main__":
    
    start = time.perf_counter()
    
    print()
    print ('Shift_memory_v10',datetime.now().strftime("%d/%m/%Y at %H:%M:%S"))
    
    # parameters
    display_parameters()
    
    # plot the potential landscape
    plot_potential_landscape()
    plot_potential_derivate()
    plot_shift_memory()    
       
    # quasistatic evolution                             
    Uqs,Wqs,Sqs,Qqs,Pqs,Hqs = compute_quasistatic_evolution()       
    plot_quasistatic_evolution_of_P_U()         
    plot_quasistatic_evolution_of_S_H()    
                
    # dynamic evolution                                           
    Umv, Qmv, Wmv, Pmv, Wf = batch_of_dynamic_simulations(tc)
    plot_work_for_ns_simulations(Wf)

    # plot results        
    plot_evolution_of_W_and_Q()
    
    # dynamic simulation for several tc's
    print ("\ndynamic simulation for several tc's")
    
    tc_list = [10*ii for ii in range(1,11)]
    tc_nb = len(tc_list)        
    W_list = []
    
    for tc in tc_list:
        Umv, Qmv, Wmv, Pmv, Wf = batch_of_dynamic_simulations(tc)            
        W_list.append(Wf)
        
        Wf_mean   = np.sum(Wf)/ns
        print ('\n%6s  %6s' % ('tc', 'Wf'))            
        print ('%6i  %6.3f' % (tc, Wf_mean))
        
    print ('\n%6s  %6s ' % ('tc', 'W_mean'))
    for tci in range(tc_nb):
        tc = tc_list[tci]
        Wf = W_list[tci]
        Wf_mean   = np.sum(Wf)/ns       
        print ('%6i  %6.3f' % (tc, Wf_mean))
            
    print ("\nend of Shift_memory_v10 after %6.3f s" % (time.perf_counter() - start),'\n')




