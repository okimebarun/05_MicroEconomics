############################################
# title: DSBalance (micro-economics simulator for demand and supply ) 
# version: 1.0
# date: 2021/12/03
# author: okimebarun
# url: https://github.com/okimebarun/
# url: https://qiita.com/oki_mebarun/
############################################

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

st.title('Demand and supply balance')

############################################
# class
class MicroModel01:
    def __init__(self):
        # supply curve
        self.S_0 = 20.0
        self.s = 0.016
        self.alpha = 0.9
        # demand curve
        self.D_0 = 100.0
        self.d = -0.025
        self.beta = 0.9
        # tax
        self.r_C = 0.1
    
    def surplus(self, P, X):
        CS = self.D_0 / self.d * (np.exp(self.d * X) - 1) - P * X
        PS = P * X - self.S_0 / self.s * (np.exp(self.s * X) - 1)
        TS = CS+PS
        return CS, PS, TS
    
    def demand(self, X):
        return self.D_0 * np.exp( self.d * X)
    
    def supply(self, X):
        return self.S_0 * np.exp( self.s * X)
    
    def invsupply(self, P):
        return np.log(P/self.S_0)/self.s

############################################
# draw functions

def drawcurve(f, xmin, xmax, label=None):
    N = 100
    x = [(xmax - xmin)*i/(N-1) + xmin for i in range(N)]
    y = [f(v) for v in x]
    plt.plot(x,y,label=label)

def drawtrend(m, hn, hx, hp, hcs, hps, hts):
    fig1 = plt.figure(figsize=(6,7))
    plt.subplot(211)
    plt.plot(hn, hcs, label='CS')
    plt.plot(hn, hps, label='PS')
    plt.plot(hn, hts, label='TS')
    plt.xlabel("time")
    plt.legend()
    plt.subplot(212)
    plt.plot(hn ,hx, label='X')
    plt.plot(hn ,hp, label='P')
    plt.xlabel("time")
    plt.legend()
    st.pyplot(fig1)

def drawhis(m, hn, hx, hp):
    fig = plt.figure()
    drawcurve(m.demand, 0, 100)
    drawcurve(m.supply, 0, 100)
    plt.plot(hx ,hp, label='sim')
    plt.xlabel("amount")
    plt.ylabel("price")
    plt.legend()
    st.pyplot(fig)

############################################
# sim functions

def calcsim(m, N, X0, P0):
    #
    hn = [0]
    hx = [X0]
    hp = [P0]
    x = X0
    p = P0
    #
    cs, ps, ts = m.surplus(p,x)
    hcs = [cs]
    hps = [ps]
    hts = [ts]
    #
    for n in range(1,N):
        # 1. supply
        xn = m.alpha * x + (1- m.alpha) * m.invsupply( p / (1 + m.r_C))
        # 2. demand
        pn = m.beta * p + (1 - m.beta) * m.demand( x )
        #
        x = xn
        p = pn
        cs, ps, ts = m.surplus(p,x)
        #
        hn.append(n)
        hx.append(x)
        hp.append(p)
        hcs.append(cs)
        hps.append(ps)
        hts.append(ts)
    return hn, hx, hp, hcs, hps, hts 

############################################
# layout

opt_X0 = st.sidebar.slider('X0 : start point of amount',0,100,50,key='opt1')
opt_P0 = st.sidebar.slider('P0 : start point of price',0,100,50,key='opt2')
opt_r_C = st.sidebar.slider('r_C: comsumption tax rate',0.0,1.0,0.1,key='opt3')
opt_S0 = st.sidebar.slider('S0 : intecept of supply curve',0.0,100.0,20.0,key='opt2')
opt_s = st.sidebar.slider('s : ascending rate of supply curve',0.0,0.1,0.016,key='opt2')
opt_D0 = st.sidebar.slider('D0 : intercept of demand curve',0.0,100.0,100.0,key='opt2')
opt_d = st.sidebar.slider('d : descending rate of demand curve',-0.1,0.0,-0.025,key='opt2')

btn_def = st.sidebar.button('Default')
if btn_def:
    opt_X0 = 50
    opt_P0 = 50
    opt_r_C = 0.1
    opt_S0 = 20.0
    opt_s = 0.016
    opt_D0 = 100.0
    opt_d = -0.025

'The tuple (X0, P0, r_C) is ',f"( {opt_X0}, {opt_P0}, {opt_r_C} )", '.'
'The tuple (S0, s, D0, d) is ',f"( {opt_S0}, {opt_s}, {opt_D0}, {opt_d} )", '.'

############################################
# main
# model
m1 = MicroModel01()
m1.r_C = opt_r_C
m1.S_0 = opt_S0
m1.s = opt_s
m1.D_0 = opt_D0
m1.d = opt_d
#
X0, P0 = opt_X0, opt_P0
hn, hx, hp, hcs, hps, hts = calcsim(m1, 100, X0, P0)
drawhis(m1, hn, hx, hp)
drawtrend(m1, hn, hx, hp, hcs, hps, hts)
#


