# file: speed_convergence.py
#
# "Growth under the Shadow of Expropriation"
#
# Mark Aguiar and Manuel Amador
#
# September 2010 
#
# Code that computes the speed of convergence to the steady state
# This code assumes that  
#    beta R = 1
# and that 
#    H'(k_ss) = f''(k_ss)  / Wbar'(k_ss)
# which  implies that 
#    kappa =  (Wbar'(k_ss))^2 c''(u_ss) / (theta f''(k_ss))
#
# Python 2.6 code


from __future__ import division 
from scipy import optimize, log, exp

# --------------------------------------------------------------

tau = .6
alpha = .33
bR = 1
R = 1.2 
beta = 1 / R
r = R - 1  
d = .2

def u(c):
    return log(c)

def up(c):
    return 1/c

def f(k):
    return k ** alpha 

def fp(k): 
    return alpha * k ** (alpha - 1)

def fpp(k):
    return alpha * (alpha - 1) * k ** (alpha - 2) 

def Wbar(k):
    return ( theta * u(f(k) - (1 - tau) * fp(k) * k) 
             + beta * (delta * (theta - 1) / (1 - beta * delta) 
                      + 1 / (1 - beta)) * u(f(kaut) - (r + d) *  kaut) )

def Wbarp(k):
    return ( theta * up(f(k) - (1 - tau) * fp(k) * k) *
             (fp(k) - (1 - tau) * (fpp(k) * k + fp(k))) )

def c(u):
    return exp(u)

def cp(u):
    return exp(u)

def cpp(u):
    return exp(u)

def characteristic(x, kappa):
    return (x * (x * theta  - bR * (theta - 1 + delta)) *
            (- theta + x * beta * (theta - 1 + delta)) + 
            (x - bR) * (x * beta - 1) * (x - bR * delta) *
            (x * beta * delta - 1) * theta * kappa )

kss = ((r + d) / alpha) ** (1 / (alpha - 1))
kaut = ((r + d) / (alpha * (1 - tau))) ** (1 / (alpha - 1))

for ratio in [3, 7]:
    print('theta    delta   ratio  speed')
    print('------------------------------')
    for delta in [0, .05, .1, .15, .2]:
        theta = ratio * (1 - delta)
        uss = Wbar(kss) / ((theta - 1) / (1 - beta * delta) + 1 / (1 - beta))
        kappa = (Wbarp(kss) ** 2) * cpp(uss) / fpp(kss) / theta
        print('{0:2.2f}     {1:2.2f}    {2:2.1f}    {3:2.4f}'.format(theta, 
               delta, ratio, 1 - optimize.brentq(lambda x: characteristic(x, kappa),
                                                 bR * (1 - (1 - delta)/ theta), bR)))
    print('')
