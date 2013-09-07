# file: simulation_class.py
#
# "Growth under the Shadow of Expropriation"
#
# Mark Aguiar and Manuel Amador
#
# September 2010 
#
# Numerical analysis of growth under the threat of expropriation,
# with CRRA utility. 
#
# 
# Python 2.7 code


#  ---------------- Basic libraries -------------------------------------------

from __future__ import division  
from matplotlib import use, rc
# use('Agg') # forces matplotlib to use a Non-GUI interface
# rc('text', usetex=True)
from matplotlib import pyplot as plt
from numpy import array, delete, linspace, diff, where, \
     zeros, cos, poly1d, dot, arccos, pi, sin 
from scipy import optimize, log, exp, interpolate as ip
from time import time
from itertools import product
import sys 


# ----------------- Some Generic Classes and Functions -----------------------

def convexify(x, convex=True):
    """
    Convexifies a function and returns it:
    
    Arguments
    ----------
    x -- array
        A 2-D array containig the values of the function f(x[0]) = x[1]
    convex -- bool
        whether the function to return is convex or concave. Default is
        True.

    """
    cond = (lambda x : diff(x) <= 0) if convex else lambda x : diff(x) >= 0
    while True:
        d = diff(x)
        slopes = d[1] / d[0]
        auxmask = where(cond(slopes))
        if len(auxmask[0]) >=1:
            x = delete(x, auxmask[0] + 1, axis = 1)
        else: break
    return x


class ChebyshevPoly:
    """Class that produces a system of Chebyshev polynomials.

    Methods
    -------
    getvalues  -- evaluates the chebyshev polynomials at a point.
    getderivatives -- evaluates the derivatives of the polynomials at a point.

    """
    def __init__(self, a=-1, b=1, degree=3):
        """
        Creates the Chebyshev object

        Keywords arguments:
        a -- float
            lowerbound of the domain, default is -1
        b -- float
            upperbound of the domain, default is 1
        degree -- int
            the highest degree of the polynomial.

        """
        self.intercept = - (a + b)/(b - a)
        self.slope = 2 / (b - a)
        self.degree = degree
        self.a = a
        self.b = b
        self.X = poly1d([self.slope, self.intercept])
        self.zeros = (cos(pi * (2 * (array(range(degree+1)) + 1) - 1) /
                          (2 * (degree+1))) - self.intercept)/self.slope

    def getvalues(self, k):
        """Returns a vector of the chebyshev polynomials evaluated at k."""
        return array([cos(i * arccos(self.X(k))) for i in
                      range(self.degree+1)])

    def getderivatives(self, k):
        """
        Returns the vector of the derivatives of the chebyshev
        polynomials evaluated at k.

        """
        x = self.X(k)
        return array([ i * sin(i * arccos(x)) * self.slope / (1 - x ** 2)
                       ** (1/2) for i in range(self.degree+1)])


# ------------------------ Main Model Classes ------------------------------

class BasicModel:
    """
    Class that generates the basic model object with technology
    and preferences.

    Instance variables
    ------------------
    r -- interest rate
    be -- discount factor
    a -- capital share
    d -- depreciation rate
    sigma -- CRRA utility parameter (sigma = 1 is log)
    kstar -- optimal capital given interest rate and technology.

    Methods
    -------
    f(k) -- production function
    finverse(k) -- inverse of the production function
    fprime(k) -- derivative of the production function
    fprimeinverse(k) -- inverse of derivative of production function
    U(c) -- utility function
    Uprime(c) -- derivative of utility function
    Uprimeinverse(c) -- inverse of derivative of utility function
    C(x) -- inverse of utility function
    Cprime(x) -- derivative of the inverse of the utility function.

    """
    def __init__(self, parameters):
        """Constructs the basic model object.

        Argument
        ---------
        parameters -- dictionary
            Contains the paramaters of the basic model.
            That dictionary must inclue the following keys:
            'r', 'be', 'a', 'd', 'sigma' all of them with
            float values. 

        """
        try:
            self.r = parameters['r']
            self.be = parameters['be']
            self.a = parameters['a']
            self.d = parameters['d']
            self.sigma = parameters['sigma']
        except KeyError as k:
            print("Parameter error. Expecting " + str(k))
            raise

        # generating some extra parameters
        self.umax = -10 ** (-6) if self.sigma > 1 else None
        self.umin = 10 **(-6)  if self.sigma < 1 else None
        self.bR, self.R = self.be * (1 + self.r), (1 + self.r)
        self.kstar = optimize.fmin(lambda kkk: - self.f(kkk) +
                                   (self.r + self.d) * kkk, 0., disp=0)[0]

    def f(self, k):
        "Production function"
        return  k ** self.a 

    def finverse(self, x):
        "Inverse of production function"
        return x ** (1 / self.a)

    def fprime (self, k):
        "Derivative of production function"
        return self.a * k ** (self.a - 1)

    def fprimeinverse (self, x):
        "Inverse of derivative of production function"
        return (x / self.a) ** (1 / (self.a - 1))

    def U(self, c):
        "Utility function"
        return ((c ** (1 - self.sigma)) / (1 - self.sigma) if self.sigma != 1
                else log(c))

    def Uprime(self, c):
        "Derivative of utility function"
        return c ** (-self.sigma) if self.sigma != 1 else 1 / c

    def Uprimeinverse(self, u):
        "Inverse of derivative of utility function"
        return u ** (-1/self.sigma) if self.sigma !=1 else 1 / u

    def C(self, x):
        "Inverse of U"
        return (((1 - self.sigma) * x) ** (1 / (1 - self.sigma))
                if self.sigma != 1 else exp(x))

    def Cprime(self, x):
        "Derivative of inverse U"
        return (((1 - self.sigma) * x) ** (1 / (1 - self.sigma) - 1)
                if self.sigma != 1 else exp(x))


class MarkovHyperbolic(BasicModel):
    """
    Class that contains the  hyperbolic closed-economy model.

    Instance Variables in adition to the BasicModel ones
    ----------------------------------------------------
    theta -- Political bias:  theta U(c) + beta W
    loss -- Loss in TFP
    gpol -- Vector of coefficients for the Chebyshev polynomial that
        characterizes the optimal policy function k_t --> k_{t+1}
    gval -- Vector of coefficients for the Chebyshev polynomial that
        characterizes the value function V(k_t):
            V(k_t)  = theta U(nf(k_t) - k_{t+1}) + be W(k_{t+1})
    valuefunction -- Interpolated object with the value function written
        in term of resources V(nf(k_t))
    kgrid -- Capital grid used fo compute the interpolated valuefunction.
    cheb -- Chebyshev polynomial object used to approximate the policy
        and the value function.

    Methods
    -------
    nf(k) -- Total production function: (1 - loss) * f(k) + (1 - d) k
    nfprime(k) -- Derivative of the production function
    g(x, gam) -- Returns the dot product of gam  with the chebyshev polynomials
        evaluated at x: the approximated function.
    gprime(x, gam) -- Returns the dot product of gam with the derivative of the
        chebyshev polynomials evaluated at x: the derivative of
        the approximated function.
    do_k_plot -- plots the k_t, k_{t+1} policy function. 
    __call__(x) -- Evaluates and returns the valuefunction at x.

    """

    def __init__(self, parameters, order=40, initial_guess=0.05,
                       error_bound = 10 ** (-3)):
        """
        Constructs the Hyperbolic model object and solves for the
        optimal policy and the value function using a Chebyshev collocation
        method.

        Arguments
        ----------
        parameters -- dictionary
            Contains the paramaters of the basic model.
            The dictionary must inclue the following keys:
            'r', 'be', 'a', 'd', 'sigma', 'theta', 'loss'.
        order -- int
            Order of the polynomial to be used for the chebyshev
            approximation. Default is 40.
        initial_guess -- float
            Initial policy guess of the form:
                initial_guess * nf(k).
            Default is 0.05.
            (Convergence can be very sensitive to this)
        error_bound -- float
            The model returns an exception if the error of the
            euler equation is above this bound.
            Default is 10 ** (-3) 

        """
        BasicModel.__init__(self, parameters);
        try:
            self.theta = parameters['theta']
            self.loss = parameters['loss']
        except KeyError as k:
            print("Parameter error. Expecting " + str(k))
            raise
        print("computing the hyperbolic markov equilibrium ...")
        kmin = self.kstar * 0.01
        kmax = self.kstar * 1.1
        self.cheb = ChebyshevPoly(a=kmin, b=kmax, degree=order)
        self.c0 = self.cheb.zeros
        gam0 = optimize.fsolve(lambda gam: .05 * self.nf(self.c0) -
                               self.g(self.c0, gam), zeros(order+1)) 
        self.gpol = optimize.fsolve(self.Rfcollocation, array(gam0))
        error = max(abs(array(self.Rfcollocation(self.gpol))))
        print ("  max error at the chebyshev zero points: "
               + str(error))
        print ("  last gamma: " + str(self.gpol[-1]))
        (self.gval, t1, ier, mesg) = optimize.fsolve(lambda gg:
                                          list(self.g(self.c0, gg) -
                                               self.U(self.nf(self.c0) -
                                                      self.g(self.c0,
                                                             self.gpol)) -
                                               self.be * self.g(self.g(self.c0,
                                                                   self.gpol),
                                                                gg)), gam0,
                                                     full_output=1)
        if ier != 1 or error > error_bound:
            print('!!Mesg: ' + (mesg if ier !=1
                                else "Euler error is too large"))
            raise Exception("Error in hyperbolic model: it did not converge.")
        self.kgrid = linspace(kmin, self.kstar * 1.09, 10000)
        xgrid = self.nf(self.kgrid)
        Vaut = (self.theta * self.U(xgrid - self.g(self.kgrid, self.gpol)) +
                self.be * self.g(self.g(self.kgrid, self.gpol), self.gval))
        temp = convexify(array([xgrid, Vaut]), convex=False)
        self.valuefunction = ip.UnivariateSpline(temp[0], temp[1], k=1, s=0.0)
        print("... done")

    def do_k_plot(self, fig=None):
        """
        Plots the policy function.

        Argument
        -----------------
        fig -- int
            Figure number, default is None which creates a new figure.

        """
        if fig != None:
            plt.figure(fig)
        else:
            plt.figure()
        plt.plot(self.kgrid, self.g(self.kgrid, self.gpol))
        plt.plot(self.kgrid, self.kgrid, '--')        
        plt.xlabel(r'$k_t$')
        plt.ylabel(r'$k_{t+1}$')

    def nf(self, k):
        """Total production subject to TFP loss."""
        return (1 - self.loss) * self.f(k) + (1 - self.d) * k

    def nfprime(self, k):
        """Derivative of total production subject to TFP loss."""
        return (1 - self.loss) * self.fprime(k) + (1 - self.d) 

    def g(self, x, gamma):
        """Approximated function where gamma is the vector of Chebyshev
        coefficiens."""
        return dot(gamma, self.cheb.getvalues(x))

    def gprime(self, x, gamma):
        """Derivative of approximated function where gamma is the vector
        of Chebyshev coefficiens."""
        return dot(gamma, self.cheb.getderivatives(x))

    def Rf(self, x, gamma):
        """Euler equation error to be minimized."""
        return (self.theta * self.Uprime(self.nf(x) - self.g(x, gamma))
                - self.be * self.Uprime(self.nf(self.g(x, gamma)) -
                                      self.g(self.g(x, gamma), gamma))
                * (self.nfprime(self.g(x, gamma)) +
                   (self.theta - 1) * self.gprime(self.g(x, gamma),
                                             gamma)))

    def Rfcollocation(self, gamma):
        """Vector of R evaluated at the zeros of the degree + 1
        Chebyshev polynomial (collocation method)"""
        return list(self.Rf(self.c0, gamma))

    def __call__(self, x):
        """Returns the interpolated value function at x."""
        return self.valuefunction(x)


class ModelSimulation(BasicModel):
    """
    Class contaning the Aguiar-Amador simulation.

    Instance variables in adition to the BasicModel ones
    ----------------------------------------------------
    theta -- Political bias:  theta U(c) + beta W
    loss -- Loss of TFP in case of default. 
    kgrid -- grid for capital stock
    kaut -- autarky level of capital or lowest capital stock
    kstar -- first best level of capital
    hgrid -- outside option corresponding to kgrid
    hpgrid -- derivative of outside option corresponding to kgrid
    vmin -- minimum value to the country
    vmax -- highest value to the country
    Bfb -- debt versus value to the country in the first best. Used
        as an initial guess.
    kss -- steady state
    taxmap -- map of tau_t versus tau_{t+1}
    kmap -- map of k_t versus k_{t+1}
    debttooutput -- gov't debt to output ratio
    totaldebttootuput -- (debt + R * k )/ output
    kpol, wpol, upol -- policies for k, continuation value w and current
        utility u
    valuefunction -- interpolated instance of value function:
        B(v) = Debt ( value to country )
    policies -- summary of policies
    valuefunctiondomain -- grid domain of the value function

    Methods
    -------
    do_main_iter -- iterates the value function until convergence
    do_growth_vs_distance_ss_plot, do_tax_plot, do_k_plot, ...
        do_debtoutput_plot, do_total_debtoutput_plot, ...
        def do_debtoutput_vs_distance_ss_plot  -- several ploting functions.

    """

    def __init__(self, parameters):
        """
        Constructs the Hyperbolic model object and solves for the
        optimal policy and the value function.

        Argument
        ----------
        parameters -- dictionary with the paramaters of the basic model.
            That dictionary should inclue the following keys:
            'r', 'be', 'a', 'd', 'sigma', 'theta', 'loss', 'tau_bar',
            'outside_option' (which is a string equal to either
            'hyperbolic' or 'aguiar-amador'),  and
            'k_grid_length' which is optional and defaults to 5000
            if not provided.

        """

        BasicModel.__init__(self, parameters)
        try:
            self.parameters = parameters
            self.klength = parameters['k_grid_length'] if 'k_grid_length' \
                           in parameters.keys() else 5000
            self.theta = parameters['theta']
            self.outside_option = parameters['outside_option']
            if self.outside_option == 'hyperbolic':
                self.loss = parameters['loss']
                self.tau_bar = 0.999
            else:
                if self.outside_option == 'aguiar-amador':
                    self.tau_bar = parameters['tau_bar']
                else:
                    raise Exception("outside_option parameter should be "+
                                    "'hyperbolic' or 'aguiar-amador'.")
        except KeyError as k:
            print("Parameter error. Expecting " + str(k))
            raise
        except: 
            print("Error when initializing model.")
            raise

    def generate_extra_parameters(self):
        """
        Generates extra parameters that are necessary
        to run the iteration, including the autarky value
        function.

        It is called by method do_main_iter().
        """
        
        self.A = 1 - self.a * (1 - self.tau_bar)
        self.kmin = self.kaut = (self.fprimeinverse((self.r + self.d) /
                                              (1 - self.tau_bar))
                                 if self.tau_bar != 1 else 0)
        if self.kaut > self.kstar:
            print("Problem: kaut > kstar. Stopping.")
        self.kgrid = exp(linspace(log(self.kstar), log(self.kmin),
                                  self.klength))
        if self.outside_option != 'hyperbolic':
            self.hgrid = self.h(self.kgrid)
        else:
            self.hyper = MarkovHyperbolic(self.parameters)
            self.hgrid = self.hyper(self.f(self.kgrid)
                                       + (1 - self.d) * self.kgrid)
        self.interp_h = ip.UnivariateSpline(self.kgrid[::-1],
                                            self.hgrid[::-1], s=0.0)
        self.hpgrid = self.interp_h(self.kgrid, 1)
        self.vmin = self.hgrid[-1] / (self.theta  + (1 - self.theta) * self.be)
        self.vmax = self.hgrid[0] / (self.theta + (1 - self.theta) * self.be)
        self.Bfb = array([linspace(self.vmin, self.vmax, 10000),
                     ((self.R * (self.f(self.kstar) - (self.r + self.d) *
                                 self.kstar) / 
                       self.r - (1  /(1 - self.be)) * ((self.bR) **
                                                       (- self.be /
                                                        (1 - self.be))) *
                       exp((1 - self.be) * linspace(self.vmin, self.vmax,
                                                    10000))) 
                      if self.sigma == 1 else
                      self.R * (self.f(self.kstar) - (self.r + self.d)
                                                   * self.kstar) / self.r -
                      ((1 - self.be * (self.bR) ** ((1 - self.sigma)
                                                    / self.sigma))**
                       (self.sigma / (1 - self.sigma))) * ((1 - self.sigma) *
                                                 linspace(self.vmin,
                                                          self.vmax, 10000))
                      **(1 / (1 - self.sigma)))])

    def __str__(self):
        return ('sig:' + str(self.sigma) + ', b:' + str(self.be)[0:4]
                + ', r:' + str(self.r) + ', a:' + str(self.a) + ', d:'
                + str(self.d)  + ', th:' + str(self.theta) + ', oo:'
                +  self.outside_option[0:3] +
                (', t_bar:' + str(self.tau_bar) if
                 self.outside_option =='aguiar-amador' else
                 ', loss:' + str(self.loss)))

    def h(self, k):
        "Aguiar-amador basic autarky function."
        return(self.theta * self.U(self.A * self.f(k))
               + self.be * self.U(self.A * self.f(self.kaut))
               / (1 - self.be))

    def hprime(self, k):
        "Derivate of autarky function h(k)."
        return (self.theta * self.A * self.Uprime(self.A * self.f(k))
                * self.fprime(k))    

    def do_main_iter(self, fast_iters=5, stop_iter_distance=10**(-5),
                     max_iters=500, judd_number_iters=None):
        """
        Performs the main iteration of the value function and
        construc the optimal policies.

        Arguments
        -----------------
        fast_iters -- int
            Numbers of iterations within a Judd's iteration.
            Default is 5. 
        stop_iter_distance -- int
            Iteration stops after distance between two
            cicles lies below this value.
            Default is 10 ** (-5)
        max_iters -- int
            Iteration stops if the total number of iterations exceeds
            this number. Default is 500
        judd_number_iters -- int
            Maximum number of Judd's iterations performs.
            A value of None means judd_number_iters=max_iters.
            Default is None.

        """

        def foc(hh,  lda, uboundary, lboundary):
            "Solves the first order condition for continuation value"
            if self.umax != None and ((hh - be * lboundary) > 
                                      theta * umax):
                lboundary = (hh - theta * umax) / be
            else:
                if self.umin != None and (hh - be * uboundary < theta * umin):
                    uboundary = (hh - theta * umin) / be
            return uboundary if (Cprime((hh - be * uboundary)/theta)
                                - (theta - 1) * lda + interp_B(uboundary, 1)
                                / bR) >= 0 \
                            else (lboundary if (Cprime((hh - be * lboundary)
                                                       /theta)
                                           - (theta - 1) * lda +
                                                interp_B(lboundary , 1) /
                                           bR) <= 0 
                                  else optimize.brentq(lambda w:
                                                       Cprime((hh - be * w)
                                                              /theta)
                                                       - (theta - 1) * lda +
                                                       interp_B(w, 1) / bR,
                                                       lboundary,
                                                       uboundary))

        t1 = t = time()  
        self.generate_extra_parameters()
        
        theta = self.theta; be = self.be; bR = self.bR; R = self.R
        kstar = self.kstar; kmin = self.kmin; vmin = self.vmin; d = self.d
        Bfb = self.Bfb; f = self.f; fprime = self.fprime; C = self.C
        Cprime = self.Cprime; kgrid = self.kgrid; r = self.r;  h = self.h
        hprime = self.hprime; hgrid = self.hgrid; hpgrid = self.hpgrid
        fgrid = self.f(self.kgrid); theta = self.theta; umax = self.umax
        umin = self.umin

        if not judd_number_iters:
            judd_number_iters = max_iters   
        BbeforeJudd = self.Bfb.copy()
        vv, BB = BbeforeJudd[0], BbeforeJudd[1]
        interp_B = ip.UnivariateSpline(vv, BB, s=0.0)
        judd_counter=0   # Initialize judd accelerator counter.
        lambdakgrid = (fprime(kgrid) - (r + d)) / hpgrid
        if lambdakgrid[1] - lambdakgrid[0] > 0:
            print("Lambda(k) is increasing at k = kaut")
            print("----->>>>> possible convexity violation  <<<<<------")

        for main_iter in range(max_iters):
            wk = vv[-1]
            vmin = vv[0]
            BB = []; vv = []; uu = []; ww = []; kk = []  

            for k, fk, hk, lambdak in zip(kgrid, fgrid, hgrid, lambdakgrid):
                wk = foc(hk, lambdak, wk, vmin) 
                uk = (hk - be * wk) / theta
                vk = uk + be * wk 
                gammak = Cprime(uk) - lambdak * theta
                if gammak < 0:
                    # We reached the flat part of the value function, so we stop
                    break
                Bk = fk - (r + d) * k - C(uk) + (1 / R) * interp_B(wk)
                BB.append(Bk)
                vv.append(vk)
                ww.append(wk)
                uu.append(uk)
                kk.append(k)

            # reverse and store the policy arrays
            BB.reverse(); vv.reverse()
            policies = {'k':array(kk)[::-1], 'u':array(uu)[::-1],
                        'w':array(ww)[::-1]}
            interp_B = ip.UnivariateSpline(vv, BB, k=1, s=0.0)
            distance = max(abs(interp_B(BbeforeJudd[0]) - BbeforeJudd[1]))
            BbeforeJudd = array([vv, BB])
            print('iter:' + str(main_iter) + ' error: ' +
                str(round(distance * (1 - be), 8))
                + ' time: ' + str(round(time() - t1, 2))+ ' seconds')
            t1 = time()
            if  distance < stop_iter_distance:
                print('Value function iteration converged')
                break

            if judd_counter < judd_number_iters:
                # Judd's iterator
                static_objective = f(array(policies['k'])) - (r + d) * \
                                     array(policies['k']) \
                                     - C(array(policies['u']))
                for counter in range(fast_iters):
                    interp_B = ip.UnivariateSpline(vv,
                        static_objective + interp_B(policies['w']) / R,
                        s=0.0)
                judd_counter = judd_counter + 1
            B_temp = convexify(array([vv, interp_B(vv)]), convex=False)
            vv = B_temp[0]
            interp_B = ip.UnivariateSpline(vv, B_temp[1], k=1, s=0.0)
            
        if main_iter >= max_iters - 1:
            raise Exception("--- ERRROR --- didn't converge ----")
        print("--->  time: " +str(round(time() - t, 2) / 60) + " min")
        # storing values, policies and maps. 
        self.valuefunction = interp_B
        self.policies = policies
        self.valuefunctiondomain = vv
        self.kpol = ip.UnivariateSpline(vv, policies['k'], s=0.0)
        self.wpol = ip.UnivariateSpline(vv, policies['w'], s=0.0)
        self.upol = ip.UnivariateSpline(vv, policies['u'], s=0.0)
        v1, v2 = min(self.valuefunctiondomain), max(self.valuefunctiondomain)
        tempf = lambda x: self.kpol(self.wpol(x)) - self.kpol(x)
        self.kss = self.kpol(self.wpol(v1)) if tempf(v1) <=0 else \
                       self.kpol(self.wpol(v2)) if tempf(v2) >= 0 else \
                       self.kpol(optimize.brentq(tempf, v1, v2))
        self.taxmap = {'taut' : 1 - (self.r + self.d) /
                       self.fprime(self.policies['k']),
                       'taut+1':  1 - (self.r + self.d) /
                       self.fprime(self.kpol(self.policies['w']))}        
        self.kmap = {'kt' : self.policies['k'],
                     'kt+1': self.kpol(self.policies['w'])}
        self.debttooutput = self.valuefunction(vv)/(self.R *
                                                    self.f(self.policies['k']))
        self.NFAtooutput = ((self.valuefunction(vv) +
                                   R * self.policies['k']) /
                                  (self.R * self.f(self.policies['k'])))
        self.get_euler_errors()
        print("maximum euler error: " + str(max(abs(self.euler_error))))
        return self

    def get_euler_errors(self, with_figure=False):
        """
        Computes the Euler equation errors.
        This method should be ran after the simulation has been completed.

        Argument
        ----------
        with_figure -- bool
            If True plots the errors in a new figure.
            Default is False.

        """
        
        v = self.valuefunctiondomain
        vprime = self.wpol(self.valuefunctiondomain)
        k = self.kpol(v)
        kprime = self.kpol(vprime)
        u = self.upol(v)
        uprime = self.upol(vprime)
        self.euler_error = (self.Cprime(uprime) - self.bR * self.Cprime(u)
                            - self.theta * (self.fprime(kprime) - self.r -
                                            self.d) /
                            self.interp_h(kprime, 1) + self.bR *
                            (self.theta - 1) * (self.fprime(k) - self.r -
                                                self.d) / self.interp_h(k, 1))
        if with_figure:
            plt.figure()
            plt.plot(v, self.euler_error, '.')
            plt.title("Euler equation errors")
            plt.xlabel("v")

    def do_growth_vs_distance_ss_plot(self, lin='-', newfigure=True, labels=True, 
                                      **kwargs):
        "Plots growth rates versus distance to steady state output."
        if newfigure:
            plt.figure()
        plt.plot(log(self.f(self.kmap['kt'])/self.f(self.kss)),
                 log(self.f(self.kmap['kt+1'])/self.f(self.kmap['kt']))/5, lin,
                 **kwargs)
        if labels:
            plt.xlabel(r'$\log \left(y_t/y_{ss}\right)$')
            plt.ylabel(r'$\log \left(y_{t+5}/y_t\right) / 5$')

    def do_change_debtoutput_vs_growth_plot(self, lin='-', newfigure=True, labels=True, 
                                            **kwargs):
        "Plots growth rates versus distance to steady state output."
        if newfigure:
            plt.figure()
        plt.plot(log(self.f(self.kpol(self.wpol(self.valuefunctiondomain))) /
                     self.f(self.kpol(self.valuefunctiondomain))),
                 self.valuefunction(self.wpol(self.valuefunctiondomain)) /
                 (self.R *
                  self.f(self.kpol(self.wpol(self.valuefunctiondomain)))) -
                  self.valuefunction(self.valuefunctiondomain) /
                 (self.R * self.f(self.kpol(self.valuefunctiondomain))),
                 lin, **kwargs)
        if labels:
            plt.xlabel(r'$\log \left(y_{t+1}/y_{t}\right)$')
            plt.ylabel(r'$Debt/GDP_{t+1} - Debt/GDP_{t}$')


    def do_change_assetoutput_vs_growth_plot(self, lin='-', newfigure=True,
                                             labels=True,  **kwargs):
        "Plots growth rates versus distance to steady state output."
        if newfigure:
            plt.figure()
        plt.plot(- (self.valuefunction(self.wpol(self.valuefunctiondomain)) /
                 (# self.R *
                  self.f(self.kpol(self.wpol(self.valuefunctiondomain)))) -
                  self.valuefunction(self.valuefunctiondomain) /
                 (# self.R *
                  self.f(self.kpol(self.valuefunctiondomain)))) / 5,
                 log(self.f(self.kpol(self.wpol(self.valuefunctiondomain))) /
                     self.f(self.kpol(self.valuefunctiondomain))) / 5,
                 lin, **kwargs)
        if labels:
            plt.ylabel(r'$\log \left(y_{t+5}/y_{t}\right) / 5$')
            plt.xlabel(r'$- (Debt_{t+5}/GDP_{t+5} - Debt_t / GDP_{t} ) / 5$')

    def do_tax_plot(self, lin='-', fortyfive=True, newfigure=True, labels=True,
                    **kwargs):
        "Plots the transition mapping for the tax rates."
        if newfigure:
            plt.figure()
        plt.plot(self.taxmap['taut'], self.taxmap['taut+1'], lin, **kwargs)
        if fortyfive:
            plt.plot(self.taxmap['taut'], self.taxmap['taut'], '--')
        if labels:
            plt.xlabel(r'$\tau_t$')
            plt.ylabel(r'$\tau_{t+1}$')

    def do_k_plot(self, lin='-', fortyfive=True, newfigure=True, labels=True,
                  **kwargs):
        "Plots the transition mapping for the capital stock."        
        if newfigure:
            plt.figure()
        plt.plot(self.kmap['kt'], self.kmap['kt+1'], lin, **kwargs)
        if fortyfive:
            plt.plot(self.kmap['kt'], self.kmap['kt'], '--')
        if labels:
            plt.xlabel(r'$k_t$')
            plt.ylabel(r'$k_{t+1}$') 

    def do_debtoutput_plot(self, lin='-', newfigure=True, labels=True,
                           **kwargs):
        "Plots the debt to output ratio for different capital stocks."
        if newfigure:
            plt.figure()
        plt.plot(self.policies['k'], self.debttooutput, lin, **kwargs)
        if labels:
            plt.xlabel(r'k')
            plt.ylabel(r'debt to output ratio') 

    def do_NFAoutput_plot(self, lin='-', newfigure=True, labels=True,
                          **kwargs):
        """
        Plots the total debt (b + Rk) to output ratio for different
        capital stocks.

        """
        if newfigure:
            plt.figure()
        plt.plot(self.policies['k'], self.NFAtooutput, lin, **kwargs)
        if labels:
            plt.xlabel(r'k')
            plt.ylabel(r'total debt to output ratio') 


def do_all_plots(x, figure=None):
    """
    Does a full plot of a model instance.

    Arguments
    ---------
    x -- ModelSimulation
        A simulation of the model. 
    figure -- int 
        if None, creates a new figure; else plots in the specified
        figure number. Default is None.

    """
    if figure:
        plt.figure(figure, figsize=(14, 8))
    else:
        plt.figure(figsize=(14, 8))
    plt.suptitle(str(x))
    plt.subplot(231)
    plt.plot(x.valuefunctiondomain, x.valuefunction(x.valuefunctiondomain))
    plt.xlabel('v')
    plt.ylabel('B(v)')
    plt.subplot(232)
    x.do_tax_plot(newfigure=False)
    plt.subplot(233)
    x.do_k_plot(newfigure=False)
    plt.subplot(234)
    x.do_debtoutput_plot(newfigure=False)
    plt.subplot(235)
    x.do_NFAoutput_plot(newfigure=False)
    plt.ylim(-3, 3)
    plt.subplot(236)
    plt.plot(x.kgrid, x.hgrid)
    plt.xlabel('k')
    plt.ylabel('h(k)')


# -----------  Running some simulations ------------------------------------------

if __name__ == '__main__':
    models = []
    for theta in [1, 3, 5, 7]:
        models.append({'r' : .2,                        # interest rate
                             'be' : 1/1.2,                    # discount factor
                             'theta' : theta,                 # political bias
                             'a' : .33,                       # capital share
                             'd': .2,                         # depreciation rate
                             'sigma' : 1,                     # CRRA parameter
                             'outside_option': 'aguiar-amador',  # outside option
                             'loss': 0,                       # TFP loss
                             'tau_bar' : .6,                  # maximal tax rate
                             })
    for (beta, theta) in [(1/1.2, 1), (0.63, 1), (1/1.2, 3)]:
        models.append({'r' : .2,                             # interest rate
                             'be' : beta,                    # discount factor
                             'theta' : theta,                    # political bias
                             'a' : .33,                       # capital share
                             'd': .2,                         # depreciation rate
                             'sigma' : 1,                     # CRRA parameter
                             'outside_option': 'hyperbolic',  # outside option
                             'loss': 0,                       # TFP loss
                             'tau_bar' : .6,                  # maximal tax rate
                             })
    d = {}
    for amodel in models:
        m = ModelSimulation(amodel)
        print("\nComputing new model:\n   " + str(m))
        m.do_main_iter()
        d[str(m)] =  m

    
    print('Simulations complete, generating plots ... ')
    fig_format = 'eps'
    keys = ["sig:1, b:0.83, r:0.2, a:0.33, d:0.2, th:1, oo:agu, t_bar:0.6",
            "sig:1, b:0.83, r:0.2, a:0.33, d:0.2, th:3, oo:agu, t_bar:0.6",
            "sig:1, b:0.83, r:0.2, a:0.33, d:0.2, th:5, oo:agu, t_bar:0.6",
            "sig:1, b:0.83, r:0.2, a:0.33, d:0.2, th:7, oo:agu, t_bar:0.6"]
    lin = ["k:", "k-o", "k-s", "k--"]
    for key, l in zip(keys, lin):
        m = d[key]
        plt.figure(1, figsize=(5,5))
        m.do_growth_vs_distance_ss_plot(newfigure=False, lin=l, labels=False,
                                        markevery=300,
                                        label=r'$\theta = ' +
                                        str(m.theta) + '$',
                                        lw=1)
        plt.xlabel(r'$\log \left(y_t/y_{ss}\right)$')
        plt.ylabel(r'$\log \left(y_{t+1}/y_t\right)$ / T')
        plt.figure(2,  figsize=(5,5))
        m.do_change_assetoutput_vs_growth_plot(newfigure=False, lin=l,
                                               labels=False, markevery=300,
                                               label=r'$\theta = ' +
                                               str(m.theta) + '$',
                                               lw=1)
        plt.ylabel(r'$\log \left(y_{t+1}/y_{t}\right) / T$')
        plt.xlabel(r'$- (b_{t+1}/y_{t+1} - b_t / y_{t} ) / T$')
        plt.figure(3,  figsize=(5,5))
        plt.plot(log(m.f(m.kmap['kt'])/m.f(m.kss)),
                 (- (m.valuefunction(m.wpol(m.valuefunctiondomain)) -
                     m.valuefunction(m.valuefunctiondomain)) +
                  m.kpol(m.wpol(m.valuefunctiondomain)) - (1 - m.d) *
                  m.kpol(m.valuefunctiondomain)) / 
                 m.f(m.kpol(m.valuefunctiondomain)), 
                 l, markevery=300, label=r'$\theta = ' + str(m.theta) + '$',
                 lw=1)
        plt.xlabel(r'$\log \left(y_{t}/y_{ss}\right)$')
        plt.ylabel(r'savings rate') #  = $(- (b_{t+1} - b_t) + i_t) /  y_t$')
    plt.figure(1)
    plt.xlim([-0.4, 0])
    plt.ylim([0, 0.065])
    plt.xticks([-.3, -.2, -.1, 0])
    plt.yticks([0, 0.02, .04, .06])
    plt.legend(loc=0)
    # plt.savefig("plotIV." + fig_format, format=fig_format)
    plt.figure(2)
    plt.xlim([0, 0.02])
    plt.ylim([0, 0.02])
    plt.xticks([0, 0.005, 0.010, 0.015, 0.020])
    plt.yticks([0.005, 0.010, 0.015, 0.020], ['', '0.01', '', '0.02'])
    plt.legend(loc=0)
    # plt.savefig("plotVI."+ fig_format, format=fig_format)    
    plt.figure(3)
    plt.legend(loc=0)
    plt.xlim([-0.4, 0])
    plt.ylim([0, 0.9])
    plt.xticks([-.3, -.2, -.1, 0])
    plt.yticks([0, 0.2, .4, .6, .8])
    # plt.savefig("plotV." + fig_format, format=fig_format)


    keys = [("sig:1, b:0.83, r:0.2, a:0.33, d:0.2, th:3, oo:agu, t_bar:0.6", "AA"),
            ("sig:1, b:0.83, r:0.2, a:0.33, d:0.2, th:1, oo:hyp, loss:0", "RCK"),
            ("sig:1, b:0.63, r:0.2, a:0.33, d:0.2, th:1, oo:hyp, loss:0", "BL"),
            ("sig:1, b:0.83, r:0.2, a:0.33, d:0.2, th:3, oo:hyp, loss:0", "AA2")
    ]
    for (lin,ki, markers) in zip([ 'k-o', 'k--', 'k-s', 'k:', 'k-^'], keys,
                                 [800, 100, 100, 100, 100]):
        k = ki[0]
        lab = ki[1]
        plt.figure(4, figsize=(5,5))
        print(k)
        m = d[k]
        x1 = log(m.f(m.kmap['kt'])/m.f(m.kss))
        y1 = (1/5.0) * log(m.f(m.kmap['kt+1'])/m.f(m.kmap['kt']))
        plt.plot([x1[50 * i] for i in range(int(len(x1)/50))] + [x1[-1]] ,
                 [y1[50 * i] for i in range(int(len(y1)/50))] + [y1[-1]],
                 lin, label=lab, linewidth=1, markevery=markers / 50)
    plt.xlabel(r'$\log \left(y_t/y_{ss}\right)$')
    plt.ylabel(r'$\log \left(y_{t+1}/y_t\right) / T$')
    k = "sig:1, b:0.83, r:0.2, a:0.33, d:0.2, th:3, oo:hyp, loss:0"    
    m = d[k]
    kgrid = m.hyper.kgrid
    kpol  = m.hyper.g(m.hyper.kgrid, m.hyper.gpol)
    kss = optimize.brentq(lambda kk: m.hyper.g(kk, m.hyper.gpol) - kk, m.kstar
                         * 0.01, m.kstar)
    plt.plot(log(m.hyper.f(kgrid)/m.hyper.f(kss)),
                 (1/5.0) * log(m.hyper.f(kpol)/m.hyper.f(kgrid)), 'k-',
                 label= "HYPER", linewidth=1, markevery=300)
    plt.legend(loc=0)
    plt.ylim([-.004,.05])
    plt.xlim([-.4, .03])
    # plt.savefig('plotVII.' + fig_format, format=fig_format)
    plt.show()
