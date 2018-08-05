## minimizer in Python
import scipy.optimize as spo
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as spo
import util as util
from portfolio import compute_portfolio_stats

def error(line, data): # error function
    """Compute error between given line model and observed data
    
    Parameters
    -----------
    line: tuple/list/array (C0,C1)
    data: 2D array where each row is a point (x, y)

    Returns error as a single real value
    
    """
    # sum of squared error
    err = np.sum((data[:,1]-(line[0] * data[:,0] + line[1]))** 2)
    return err

def error_poly(C, data):
    """Compute error given polynomial and observed data.
    
    Parameters
    ----------
    C: numpy.poly1d object or equivalent array representing polynomial coefficients
    data: 2D array where each row is a point (x, y)

    Returns error as a single real value
    """
    err = np.sum((data[:,1]-np.polyval(C, data[:,0])) ** 2)
    return err

def fit_line(data, error_func):
    """Fit a line to given data, using a supplied error function

    Parameters
    ----------
    data: 2D array where each row is a point
    error_func: function that computes the error between a line and observed data

    Returns line that minimizes the error function
    """
    l = np.float32([0, np.mean(data[:,1])]) # slope = 0, intercept = mean(y values)
    x_ends = np.float32([-5,5])
    plt.plot(x_ends, l[0] * x_ends + l[1], 'm--', linewidth = 2.0, label = 'Inital guess')
    result = spo.minimize(error_func, l, args=(data,), method='SLSQP', options={'disp': True})
    return result.x

def fit_poly(data, error_func, degree=3):
    """Fit a polynomial to given data, using supplied error function.
    """
    # generate initial guess for polynomial model (all coeffs = 1)
    Cguess = np.poly1d(np.ones(degree + 1, dtype=np.float32))

    # Plot initial guess (optional)
    x = np.linspace(-5, 5, 21)
    plt.plot(x, np.polyval(Cguess, x), 'm--', linewidth=2.0, label="Initial guess")

    # Call optimizer to minimize error function
    result = spo.minimize(error_func, Cguess, args=(data,), method='SLSQP', options={'disp': True})
    return np.poly1d(result.x) # convert optimal result into poly1d object

def f(X):
    """Given a scalar X, return some values (a real number)"""
    Y = (X-1.5)**2 + 0.5
    print('X = {}, Y={}'.format(X,Y))
    return Y

    Xguess = 2.0
    min_result = spo.minimize(f, Xguess, method='SLSQP', options={'disp': True})
    print("minima found")
    print('X = {}, Y={}'.format(min_result.x, min_result.fun))

def test_run():
    # Define original line
    l_orig = np.float32([4,2])
    print("Original line: C0 = {}, C1 = {}".format(l_orig[0], l_orig[1]))
    Xorig = np.linspace(0, 10, 21)
    Yorig = l_orig[0] * Xorig + l_orig[1]
    plt.plot(Xorig, Yorig, 'b--', linewidth=2.0, label="Original line")

  
    # Generate noisy data points
    noise_sigma = 3.0
    noise = np.random.normal(0, noise_sigma, Yorig.shape)
    data = np.asarray([Xorig, Yorig + noise]).T
    plt.plot(data[:,0], data[:,1], 'go', label="Data points")

    l_fit = fit_line(data, error)
    print("Fitted line: C0 = {}, C1 = {}".format(l_fit[0], l_fit[1]))
    plt.plot(data[:,0], l_fit[0] * data[:,0] + l_fit[1], 'r--', linewidth=2.0, label="Fitted line")
    plt.legend()
    plt.show()



def test_poly():
    """Not fully implemented

    Need to generate a random polygon to determine how well the function
    performs

    """

    # # Define original line
    # l_orig = np.float32([4,2])
    # print("Original line: C0 = {}, C1 = {}".format(l_orig[0], l_orig[1]))
    # Xorig = np.linspace(0, 10, 21)
    # Yorig = l_orig[0] * Xorig + l_orig[1]
    # plt.plot(Xorig, Yorig, 'b--', linewidth=2.0, label="Original line")

    # def arbitrary_poly(x, *params):
    #     return sum([p*(x**i) for i, p in enumerate(params)])

    
    # Generate noisy data points
    noise_sigma = 3.0
    noise = np.random.normal(0, noise_sigma, Yorig.shape)
    data = np.asarray([Xorig, Yorig + noise]).T
    plt.plot(data[:,0], data[:,1], 'go', label="Data points")

    l_fit = fit_line(data, error)
    print("Fitted line: C0 = {}, C1 = {}".format(l_fit[0], l_fit[1]))
    plt.plot(data[:,0], l_fit[0] * data[:,0] + l_fit[1], 'r--', linewidth=2.0, label="Fitted line")
    plt.legend()
    plt.show()

def calc_error_sharpe(allocs, data):
    return 1 - compute_portfolio_stats(allocs, data)[3]

def optimize_portfolio(sd, ed, syms, gen_plot=False):
    """Find the allocation that will maximize sharpre ratio
    
        Example
        -------
        optimize_portfolio('2010-01-01', '2010-12-31', ['GOOG', 'SPY'])
    """

    data = util.get_data(syms, sd, ed)
    print(data.head())
    # generate an initial guess for the minimizer
    initial_guess = np.ones(len(syms))/len(syms)

    # result = [ 1.35003613, -1.29013143,  0.30216938,  1.01683222]
    
    # boundaries
    bnds = [(0.0,1.0)] * len(syms)
    constraints = ({'type': 'eq', 'fun': lambda x:  np.sum(x)-1.0},)
    # constraints = ({ 'type': 'ineq', 'fun': lambda inputs: 1 - np.sum(inputs) },{ 'type': 'ineq', 'fun': lambda x: x[0] * (x[0]-1) })

    result = spo.minimize(calc_error_sharpe, initial_guess, args=(data,), method='SLSQP', bounds = bnds, constraints=constraints, options={'disp': True})
    optimal_alloc = result.x

    print("OPTIMAL_ALLOC: {}".format(str(optimal_alloc)))
    # take the optimal allocation and recalc stats for output
    cr, adr, sddr, sr = compute_portfolio_stats(optimal_alloc, data)
    
    print("sharpe ratio: {}".format(sr))
    
    return result, cr, adr, sddr, sr
    # optimize for Sharpe Ratio - 1

    
if __name__ == '__main__':
    
    allocs, cr, adr, sddr, sr = \
    optimize_portfolio(sd=dt.datetime(2010,1,1), ed=dt.datetime(2010,12,31), \
     syms=['GOOG','AAPL','GLD','XOM'], gen_plot=True)

    #print('complete')

    