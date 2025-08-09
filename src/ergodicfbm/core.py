# src/ergodicfbm/core.py
from __future__ import annotations

import numpy as np
from typing import Callable, Iterable, Tuple, Optional
np.float = float

#fBm helper functions

def fBm(T, N, H):
    '''
    Generates sample paths of fractional Brownian Motion using the Davies Harte method
    
    args:
        T:      length of time (in years)
        N:      number of time steps within timeframe
        H:      Hurst parameter
    '''
    gamma = lambda k,H: 0.5*(np.abs(k-1)**(2*H) - 2*np.abs(k)**(2*H) + np.abs(k+1)**(2*H))  
    g = [gamma(k,H) for k in range(0,N)];    r = g + [0] + g[::-1][0:N-1]

    # Step 1 (eigenvalues)
    j = np.arange(0,2*N);   k = 2*N-1
    lk = np.fft.fft(r*np.exp(2*np.pi*complex(0,1)*k*j*(1/(2*N))))[::-1]

    # Step 2 (get random variables)
    Vj = np.zeros((2*N,2), dtype=np.complex128); 
    Vj[0,0] = np.random.standard_normal();  Vj[N,0] = np.random.standard_normal()
    
    for i in range(1,N):
        Vj1 = np.random.standard_normal();    Vj2 = np.random.standard_normal()
        Vj[i][0] = Vj1; Vj[i][1] = Vj2; Vj[2*N-i][0] = Vj1;    Vj[2*N-i][1] = Vj2
    
    # Step 3 (compute Z)
    wk = np.zeros(2*N, dtype=np.complex128)   
    wk[0] = np.sqrt((lk[0]/(2*N)))*Vj[0][0];          
    wk[1:N] = np.sqrt(lk[1:N]/(4*N))*((Vj[1:N].T[0]) + (complex(0,1)*Vj[1:N].T[1]))       
    wk[N] = np.sqrt((lk[0]/(2*N)))*Vj[N][0]       
    wk[N+1:2*N] = np.sqrt(lk[N+1:2*N]/(4*N))*(np.flip(Vj[1:N].T[0]) - (complex(0,1)*np.flip(Vj[1:N].T[1])))
    
    Z = np.fft.fft(wk);     fGn = Z[0:N] 
    fBm = np.cumsum(fGn)*(N**(-H))
    fBm = (T**H)*(fBm)
    path = np.array([0] + list(fBm))
    return path.real


def generate_fbm(T, N, H, MC=1):
    """
    Generate a fractional Brownian motion path of Hurst H on [0, T].
    
    Returns:
        times: np.ndarray of shape (N,)
        B:     np.ndarray of shape (N,)
    """
    times = np.linspace(0, T, N)
    B = np.empty((N+1,MC))
    for i in range(MC):
        B[:,i] = fBm(T, N, H)
    return times, B

def compute_increments(path):
    """
    Compute discrete increments ΔB[k] = path[k+1] - path[k].
    
    Returns:
        increments: np.ndarray of shape (len(path)-1,)
    """
    return np.diff(path)


#Core simulation function
#theta= lam, p, q

def drift1(theta):
    return lambda x: (theta[0])*x*((np.abs(x)+1e-8)**(theta[1]))

def drift2(theta):
    return lambda x: - x*(np.abs(x)**theta[2])

def simulate_sde_fbm(theta, H, sigma, T, N, X0, MC=1):
    """
    Simulate X_t via Euler scheme:
        dX_t = b_theta(X_t) dt + sigma dB^H_t
    
    Args:
        theta: parameter(s) for the drift
        H:     Hurst exponent of the fBm
        sigma: noise coefficient
        T:     time horizon
        N:     number of discretization points
        X0:    initial condition
        MC: Monte Carlo parameter
    
    Returns:
        times: np.ndarray of shape (N,)
        X:     np.ndarray of shape (N,)
    """
    # Drift function
    b1 = drift1(theta)
    b2 = drift2(theta) 
    
    # Generate fBm and increments
    times, B = generate_fbm(T, N, H, MC)
    dB = np.empty((N,MC))
    for i in range(MC):
        #print(B[:,i].size) (this is just for debugging)
        dB[:,i] = compute_increments(B[:,i])
    #print(dB.size)  #debugging  
    dt = times[1] - times[0]
    #print(dt)
    # Prepare solution array
    X = np.empty((N,MC))
    X[0] = X0
    for k in range(N - 1):
        if theta[1]>=0:
            X[k+1,:] = X[k,:] +  (b2(X[k,:])*dt) / ( 1+ dt*np.abs(b2(X[k,:])) ) +  sigma * dB[k,:] #taming the superlinearity
            #X[k+1,:] = X[k,:] + b1(X[k,:]) * dt + (b2(X[k,:])(1+dt*np.abs(b2(X[k,:])))) *dt+  sigma * dB[k,:] #simple euler scheme
            #print(X[k+1,:].min())
        if theta[1]<0:
            X[k+1,:] = X[k,:] + b1(X[k,:]) / (1+dt*np.abs(b1(X[k,:]))) * dt + ( b2(X[k,:]) / ( 1+ dt*np.abs(b2(X[k,:])) ) )*dt +  sigma * dB[k,:] #taming the superlinearity
    
    X_final = np.clip(X[N-1,:],-5,5)
    return times, X, X_final


#density simulation at time T
#We define: 1.`estimate_density — runs the SDE simulation to simulate the density

def sigmoid(x):
    return 1/(1+np.exp(-x))
def inv_sigmoid(x):
    return np.log(x)-np.log(1-x)

from sklearn.neighbors import KernelDensity
def estimate_density(theta, H, sigma, T, N, X0=0 ,MC=1):
    """
    Simulate one path of the SDE and estimate the density at the array x
    Args:
      theta    : parameters for the drift
      H        : Hurst exponent
      sigma    : noise coefficient
      T        : time horizon
      N        : number of points
      X0       : initial value (default 0)
      
    Returns:
      density_est  : float, the estimated f(X_T)
      Y_final : transformed X_T via sigmoid
      X_final-X0 : X_T-X_0
    """
    # 1. simulate the path
    times, X, X_final = simulate_sde_fbm(theta, H, sigma, T, N, X0, MC)
    Y_final = sigmoid(X_final) #for simplicity, i'm always taking the image of the densities by a sigmoid
    #print(X_final) #debugging
    #print(Y_final.max(),Y_final.min())
    Y_final_kde = np.asarray(Y_final)[:, None]
    
    #2. estimate the density with a gaussian kernel
    kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(Y_final_kde)
    
    sigma_prime = Y_final*(1-Y_final)
    
    density_est = np.exp(kde.score_samples(np.asarray(Y_final)[:, None]))
    
    density_est = density_est * sigma_prime
    
    density_est = np.clip(density_est, 1e-10, np.inf)
    
    return density_est, Y_final, X_final

#Target density

#Custom density function

import numpy as np

def custom_density(x, lam, p, q):
    """
    Compute the (unnormalized) density
        f(x) = exp( - (λ/p) * |x|^p  -  (1/q) * |x|^q )
    
    Args:
        x   : float or array-like
        lam : positive float, the λ parameter
        p   : positive float exponent for the first term
        q   : positive float exponent for the second term
    
    Returns:
        float or np.ndarray of same shape as x
    """
    x = np.asarray(x)
    return np.exp( (lam / p) * np.abs(x)**p
                  - (1.0  / q) * np.abs(x)**q)

def log_g(x, lam, p, q):
    """Unnormalized log-density: log g(x)"""
    x = float(x)
    return (lam / p) * np.abs(x)**p - (1.0 / q) * np.abs(x)**q

def metropolis_sampler(log_g, x0, n_samples, step=1.0, burn=1000, args=()):
    """
    Metropolis-Hastings sampler from a 1D log-density.
    
    Args:
        log_g      : function(x, *args) → log density (unnormalized)
        x0         : float, initial point
        n_samples  : number of samples to return (after burn-in)
        step       : proposal std dev (tuning parameter)
        burn       : number of burn-in steps to discard
        args       : extra arguments for log_g (e.g. lam, p, q, X0)

    Returns:
        np.ndarray of shape (n_samples,)
    """
    samples = []
    x = x0
    total = burn + n_samples

    for i in range(total):
        x_prop = np.random.normal(0, step)
        log_accept_ratio = log_g(x_prop, *args) - log_g(x, *args)
        if np.log(np.random.rand()) < log_accept_ratio:
            x = x_prop
        if i >= burn:
            samples.append(x)

    return np.array(samples)

#Objective function

def compute_objective(theta, H, sigma, T, N, lam, p, q, X0, MC):
    """
    Compute the objective
        1/n * Σ_i (density_est(x_i)-custom_density(x_i))**2 / density_est(x_i)
    where:
      - density_est(x_i) is returned by estimate_density(...)

    Args:
        theta     : parameter(s) for the drift
        H         : Hurst exponent
        sigma     : noise coefficient
        T         : time horizon
        N         : number of time-steps
        lam, p, q : parameters of the unnormalized density p(x)
        X0        : initial condition for SDE (default 0)

    Returns:
        float : the computed objective value
    """    
    # estimate_density 
    density_est, y_final, x_final = estimate_density(theta, H, sigma, T, N, X0, MC)
    
    term1 = np.array(density_est)
    #print(term1)
    # 2) Compute custom_density
    term2 = custom_density(x_final, lam, p, q)
    
    integrand = (term1 - term2)**2 / term1
    return np.log(integrand.mean() + 1e-12)


#Optimisation

def simulator(param,T,n_samples=10000):
    _,X,_ = simulate_sde_fbm(np.array([param[0],param[1],param[2]]), sigmoid(param[4])/2, 1, T, n_samples, param[3], MC=1)
    return X

def wp_1d(param, T, simulator, target_samples, n_samples=1000, p=1):
    """
    Compute the empirical 1D Wasserstein-p distance between:
      • samples = simulator(theta, n_samples)
      • target_samples (array of length >= n_samples)
    
    Parameters
    ----------
    theta : any
        Parameter passed to your simulator.
    simulator : callable
        simulator(theta,T, n_samples) -> array_like of shape (n_samples,)
    target_samples : array_like
        Samples from your target distribution (length >= n_samples).
    n_samples : int, optional (default=1000)
        Number of points to draw from the simulator.
    p : float, optional (default=1)
        Order of the Wasserstein distance (1 or 2 typically).
    
    Returns
    -------
    Wp : float
        The empirical 1D Wasserstein-p distance.
    """
    # draw from the model
    x = np.asarray(simulator(param, n_samples)).ravel()
    if x.shape[0] != n_samples:
        raise ValueError(f"simulator must return {n_samples} samples, got {x.shape[0]}")
    
    # draw (or subsample) target
    y = np.asarray(target_samples).ravel()
    if y.shape[0] < n_samples:
        raise ValueError(f"target_samples must have at least {n_samples} points")
    # randomly sub-select n_samples from target for fairness
    y = np.random.choice(y, size=n_samples, replace=False)
    
    # sort both
    x_sorted = np.sort(x)
    y_sorted = np.sort(y)
    
    # compute the mean |x_i - y_i|^p
    return np.mean(np.abs(x_sorted - y_sorted)**p)**(1/p)





