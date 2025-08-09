"""
examples/demo.py — mets ici tous tes exemples.
Pour exécuter sans installer le package :
    PYTHONPATH=src python examples/demo.py
"""


import numpy as np
import matplotlib.pyplot as plt
from ergodicfbm import fBm, drift1, drift2, simulate_sde_fbm, sigmoid, estimate_density, custom_density, metropolis_sampler, log_g, compute_objective, simulator, wp_1d


#Plotting helper

def plot_path(times, X, H=None, sigma=None):
    """
    Plot the simulated path X over times.
    """
    plt.figure(figsize=(8,4))
    plt.plot(times, X, label=r'$X_t$')
    plt.xlabel('Time')
    plt.ylabel('X')
    title = 'SDE Path'
    if H is not None and sigma is not None:
        title += f' (H={H}, σ={sigma})'
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Example usage

# parameters
theta = np.array([1,-1,2])  # your drift parameter(s)
H     = 0.3
sigma = 1
T     = 10
N     = 1000
X0    = 1
MC    = 100

# simulate
times, X, X_final = simulate_sde_fbm(theta, H, sigma, T, N, X0, MC)

# visualize
#plot_path(times, X, H, sigma)
print(X_final.max())
plt.figure(figsize=(6,4))
plt.hist(X_final)
plt.close()

#Example usage

theta   = np.array([3.        , -4.99999998,  2.89381672])#np.array([1,-3,1])      # your drift parameter(s)
H       = 0.5
sigma   = 1
T       = 1.0      # longer horizon to approximate stationarity
N       = 1000
X0      = 0.1
MC      = 100

density_est, Y_final, X_final = estimate_density(theta, H, sigma, T, N, X0, MC)
#print(f"Estimated density f({x}) = {density_est}")

#Example of plot
#plt.plot(x-X0, density_est, label=f'KDE (bw={0.2})')
plt.figure(figsize=(6,4))
plt.hist(Y_final, density=True, alpha=0.3, label='Histogram')
plt.xlabel('x')
plt.ylabel('Density')
plt.legend()
plt.show()
plt.close()

# scalar example
#print(custom_density(0.5, lam=2.0, p=4, q=6))

# vector example
xs = np.linspace(-2, 2, 100)
fs = custom_density(xs, lam=1, p=-2, q=2)
plt.figure(figsize=(6,4))
plt.plot(xs,fs, color='r')
plt.close()

# Parameters
lam = 1.0
p = -2
q = 2
#X0 = 0.1

# Run sampler
samples = metropolis_sampler(log_g, x0=0.0, n_samples=10000, step=0.5, args=(lam, p, q,))

# Plot the result
plt.figure(figsize=(6,4))
plt.hist(sigmoid(samples), bins=20, density=True, alpha=0.6, label='MCMC samples')
plt.hist(Y_final, density=True, alpha=0.3, label='Histogram of X_T')
plt.title("Samples from $g(x) \propto \\exp(\\lambda/p |x|^p - 1/q |x|^q)$")
plt.xlabel('x')
plt.legend()
plt.show()
plt.close()

#Example usage
T=1
N=1000
lam=1
p=-2
q=2
MC=10000
X0=0.1
H = 0.3
#train_params(lam, p, q,T, N, MC, X0, lr=1e-1,epochs=200,print_every=5)

target_samples = metropolis_sampler(log_g, x0=0.0, n_samples=10000, step=0.5, args=(lam, p, q,))
print(target_samples)

from cma import CMAEvolutionStrategy

def evaluate(param):
    K = 10
    vals = [wp_1d(param, T, simulator, target_samples, n_samples=10000, p=2) for k in range(K)]
    return float(np.mean(vals))
# initialize CMA-ES to optimize theta (1-D) between, say, -5 and +5
#init_mean = np.array([1, -2,  5, 1, 0, -0.5]) #theta,sigma,x0,H
init_mean = np.array([1, -2,  5, 0, -0.5]) #theta,sigma,x0,H

init_std  = 0.5
es = CMAEvolutionStrategy(init_mean, init_std, {'popsize': 10, 'maxiter':100})
#es.opts.set({'c_sigma': 0.2, 'damping': 0.8})

es.optimize(lambda th: evaluate(th))
#while not es.stop():
#    solutions = es.ask()
#    fitnesses = [wp_1d(param, simulator, target_samples, n_samples=1000, p=1)
#                 for param in solutions]
#    es.tell(solutions, fitnesses)
param_opt, _ = es.result.xbest, es.result.fbest
print("Optimal param:", param_opt)

#Looking at the density with the found parameters
def test_param(param):
    theta, X0 = param[:3],param[3]
    H = sigmoid(param[4]) #plug your parameters here
    density_est, Y_final, X_final = estimate_density(theta, H, sigma, T, N, X0, MC)
    #Example of plot
    #print(Y_final.min())
    plt.figure(figsize=(6,4))
    plt.hist(Y_final, bins=20, density=True, alpha=0.3, label='estimated sample')
    samples = metropolis_sampler(log_g, x0=0.0, n_samples=10000, step=0.5, args=(lam, p, q))
    plt.hist(sigmoid(samples), bins=20, density=True, alpha=0.6, label='target samples')
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.legend()
    plt.show()
    plt.close()

test_param(param_opt)


