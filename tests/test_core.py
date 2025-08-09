import numpy as np
import ergodicfbm as ef

def test_fbm_shapes():
    T, N, H, MC = 1.0, 50, 0.3, 4
    times, B = ef.generate_fbm(T, N, H, MC)
    assert times.shape == (N,)
    assert B.shape == (N+1, MC)

def test_simulate_sde_shapes():
    T, N, H, MC = 1.0, 60, 0.3, 5
    theta = np.array([1.0, -1.0, 2.0])
    sigma, X0 = 1.0, 0.1
    times, X, X_final = ef.simulate_sde_fbm(theta, H, sigma, T, N, X0, MC)
    assert times.shape == (N,)
    assert X.shape == (N, MC)
    assert X_final.shape == (MC,)

def test_estimate_density_bounds():
    T, N, H, MC = 0.5, 80, 0.4, 20
    theta = np.array([1.0, -1.0, 2.0])
    sigma, X0 = 1.0, 0.0
    dens, Yf, Xf = ef.estimate_density(theta, H, sigma, T, N, X0, MC)
    assert dens.shape == (MC,)
    assert Yf.shape == (MC,)
    assert Xf.shape == (MC,)
    assert np.all(Yf > 0) and np.all(Yf < 1)   # sigmoid output

def test_custom_density_and_mcmc():
    xs = np.linspace(-1, 1, 11)
    g = ef.custom_density(xs, lam=1.0, p=2.0, q=4.0)
    assert g.shape == xs.shape
    assert np.all(g > 0)

    samples = ef.metropolis_sampler(ef.log_g, x0=0.0, n_samples=200, step=0.3, args=(1.0, 2.0, 4.0))
    assert samples.shape == (200,)


