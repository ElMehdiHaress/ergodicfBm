[![CI](https://github.com/ElMehdiHaress/ergodicfBm/actions/workflows/ci.yml/badge.svg)](https://github.com/ElMehdiHaress/ergodicfBm/actions/workflows/ci.yml)

## Simulateur d’EDS dirigées par un **mouvement brownien fractionnaire (fBm)**, avec :
- génération de trajectoires fBm et schéma d’Euler,
- estimation de la densité à l’instant \(T\) (KDE),
- cible personnalisée + échantillonnage **MCMC (Metropolis–Hastings)**,
- objectif type **Wasserstein** pour l’ajustement de paramètres,
- un script d’exemples (`examples/demo.py`).

## Installation rapide

```bash
python -m venv .venv
source .venv/bin/activate   # sous Windows: .venv\Scripts\activate
pip install -U pip
pip install -e .
import numpy as np
import matplotlib.pyplot as plt
from ergodicfbm import simulate_sde_fbm

# paramètres
theta = np.array([1.0, -1.0, 2.0])
H, sigma, T, N, X0, MC = 0.3, 1.0, 1_000.0, 1000, 1.0, 100

# simulation
times, X, X_final = simulate_sde_fbm(theta, H, sigma, T, N, X0, MC)

# quelques trajectoires
plt.figure()
for i in range(min(MC, 5)):
    plt.plot(times, X[:, i], alpha=0.6)
plt.xlabel("t"); plt.ylabel("X_t"); plt.title("fBm–driven SDE (samples)")
plt.tight_layout(); plt.show()

# distribution terminale
plt.figure()
plt.hist(X_final, bins=30, density=True, alpha=0.7)
plt.xlabel("X_T"); plt.title("Terminal distribution")
plt.tight_layout(); plt.show()
```

## API (résumé)
fBm(T, N, H): génère un chemin de fBm (méthode Davies–Harte).

generate_fbm(T, N, H, MC=1): grille de temps + MC chemins.

compute_increments(path): incréments discrets ΔB.

drift1(theta), drift2(theta): fonctions de drift (exemples fournis).

simulate_sde_fbm(theta, H, sigma, T, N, X0, MC): schéma d’Euler fBm.

sigmoid(x), inv_sigmoid(x).

estimate_density(...): KDE, renvoie (density_est, Y_final, X_final).

custom_density(x, lam, p, q), log_g(x, lam, p, q): densité cible (non normalisée) + log-densité.

metropolis_sampler(log_g, x0, n_samples, step, args=()): MH 1D générique.

simulator(param, n_samples): wrapper de simulation pour l’optim.

wp_1d(param, simulator, target_samples, n_samples=1000, p=1|2): Wasserstein 1D empirique.

## Dépendances
Requises : numpy, matplotlib, scikit-learn, cma

## License
MIT
