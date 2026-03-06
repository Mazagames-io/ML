from __future__ import annotations

import numpy as np


class MonteCarloSimulator:
    @staticmethod
    def simulate_paths(last_price: float, annual_return: float, annual_volatility: float, horizon: int = 30, n_paths: int = 1000) -> np.ndarray:
        dt = 1 / 252
        drift = (annual_return - 0.5 * annual_volatility**2) * dt
        diffusion = annual_volatility * np.sqrt(dt)
        shocks = np.random.normal(drift, diffusion, size=(n_paths, horizon))
        return last_price * np.exp(np.cumsum(shocks, axis=1))
