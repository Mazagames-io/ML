from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EnsembleResult:
    predictions: dict[str, np.ndarray]
    ensemble_prediction: np.ndarray
    weights: dict[str, float]


class EnsemblePredictor:
    def combine(self, model_predictions: dict[str, np.ndarray], rmse: dict[str, float]) -> EnsembleResult:
        if not model_predictions:
            raise ValueError("No model predictions to combine")

        inv = {name: 1.0 / max(rmse.get(name, 1.0), 1e-8) for name in model_predictions}
        total = sum(inv.values())
        weights = {name: val / total for name, val in inv.items()}

        size = len(next(iter(model_predictions.values())))
        ens = np.zeros(size)
        for name, pred in model_predictions.items():
            ens += pred * weights[name]

        return EnsembleResult(model_predictions, ens, weights)
