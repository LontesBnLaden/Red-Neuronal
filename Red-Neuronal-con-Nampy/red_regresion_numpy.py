"""A small, reusable NumPy multilayer perceptron for regression."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np


class NeuralNetwork:
    """Fully connected regression network with tanh hidden layers."""

    def __init__(
        self,
        input_size: int,
        hidden_layers: Sequence[int] = (8,),
        output_size: int = 1,
        learning_rate: float = 0.03,
        epochs: int = 2500,
        batch_size: int | None = None,
        seed: int | None = 1,
    ) -> None:
        sizes = (input_size, *hidden_layers, output_size)
        if any(not isinstance(size, (int, np.integer)) or size < 1 for size in sizes):
            raise ValueError("input_size, hidden layer sizes and output_size must be positive integers.")
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be a positive finite number.")
        if not isinstance(epochs, int) or epochs < 1:
            raise ValueError("epochs must be a positive integer.")
        if batch_size is not None and (not isinstance(batch_size, int) or batch_size < 1):
            raise ValueError("batch_size must be a positive integer or None.")

        self.input_size = int(input_size)
        self.hidden_layers = tuple(int(size) for size in hidden_layers)
        self.output_size = int(output_size)
        self.learning_rate = float(learning_rate)
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self._sizes = (self.input_size, *self.hidden_layers, self.output_size)
        self._weights: list[np.ndarray] = []
        self._biases: list[np.ndarray] = []
        self._x_mean = np.empty(0)
        self._x_scale = np.empty(0)
        self._y_mean = np.empty(0)
        self._y_scale = np.empty(0)
        self._fitted = False

    def fit(self, inputs: np.ndarray, targets: np.ndarray) -> list[float]:
        """Train on 2D feature and target arrays; return normalized MSE per epoch."""
        x = np.asarray(inputs, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.input_size or x.shape[0] == 0:
            raise ValueError(f"inputs must have shape (samples, {self.input_size}) and contain at least one sample.")
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if y.ndim != 2 or y.shape != (x.shape[0], self.output_size):
            raise ValueError(f"targets must have shape (samples, {self.output_size}) matching inputs.")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("inputs and targets must contain only finite numbers.")

        self._x_mean = x.mean(axis=0)
        self._x_scale = self._safe_scale(x.std(axis=0))
        self._y_mean = y.mean(axis=0)
        self._y_scale = self._safe_scale(y.std(axis=0))
        x_scaled = (x - self._x_mean) / self._x_scale
        y_scaled = (y - self._y_mean) / self._y_scale

        rng = np.random.default_rng(self.seed)
        self._weights = [
            rng.normal(0.0, np.sqrt(2.0 / (left + right)), (left, right))
            for left, right in zip(self._sizes, self._sizes[1:])
        ]
        self._biases = [np.zeros((1, size)) for size in self._sizes[1:]]
        batch_size = self.batch_size or x.shape[0]
        history: list[float] = []

        for _ in range(self.epochs):
            order = rng.permutation(x.shape[0])
            for start in range(0, x.shape[0], batch_size):
                indices = order[start : start + batch_size]
                activations = self._forward(x_scaled[indices])
                error = activations[-1] - y_scaled[indices]
                delta = (2.0 / error.size) * error
                weight_gradients: list[np.ndarray] = []
                bias_gradients: list[np.ndarray] = []

                for layer in range(len(self._weights) - 1, -1, -1):
                    weight_gradients.append(activations[layer].T @ delta)
                    bias_gradients.append(delta.sum(axis=0, keepdims=True))
                    if layer > 0:
                        delta = (delta @ self._weights[layer].T) * (1.0 - activations[layer] ** 2)

                for layer, (weight_gradient, bias_gradient) in enumerate(
                    zip(reversed(weight_gradients), reversed(bias_gradients))
                ):
                    self._weights[layer] -= self.learning_rate * weight_gradient
                    self._biases[layer] -= self.learning_rate * bias_gradient

            predictions = self._forward(x_scaled)[-1]
            history.append(float(np.mean((predictions - y_scaled) ** 2)))

        self._fitted = True
        return history

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Return predictions as a 2D array with shape (samples, output_size)."""
        self._require_fitted()
        x = np.asarray(inputs, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.ndim != 2 or x.shape[1] != self.input_size:
            raise ValueError(f"inputs must have shape (samples, {self.input_size}).")
        if not np.isfinite(x).all():
            raise ValueError("inputs must contain only finite numbers.")
        scaled = (x - self._x_mean) / self._x_scale
        return self._forward(scaled)[-1] * self._y_scale + self._y_mean

    def save(self, path: str | Path) -> None:
        """Save the trained model to a portable NumPy .npz file."""
        self._require_fitted()
        metadata = {
            "format_version": 1,
            "input_size": self.input_size,
            "hidden_layers": self.hidden_layers,
            "output_size": self.output_size,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "seed": self.seed,
        }
        arrays: dict[str, np.ndarray] = {
            "metadata": np.asarray(json.dumps(metadata)),
            "x_mean": self._x_mean,
            "x_scale": self._x_scale,
            "y_mean": self._y_mean,
            "y_scale": self._y_scale,
        }
        arrays.update({f"weight_{i}": value for i, value in enumerate(self._weights)})
        arrays.update({f"bias_{i}": value for i, value in enumerate(self._biases)})
        with Path(path).open("wb") as file:
            np.savez_compressed(file, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> NeuralNetwork:
        """Load a model previously saved with :meth:`save`."""
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"].item()))
            if metadata.get("format_version") != 1:
                raise ValueError("Unsupported model file format.")
            model = cls(
                input_size=metadata["input_size"],
                hidden_layers=metadata["hidden_layers"],
                output_size=metadata["output_size"],
                learning_rate=metadata["learning_rate"],
                epochs=metadata["epochs"],
                batch_size=metadata["batch_size"],
                seed=metadata["seed"],
            )
            model._x_mean = data["x_mean"].copy()
            model._x_scale = data["x_scale"].copy()
            model._y_mean = data["y_mean"].copy()
            model._y_scale = data["y_scale"].copy()
            model._weights = [data[f"weight_{i}"].copy() for i in range(len(model._sizes) - 1)]
            model._biases = [data[f"bias_{i}"].copy() for i in range(len(model._sizes) - 1)]
            model._validate_saved_state()
            model._fitted = True
        return model

    @staticmethod
    def _safe_scale(scale: np.ndarray) -> np.ndarray:
        return np.where(scale < 1e-12, 1.0, scale)

    def _forward(self, inputs: np.ndarray) -> list[np.ndarray]:
        activations = [inputs]
        for index, (weights, bias) in enumerate(zip(self._weights, self._biases)):
            values = activations[-1] @ weights + bias
            activations.append(np.tanh(values) if index < len(self._weights) - 1 else values)
        return activations

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("The network has not been trained. Call fit() before predict() or save().")

    def _validate_saved_state(self) -> None:
        expected_shapes = [(left, right) for left, right in zip(self._sizes, self._sizes[1:])]
        for weight, bias, shape in zip(self._weights, self._biases, expected_shapes):
            if weight.shape != shape or bias.shape != (1, shape[1]):
                raise ValueError("The model file contains incompatible weight or bias shapes.")
        if (
            self._x_mean.shape != (self.input_size,)
            or self._x_scale.shape != (self.input_size,)
            or self._y_mean.shape != (self.output_size,)
            or self._y_scale.shape != (self.output_size,)
            or np.any(self._x_scale <= 0)
            or np.any(self._y_scale <= 0)
        ):
            raise ValueError("The model file contains incompatible normalization data.")


def run_demo(epochs: int = 2500, save_path: Path | None = None) -> None:
    """Train and run the original fruit-purchase example."""
    inputs = np.array(
        [
            [1, 2, 1],
            [2, 1, 0],
            [3, 0, 2],
            [0, 1, 3],
            [4, 2, 1],
        ],
        dtype=np.float64,
    )
    targets = np.array([10, 8, 12, 9, 15], dtype=np.float64)
    model = NeuralNetwork(input_size=3, hidden_layers=(8,), epochs=epochs, seed=1)
    history = model.fit(inputs, targets)
    print("Red neuronal entrenada (regresión de gasto en dólares)")
    print(f"Error cuadrático medio normalizado: {history[-1]:.6f}")
    print("Predicciones sobre los datos de ejemplo:")
    for prediction, expected in zip(model.predict(inputs).ravel(), targets):
        print(f"  predicho: ${prediction:.2f} | real: ${expected:.2f}")
    if save_path is not None:
        model.save(save_path)
        print(f"Modelo guardado en: {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena una red neuronal de demostración.")
    parser.add_argument("--epochs", type=int, default=2500, help="Número de épocas de entrenamiento.")
    parser.add_argument("--save-model", type=Path, help="Ruta opcional para guardar el modelo entrenado.")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs debe ser mayor que cero.")

    run_demo(epochs=args.epochs, save_path=args.save_model)


if __name__ == "__main__":
    main()
