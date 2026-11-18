import tempfile
import unittest
from pathlib import Path

import numpy as np

from red_regresion_numpy import NeuralNetwork


class NeuralNetworkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.inputs = np.arange(-3, 4, dtype=np.float64).reshape(-1, 1)
        self.targets = 3 * self.inputs + 2

    def test_learns_regression_and_round_trips_saved_model(self) -> None:
        model = NeuralNetwork(input_size=1, hidden_layers=(), epochs=300, learning_rate=0.03, seed=4)
        history = model.fit(self.inputs, self.targets)
        expected = model.predict(self.inputs)
        self.assertLess(history[-1], history[0])
        np.testing.assert_allclose(expected, self.targets, atol=1e-2)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            model.save(path)
            restored = NeuralNetwork.load(path)
            np.testing.assert_allclose(restored.predict(self.inputs), expected)

    def test_rejects_mismatched_training_dimensions(self) -> None:
        model = NeuralNetwork(input_size=2)
        with self.assertRaises(ValueError):
            model.fit(self.inputs, self.targets)

    def test_predict_requires_training(self) -> None:
        model = NeuralNetwork(input_size=1)
        with self.assertRaises(RuntimeError):
            model.predict(self.inputs)


if __name__ == "__main__":
    unittest.main()
