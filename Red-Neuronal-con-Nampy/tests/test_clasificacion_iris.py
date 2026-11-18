import unittest

import numpy as np

from red_neuronal import (
    build_functional_model,
    build_sequential_model,
    load_iris_data,
)


class IrisClassificationTests(unittest.TestCase):
    def test_split_is_balanced_and_scaler_uses_training_data(self) -> None:
        data = load_iris_data(seed=42)
        self.assertEqual((len(data.x_train), len(data.x_validation), len(data.x_test)), (90, 30, 30))
        self.assertEqual(data.y_train.shape, (90, 3))
        np.testing.assert_allclose(data.x_train.mean(axis=0), np.zeros(4), atol=1e-6)
        np.testing.assert_allclose(data.y_test.sum(axis=1), np.ones(30))

    def test_both_keras_apis_return_three_class_probabilities(self) -> None:
        features = np.zeros((2, 4), dtype=np.float32)
        for model in (build_sequential_model(), build_functional_model()):
            probabilities = model(features, training=False).numpy()
            self.assertEqual(probabilities.shape, (2, 3))
            np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(2), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
