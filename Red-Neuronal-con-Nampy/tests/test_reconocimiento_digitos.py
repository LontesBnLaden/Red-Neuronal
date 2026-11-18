import unittest

import numpy as np
from PIL import Image, ImageDraw

from app_digitos import predict_digit, preprocess_drawing
from entrenar_digitos import build_digit_model, load_digit_data


class DigitRecognizerTests(unittest.TestCase):
    def test_digit_data_has_64_normalized_features_and_ten_classes(self) -> None:
        data = load_digit_data(seed=42)
        self.assertEqual((len(data.x_train), len(data.x_test)), (1437, 360))
        self.assertEqual(data.x_train.shape[1], 64)
        self.assertEqual(set(data.class_names), set(str(digit) for digit in range(10)))
        self.assertGreaterEqual(float(data.x_train.min()), 0.0)
        self.assertLessEqual(float(data.x_train.max()), 1.0)

    def test_drawing_preprocessing_produces_a_nonempty_64_pixel_input(self) -> None:
        drawing = Image.new("L", (280, 280), 0)
        ImageDraw.Draw(drawing).line((130, 40, 130, 240), fill=255, width=24)

        pixels = preprocess_drawing(drawing)

        self.assertEqual(pixels.shape, (1, 64))
        self.assertGreater(float(pixels.max()), 0.0)
        self.assertLessEqual(float(pixels.max()), 1.0)

    def test_blank_drawing_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            preprocess_drawing(Image.new("L", (280, 280), 0))

    def test_predict_digit_returns_class_and_ten_probabilities(self) -> None:
        drawing = Image.new("L", (280, 280), 0)
        ImageDraw.Draw(drawing).line((130, 40, 130, 240), fill=255, width=24)
        model = build_digit_model()

        digit, probabilities = predict_digit(model, drawing)

        self.assertIn(digit, range(10))
        self.assertEqual(probabilities.shape, (10,))
        np.testing.assert_allclose(probabilities.sum(), 1.0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
