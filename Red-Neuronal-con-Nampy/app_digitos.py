"""Interfaz de escritorio para dibujar y reconocer digitos con un modelo guardado."""

from __future__ import annotations

import argparse
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np
from PIL import Image, ImageDraw
from tensorflow import keras

CANVAS_SIZE = 280
BRUSH_WIDTH = 24
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "modelos" / "modelo_digitos.keras"


def preprocess_drawing(drawing: Image.Image) -> np.ndarray:
    """Centra el trazo en una imagen 8x8 y devuelve 64 pixeles en [0, 1]."""
    grayscale = drawing.convert("L")
    bounding_box = grayscale.getbbox()
    if bounding_box is None:
        raise ValueError("Dibuja un numero antes de solicitar una prediccion.")

    digit = grayscale.crop(bounding_box)
    width, height = digit.size
    scale = 7.0 / max(width, height)
    resized_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    digit = digit.resize(resized_size, Image.Resampling.LANCZOS)

    image_8x8 = Image.new("L", (8, 8), 0)
    position = ((8 - resized_size[0]) // 2, (8 - resized_size[1]) // 2)
    image_8x8.paste(digit, position)
    pixels = np.asarray(image_8x8, dtype=np.float32) / 255.0
    return pixels.reshape(1, 64)


def predict_digit(model: keras.Model, drawing: Image.Image) -> tuple[int, np.ndarray]:
    """Predice un digito y devuelve su clase junto con las 10 probabilidades."""
    pixels = preprocess_drawing(drawing)
    probabilities = np.asarray(model(pixels, training=False))[0]
    return int(np.argmax(probabilities)), probabilities


class DigitRecognizerApp:
    def __init__(self, root: tk.Tk, model: keras.Model) -> None:
        self.root = root
        self.model = model
        self.root.title("Reconocedor de digitos escritos a mano")
        self.root.resizable(False, False)

        self.drawing = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
        self.drawer = ImageDraw.Draw(self.drawing)
        self.previous_point: tuple[int, int] | None = None

        content = ttk.Frame(root, padding=16)
        content.grid()
        ttk.Label(
            content,
            text="Dibuja un numero del 0 al 9",
            font=("Segoe UI", 16, "bold"),
        ).grid(row=0, column=0, columnspan=2, pady=(0, 12))

        self.canvas = tk.Canvas(
            content,
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            background="black",
            highlightthickness=1,
            highlightbackground="#777777",
        )
        self.canvas.grid(row=1, column=0, rowspan=3, padx=(0, 18))
        self.canvas.bind("<Button-1>", self.start_drawing)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_drawing)

        self.prediction_label = ttk.Label(content, text="Prediccion: -", font=("Segoe UI", 16, "bold"))
        self.prediction_label.grid(row=1, column=1, sticky="w")
        self.confidence_label = ttk.Label(content, text="Confianza: -")
        self.confidence_label.grid(row=2, column=1, sticky="w", pady=(2, 8))

        self.probability_bars: list[ttk.Progressbar] = []
        self.probability_labels: list[ttk.Label] = []
        probability_frame = ttk.Frame(content)
        probability_frame.grid(row=3, column=1, sticky="nsew")
        for digit in range(10):
            ttk.Label(probability_frame, text=str(digit), width=2).grid(
                row=digit, column=0, sticky="w"
            )
            bar = ttk.Progressbar(probability_frame, maximum=100, length=180)
            bar.grid(row=digit, column=1, padx=4, pady=2)
            value = ttk.Label(probability_frame, text="0.0%", width=7)
            value.grid(row=digit, column=2, sticky="e")
            self.probability_bars.append(bar)
            self.probability_labels.append(value)

        buttons = ttk.Frame(content)
        buttons.grid(row=4, column=0, columnspan=2, pady=(14, 0))
        ttk.Button(buttons, text="Predecir", command=self.predict).grid(row=0, column=0, padx=5)
        ttk.Button(buttons, text="Limpiar", command=self.clear).grid(row=0, column=1, padx=5)

    def start_drawing(self, event: tk.Event) -> None:
        point = (event.x, event.y)
        self.previous_point = point
        self.drawer.ellipse(
            self._brush_bounds(point),
            fill=255,
        )
        self.canvas.create_oval(*self._brush_bounds(point), fill="white", outline="white")

    def draw(self, event: tk.Event) -> None:
        if self.previous_point is None:
            return
        point = (event.x, event.y)
        self.drawer.line(
            (*self.previous_point, *point),
            fill=255,
            width=BRUSH_WIDTH,
        )
        self.drawer.ellipse(self._brush_bounds(point), fill=255)
        self.canvas.create_line(
            *self.previous_point,
            *point,
            fill="white",
            width=BRUSH_WIDTH,
            capstyle=tk.ROUND,
            smooth=True,
        )
        self.previous_point = point

    def stop_drawing(self, _event: tk.Event) -> None:
        self.previous_point = None

    def predict(self) -> None:
        try:
            digit, probabilities = predict_digit(self.model, self.drawing)
        except ValueError as error:
            messagebox.showinfo("Falta el dibujo", str(error), parent=self.root)
            return

        confidence = float(probabilities[digit])
        self.prediction_label.configure(text=f"Prediccion: {digit}")
        self.confidence_label.configure(text=f"Confianza: {confidence:.1%}")
        for probability, bar, label in zip(
            probabilities,
            self.probability_bars,
            self.probability_labels,
        ):
            percentage = float(probability) * 100
            bar.configure(value=percentage)
            label.configure(text=f"{percentage:.1f}%")

    def clear(self) -> None:
        self.drawing = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
        self.drawer = ImageDraw.Draw(self.drawing)
        self.previous_point = None
        self.canvas.delete("all")
        self.prediction_label.configure(text="Prediccion: -")
        self.confidence_label.configure(text="Confianza: -")
        for bar, label in zip(self.probability_bars, self.probability_labels):
            bar.configure(value=0)
            label.configure(text="0.0%")

    @staticmethod
    def _brush_bounds(point: tuple[int, int]) -> tuple[int, int, int, int]:
        radius = BRUSH_WIDTH // 2
        return (
            point[0] - radius,
            point[1] - radius,
            point[0] + radius,
            point[1] + radius,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Dibuja y reconoce digitos del 0 al 9.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Ruta del modelo Keras ya entrenado.",
    )
    args = parser.parse_args()
    if not args.model_path.is_file():
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "Falta el modelo entrenado",
            "Primero entrena y guarda el modelo con:\n"
            "python entrenar_digitos.py",
            parent=root,
        )
        root.destroy()
        return

    model = keras.models.load_model(args.model_path)
    if model.input_shape[-1] != 64 or model.output_shape[-1] != 10:
        raise ValueError("El modelo debe aceptar 64 pixeles y producir 10 clases.")
    root = tk.Tk()
    DigitRecognizerApp(root, model)
    root.mainloop()


if __name__ == "__main__":
    main()
