"""Entrena una red Keras para reconocer digitos escritos a mano."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_digits
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = PROJECT_DIR / "modelos" / "modelo_digitos.keras"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "resultados_digitos"


@dataclass
class DigitData:
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    class_names: list[str]


def load_digit_data(seed: int = 42) -> DigitData:
    """Carga las 1,797 imagenes 8x8 y crea una particion estratificada."""
    digits = load_digits()
    features = digits.images.reshape(-1, 64).astype(np.float32) / 16.0
    labels = digits.target.astype(np.int64)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=seed,
        stratify=labels,
    )
    return DigitData(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        class_names=[str(label) for label in digits.target_names],
    )


def build_digit_model() -> keras.Sequential:
    """Crea un clasificador denso de 64 pixeles de entrada y 10 clases."""
    return keras.Sequential(
        [
            keras.Input(shape=(64,), name="pixeles_8x8"),
            layers.Dense(128, activation="relu", name="oculta_1"),
            layers.Dense(64, activation="relu", name="oculta_2"),
            layers.Dense(10, activation="softmax", name="probabilidades"),
        ],
        name="reconocedor_digitos",
    )


def save_training_plots(
    history: keras.callbacks.History,
    y_test: np.ndarray,
    predictions: np.ndarray,
    output_dir: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["accuracy"], label="Entrenamiento")
    axes[0].plot(history.history["val_accuracy"], label="Validacion")
    axes[0].set(title="Exactitud", xlabel="Epoca", ylabel="Exactitud")
    axes[0].legend()
    axes[1].plot(history.history["loss"], label="Entrenamiento")
    axes[1].plot(history.history["val_loss"], label="Validacion")
    axes[1].set(title="Perdida", xlabel="Epoca", ylabel="Perdida")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_dir / "curvas_entrenamiento.png", dpi=150)
    plt.close(figure)

    matrix = confusion_matrix(y_test, predictions, labels=np.arange(10))
    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        title="Matriz de confusion: digitos escritos a mano",
        xlabel="Prediccion",
        ylabel="Valor real",
        xticks=np.arange(10),
        yticks=np.arange(10),
    )
    threshold = matrix.max() / 2
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    figure.tight_layout()
    figure.savefig(output_dir / "matriz_confusion.png", dpi=150)
    plt.close(figure)


def train(
    epochs: int = 50,
    batch_size: int = 32,
    seed: int = 42,
    model_path: Path = DEFAULT_MODEL_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> float:
    """Entrena una vez, evalua y guarda el modelo para la aplicacion interactiva."""
    keras.utils.set_random_seed(seed)
    data = load_digit_data(seed)
    print("Conjunto load_digits: 1,797 imagenes en escala de grises de 8x8.")
    print(f"Entrenamiento: {data.x_train.shape}; prueba: {data.x_test.shape}")
    print("Pixeles normalizados al rango [0, 1]; 64 entradas y 10 clases.")

    model = build_digit_model()
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
    history = model.fit(
        data.x_train,
        data.y_train,
        validation_split=0.15,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=7,
                restore_best_weights=True,
            )
        ],
        verbose=1,
    )
    test_loss, test_accuracy = model.evaluate(data.x_test, data.y_test, verbose=0)
    predictions = np.argmax(model.predict(data.x_test, verbose=0), axis=1)
    print(f"\nExactitud en datos de prueba: {test_accuracy:.2%}")
    print(f"Perdida en datos de prueba: {test_loss:.4f}")
    print(
        classification_report(
            data.y_test,
            predictions,
            labels=np.arange(10),
            target_names=data.class_names,
            zero_division=0,
        )
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    save_training_plots(history, data.y_test, predictions, output_dir)
    metadata = {
        "conjunto": "sklearn.datasets.load_digits",
        "forma_imagen": [8, 8],
        "pixeles_entrada": 64,
        "clases": data.class_names,
        "semilla": seed,
        "exactitud_prueba": float(test_accuracy),
        "perdida_prueba": float(test_loss),
        "epocas_configuradas": epochs,
        "tamano_lote": batch_size,
    }
    (model_path.parent / "metadatos_digitos.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(f"Modelo listo para reutilizar: {model_path}")
    print(f"Graficas y metadatos: {output_dir}")
    print("Para dibujar y reconocer numeros, ejecuta: python app_digitos.py")
    return float(test_accuracy)


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena el modelo de reconocimiento de digitos.")
    parser.add_argument("--epochs", type=int, default=50, help="Maximo de epocas de entrenamiento.")
    parser.add_argument("--batch-size", type=int, default=32, help="Muestras por lote.")
    parser.add_argument("--seed", type=int, default=42, help="Semilla para reproducir la particion.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs debe ser mayor que cero.")
    if args.batch_size < 1:
        parser.error("--batch-size debe ser mayor que cero.")
    train(args.epochs, args.batch_size, args.seed, args.model_path, args.output_dir)


if __name__ == "__main__":
    main()
