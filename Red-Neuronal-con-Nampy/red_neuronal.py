"""Entrena y compara redes neuronales Keras con el conjunto de flores Iris."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers


@dataclass
class IrisData:
    """Datos de Iris divididos y escalados sin filtrar información de prueba."""

    x_train: np.ndarray
    x_validation: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_validation: np.ndarray
    y_test: np.ndarray
    y_test_ids: np.ndarray
    scaler: StandardScaler
    feature_names: list[str]
    class_names: list[str]


@dataclass
class ModelResult:
    """Métricas, predicciones e historial de un modelo ya evaluado."""

    name: str
    model: keras.Model
    history: keras.callbacks.History
    test_loss: float
    test_accuracy: float
    predictions: np.ndarray


def demonstrate_numpy_basics() -> None:
    """Muestra vectores, matrices y una multiplicación entrada-por-pesos."""
    vector = np.array([5.1, 3.5, 1.4, 0.2])
    weights = np.array(
        [
            [0.2, 0.4, 0.1],
            [0.3, 0.1, 0.5],
            [0.6, 0.2, 0.4],
            [0.1, 0.3, 0.2],
        ]
    )
    bias = np.array([0.1, 0.2, 0.3])
    output = vector @ weights + bias
    print("\nPASO 1: vectores y matrices de NumPy")
    print(f"Vector de 4 caracteristicas: {vector} | forma: {vector.shape}")
    print(f"Matriz de pesos: forma {weights.shape}")
    print(f"Salida (vector @ pesos + sesgo): {output} | forma: {output.shape}")


def load_iris_data(seed: int = 42) -> IrisData:
    """Divide Iris en 60/20/20 y ajusta el escalador solo con entrenamiento."""
    iris = load_iris()
    x_train, x_remaining, y_train, y_remaining = train_test_split(
        iris.data,
        iris.target,
        test_size=0.4,
        random_state=seed,
        stratify=iris.target,
    )
    x_validation, x_test, y_validation, y_test = train_test_split(
        x_remaining,
        y_remaining,
        test_size=0.5,
        random_state=seed,
        stratify=y_remaining,
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_validation = scaler.transform(x_validation).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)

    class_count = len(iris.target_names)
    return IrisData(
        x_train=x_train,
        x_validation=x_validation,
        x_test=x_test,
        y_train=keras.utils.to_categorical(y_train, num_classes=class_count),
        y_validation=keras.utils.to_categorical(y_validation, num_classes=class_count),
        y_test=keras.utils.to_categorical(y_test, num_classes=class_count),
        y_test_ids=y_test,
        scaler=scaler,
        feature_names=[str(name) for name in iris.feature_names],
        class_names=[str(name) for name in iris.target_names],
    )


def build_sequential_model(input_size: int = 4, class_count: int = 3) -> keras.Sequential:
    """Construye la red multicapa con la API Sequential."""
    return keras.Sequential(
        [
            keras.Input(shape=(input_size,), name="caracteristicas"),
            layers.Dense(16, activation="relu", name="oculta_1"),
            layers.Dense(8, activation="relu", name="oculta_2"),
            layers.Dense(class_count, activation="softmax", name="clasificador"),
        ],
        name="iris_sequential",
    )


def build_functional_model(input_size: int = 4, class_count: int = 3) -> keras.Model:
    """Construye la misma arquitectura usando la API funcional de Keras."""
    inputs = keras.Input(shape=(input_size,), name="caracteristicas")
    hidden = layers.Dense(16, activation="relu", name="oculta_1")(inputs)
    hidden = layers.Dense(8, activation="relu", name="oculta_2")(hidden)
    outputs = layers.Dense(class_count, activation="softmax", name="clasificador")(hidden)
    return keras.Model(inputs=inputs, outputs=outputs, name="iris_functional")


def train_and_evaluate(
    model: keras.Model,
    name: str,
    data: IrisData,
    epochs: int,
    batch_size: int,
    seed: int,
) -> ModelResult:
    """Entrena, evalua y muestra el reporte de un clasificador."""
    keras.utils.set_random_seed(seed)
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    print(f"\nEntrenamiento: {name}")
    model.summary()
    history = model.fit(
        data.x_train,
        data.y_train,
        validation_data=(data.x_validation, data.y_validation),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
    )
    test_loss, test_accuracy = model.evaluate(data.x_test, data.y_test, verbose=0)
    probabilities = model.predict(data.x_test, verbose=0)
    predictions = np.argmax(probabilities, axis=1)

    print(f"\nReporte de clasificacion: {name}")
    print(
        classification_report(
            data.y_test_ids,
            predictions,
            labels=np.arange(len(data.class_names)),
            target_names=data.class_names,
            zero_division=0,
        )
    )
    print(f"Perdida en prueba: {test_loss:.4f}")
    print(f"Exactitud en prueba: {test_accuracy:.2%}")
    if test_accuracy >= 0.85:
        print("Meta alcanzada: exactitud de al menos 85%.")
    else:
        print("Meta no alcanzada: revisa los datos o los parametros del entrenamiento.")

    return ModelResult(
        name=name,
        model=model,
        history=history,
        test_loss=float(test_loss),
        test_accuracy=float(test_accuracy),
        predictions=predictions,
    )


def save_confusion_matrix(result: ModelResult, data: IrisData, output_dir: Path) -> None:
    matrix = confusion_matrix(
        data.y_test_ids,
        result.predictions,
        labels=np.arange(len(data.class_names)),
    )
    figure, axis = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=data.class_names,
        yticklabels=data.class_names,
        ax=axis,
    )
    axis.set(
        title=f"Matriz de confusion: {result.name}",
        xlabel="Clase predicha",
        ylabel="Clase real",
    )
    figure.tight_layout()
    figure.savefig(output_dir / f"matriz_confusion_{result.name}.png", dpi=150)
    plt.close(figure)


def save_training_curves(results: list[ModelResult], output_dir: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    for column, result in enumerate(results):
        history = result.history.history
        axes[0, column].plot(history["accuracy"], label="Entrenamiento")
        axes[0, column].plot(history["val_accuracy"], label="Validacion")
        axes[0, column].set(
            title=f"Exactitud: {result.name}",
            xlabel="Epoca",
            ylabel="Exactitud",
        )
        axes[0, column].legend()

        axes[1, column].plot(history["loss"], label="Entrenamiento")
        axes[1, column].plot(history["val_loss"], label="Validacion")
        axes[1, column].set(
            title=f"Perdida: {result.name}",
            xlabel="Epoca",
            ylabel="Perdida",
        )
        axes[1, column].legend()

    figure.tight_layout()
    figure.savefig(output_dir / "curvas_aprendizaje.png", dpi=150)
    plt.close(figure)


def save_artifacts(
    results: list[ModelResult],
    data: IrisData,
    output_dir: Path,
    seed: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(data.scaler, output_dir / "escalador_iris.joblib")

    for result in results:
        result.model.save(output_dir / f"modelo_{result.name}.keras")
        save_confusion_matrix(result, data, output_dir)

    save_training_curves(results, output_dir)
    metadata = {
        "clases": data.class_names,
        "caracteristicas_en_orden": data.feature_names,
        "semilla": seed,
        "modelos": {
            result.name: {
                "exactitud_prueba": result.test_accuracy,
                "perdida_prueba": result.test_loss,
            }
            for result in results
        },
    }
    (output_dir / "metadatos.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def predict_flowers(
    model_path: str | Path,
    scaler_path: str | Path,
    features: np.ndarray,
    class_names: list[str],
) -> list[dict[str, Any]]:
    """Carga artefactos y predice flores nuevas con sus probabilidades por clase."""
    model = keras.models.load_model(model_path)
    scaler: StandardScaler = joblib.load(scaler_path)
    values = np.asarray(features, dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] != scaler.n_features_in_:
        raise ValueError(f"features debe tener forma (muestras, {scaler.n_features_in_}).")
    if not np.isfinite(values).all():
        raise ValueError("features debe contener solo valores numericos finitos.")
    if len(class_names) != int(model.output_shape[-1]):
        raise ValueError("class_names debe contener una etiqueta por cada salida del modelo.")

    probabilities = model.predict(scaler.transform(values), verbose=0)
    results: list[dict[str, Any]] = []
    for row in probabilities:
        class_id = int(np.argmax(row))
        results.append(
            {
                "clase": class_names[class_id],
                "probabilidad": float(row[class_id]),
                "probabilidades_por_clase": {
                    class_name: float(probability)
                    for class_name, probability in zip(class_names, row)
                },
            }
        )
    return results


def run_experiment(
    epochs: int = 150,
    batch_size: int = 16,
    seed: int = 42,
    output_dir: Path = Path("resultados"),
) -> list[ModelResult]:
    """Ejecuta los pasos de la practica y guarda modelos y graficas."""
    demonstrate_numpy_basics()
    data = load_iris_data(seed=seed)
    print("\nPASO 2: conjunto Iris y preprocesamiento")
    print(f"Caracteristicas: {data.feature_names}")
    print(f"Clases: {data.class_names}")
    print(
        f"Muestras: entrenamiento={len(data.x_train)}, "
        f"validacion={len(data.x_validation)}, prueba={len(data.x_test)}"
    )
    print(f"Forma de entrada: {data.x_train.shape}")
    print(f"Forma de etiquetas one-hot: {data.y_train.shape}")
    print("Escalador ajustado solo con los datos de entrenamiento.")

    builders = [
        ("secuencial", build_sequential_model),
        ("funcional", build_functional_model),
    ]
    results = []
    for name, builder in builders:
        keras.backend.clear_session()
        keras.utils.set_random_seed(seed)
        model = builder(input_size=data.x_train.shape[1], class_count=len(data.class_names))
        results.append(train_and_evaluate(model, name, data, epochs, batch_size, seed))

    print("\nComparacion en el conjunto de prueba")
    for result in results:
        print(
            f"  {result.name:12} exactitud={result.test_accuracy:.2%} "
            f"perdida={result.test_loss:.4f}"
        )

    save_artifacts(results, data, output_dir, seed)
    print(f"\nModelos, graficas y metadatos guardados en: {output_dir.resolve()}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compara redes Keras Sequential y funcionales con el conjunto Iris."
    )
    parser.add_argument("--epochs", type=int, default=150, help="Epocas de entrenamiento.")
    parser.add_argument("--batch-size", type=int, default=16, help="Muestras por lote.")
    parser.add_argument("--seed", type=int, default=42, help="Semilla para reproducir los resultados.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("resultados"),
        help="Carpeta donde se guardan modelos, metricas y graficas.",
    )
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs debe ser mayor que cero.")
    if args.batch_size < 1:
        parser.error("--batch-size debe ser mayor que cero.")
    run_experiment(args.epochs, args.batch_size, args.seed, args.output_dir)


if __name__ == "__main__":
    main()
