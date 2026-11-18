# Clasificacion de flores Iris con redes neuronales

Este proyecto incluye los dos ejercicios del docente: comparacion de redes Keras `Sequential` y API funcional para clasificar flores Iris, y un reconocedor interactivo de digitos manuscritos que carga un modelo guardado.

## Prepararlo en Visual Studio Code (Windows)

Abre esta carpeta en VS Code y ejecuta estos comandos en la terminal integrada de PowerShell:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

En VS Code, selecciona el interprete del proyecto con **Ctrl+Shift+P** → **Python: Select Interpreter** → `.venv\Scripts\python.exe`. Luego puedes:

- Ejecutar la practica de Iris: `.\.venv\Scripts\python.exe red_neuronal.py`
- Entrenar el reconocedor de digitos una vez: `.\.venv\Scripts\python.exe entrenar_digitos.py`
- Iniciar la interfaz para dibujar: `.\.venv\Scripts\python.exe app_digitos.py`
- Presionar **F5** y elegir **Entrenar modelo de digitos** o **Reconocer digitos dibujados**.
- Ejecutar las pruebas: `.\.venv\Scripts\python.exe -m unittest discover -s tests`

El archivo `neurona 1.0.py` se conserva como acceso compatible con el nombre anterior.

## Caso practico: reconocer digitos dibujados

1. `entrenar_digitos.py` carga `load_digits` de scikit-learn (1,797 imagenes de 8 x 8), aplana cada imagen a 64 pixeles y normaliza sus valores de 0-16 al rango 0-1.
2. Separa entrenamiento y prueba de forma estratificada y entrena una red densa `64 -> 128 -> 64 -> 10`, con ReLU y salida softmax. Usa Adam, entropia cruzada, validacion y parada temprana.
3. Guarda el modelo en `modelos/modelo_digitos.keras`, y la exactitud, perdida y clases en `modelos/metadatos_digitos.json`. Tambien genera el reporte de clasificacion, la matriz de confusion y las curvas en `resultados_digitos/`.
4. `app_digitos.py` carga el archivo ya entrenado; no importa el codigo de entrenamiento ni vuelve a ajustar pesos al iniciar. Dibuja con el mouse, centra y normaliza el trazo a 8 x 8, y muestra el digito, confianza y probabilidades de las diez clases.

La primera vez, ejecuta el entrenamiento y espera a que se guarde el modelo. En los siguientes usos, abre directamente la interfaz. Para entrenar de nuevo a proposito, vuelve a ejecutar `entrenar_digitos.py`.

La funcion `preprocess_drawing` convierte una imagen PIL a los 64 valores que espera la red y `predict_digit` devuelve la clase y las diez probabilidades; ambas se pueden reutilizar desde otro programa Python.

## Contenido de la practica

Al ejecutar `red_neuronal.py`, el programa:

- Muestra un vector, una matriz de pesos y el calculo `entrada @ pesos + sesgo` con NumPy.
- Carga Iris desde scikit-learn y separa los datos en entrenamiento, validacion y prueba (60/20/20), manteniendo las proporciones de las tres clases.
- Ajusta `StandardScaler` solo con entrenamiento para evitar fuga de datos y convierte las etiquetas a formato one-hot.
- Construye la misma red con Keras `Sequential` y con la API funcional: `Dense(16, relu)`, `Dense(8, relu)` y `Dense(3, softmax)`.
- Compila con Adam, entropia cruzada categorica y exactitud; entrena con validacion y evalua con el conjunto de prueba.
- Imprime el reporte de clasificacion, la exactitud y una comparacion de los dos modelos.
- Guarda modelos Keras, el escalador, metadatos, matrices de confusion y curvas de aprendizaje en `resultados/`.

La exactitud puede variar segun la semilla y el entrenamiento; el programa informa si alcanza el objetivo de referencia del 85%, sin ocultar el resultado medido.

## Conectar las predicciones a otra aplicacion

Despues de ejecutar el entrenamiento, `predict_flowers` permite cargar el modelo funcional y el escalador y predecir flores nuevas. Las cuatro medidas deben conservar el orden listado en `resultados/metadatos.json` (longitud y ancho del sepalo, longitud y ancho del petalo):

```python
from red_neuronal import predict_flowers

resultado = predict_flowers(
    "resultados/modelo_funcional.keras",
    "resultados/escalador_iris.joblib",
    [5.1, 3.5, 1.4, 0.2],
    ["setosa", "versicolor", "virginica"],
)
print(resultado)
```

## Modulo adicional de regresion NumPy

`red_regresion_numpy.py` conserva la red de regresion hecha desde cero: acepta matrices propias, entrena por retropropagacion y permite predecir y guardar/cargar pesos. Es independiente del ejercicio Iris y tambien se ejecuta con el mismo entorno.

## Qué pasó con el error de la imagen

VS Code estaba ejecutando Python 3.13, pero NumPy no estaba instalado en ese interprete. Ahora `requirements.txt` declara NumPy y las librerias que requiere la practica; instalarlas en `.venv` evita depender de paquetes globales.
