import sys
from pathlib import Path

import pandas as pd
import numpy as np


EX06 = Path(__file__).resolve().parent


# Convierte un número en una probabilidad entre 0 y 1.
def sigmoid(value):
    return 1 / (1 + np.exp(-value))


# Entrena la regresión logística.
def train_logistic(X, y, learning_rate=0.01, iterations=10000):

    # Un peso por cada característica.
    weights = np.zeros(X.shape[1])

    # Bias inicial.
    bias = 0

    # Vamos corrigiendo los pesos poco a poco.
    for i in range(iterations):

        # Resultado lineal.
        linear = X @ weights + bias

        # Lo convertimos en probabilidades.
        predictions = sigmoid(linear)

        # Calculamos el error.
        error = predictions - y

        # Gradiente de los pesos.
        dw = (X.T @ error) / len(y)

        # Gradiente del bias.
        db = np.mean(error)

        # Actualizamos.
        weights = weights - learning_rate * dw
        bias = bias - learning_rate * db

    return weights, bias


# Hace una predicción.
def predict_logistic(row, weights, bias):

    linear = row @ weights + bias

    probability = sigmoid(linear)

    # 0 = Jedi
    # 1 = Sith
    if probability >= 0.5:
        return 1

    return 0


def main():

    if len(sys.argv) != 3:
        print(
            "Usage: python3 Logistic.py "
            "Train_knight.csv Test_knight.csv"
        )
        return

    # Leemos Train y Test.
    train = pd.read_csv(sys.argv[1])
    test = pd.read_csv(sys.argv[2])

    # Cogemos todas las columnas menos knight.
    features = []

    for column in train.columns:
        if column != "knight":
            features.append(column)

    # Datos de entrenamiento.
    X_train = train[features].values

    # Convertimos Jedi/Sith a 0/1.
    y_train = []

    for knight in train["knight"]:

        if knight == "Jedi":
            y_train.append(0)

        else:
            y_train.append(1)

    y_train = np.array(y_train)

    # Datos a predecir.
    X_test = test[features].values

    # Normalizamos.
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)

    # Evitamos división entre 0.
    std[std == 0] = 1

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    # Entrenamos.
    weights, bias = train_logistic(
        X_train,
        y_train
    )

    predictions = []

    # Predecimos cada fila del Test.
    for row in X_test:

        result = predict_logistic(
            row,
            weights,
            bias
        )

        if result == 0:
            predictions.append("Jedi")
        else:
            predictions.append("Sith")

    # Guardamos el resultado.
    with open(EX06 / "Logistic.txt", "w") as file:

        for prediction in predictions:
            file.write(prediction + "\n")


if __name__ == "__main__":
    main()