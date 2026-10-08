import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


EX05 = Path(__file__).resolve().parent

# K elegido después de probar con Validation_knight.csv
K = 5


# Calcula la distancia entre dos knights
def distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


# Predice una fila usando los K vecinos más cercanos
def predict_knn(X_train, y_train, row, k):
    distances = []

    # Calculamos la distancia entre el knight nuevo
    # y todos los knights conocidos
    for i in range(len(X_train)):
        dist = distance(row, X_train[i])

        # Guardamos distancia + clase
        distances.append((dist, y_train[i]))

    # Ordenamos de menor distancia a mayor
    distances.sort(key=lambda x: x[0])

    jedi = 0
    sith = 0

    # Miramos solo los K vecinos más cercanos
    for i in range(k):
        label = distances[i][1]

        if label == 0:
            jedi += 1
        else:
            sith += 1

    # La mayoría decide
    if jedi > sith:
        return 0

    return 1


# Calcula el porcentaje de aciertos
def accuracy(y_real, y_pred):
    correct = 0

    for i in range(len(y_real)):
        if y_real[i] == y_pred[i]:
            correct += 1

    return correct / len(y_real)


def main():
    if len(sys.argv) != 3:
        print("Usage: python3 KNN.py Training_knight.csv Test_knight.csv")
        return

    # Primer argumento -> datos para aprender
    train = pd.read_csv(sys.argv[1])

    # Segundo argumento -> datos que queremos comprobar o predecir
    test = pd.read_csv(sys.argv[2])

    # --------------------------------------------------
    # CARACTERÍSTICAS
    # --------------------------------------------------

    features = []

    for column in train.columns:
        if column != "knight":
            features.append(column)

    # --------------------------------------------------
    # TRAINING
    # --------------------------------------------------

    X_train = train[features].values

    y_train = []

    for knight in train["knight"]:

        if knight == "Jedi":
            y_train.append(0)

        else:
            y_train.append(1)

    y_train = np.array(y_train)

    # --------------------------------------------------
    # NORMALIZACIÓN
    # --------------------------------------------------
    # KNN trabaja con distancias.
    # Normalizamos para que todas las columnas tengan
    # una escala parecida.

    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)

    # Evitamos dividir entre 0
    std[std == 0] = 1

    X_train = (X_train - mean) / std

    # --------------------------------------------------
    # SI EL SEGUNDO CSV TIENE "knight"
    # ESTAMOS USANDO VALIDATION_KNIGHT.CSV
    # --------------------------------------------------

    if "knight" in test.columns:

        X_validation = test[features].values

        y_validation = []

        for knight in test["knight"]:

            if knight == "Jedi":
                y_validation.append(0)

            else:
                y_validation.append(1)

        y_validation = np.array(y_validation)

        # Normalizamos Validation usando SIEMPRE
        # la media y desviación del Training
        X_validation = (
            X_validation - mean
        ) / std

        k_values = []
        accuracies = []

        best_k = 1
        best_accuracy = 0

        # Probamos diferentes valores de K
        for k in range(1, 30, 2):

            predictions = []

            for row in X_validation:

                result = predict_knn(
                    X_train,
                    y_train,
                    row,
                    k
                )

                predictions.append(result)

            current_accuracy = accuracy(
                y_validation,
                predictions
            )

            k_values.append(k)
            accuracies.append(
                current_accuracy * 100
            )

            print(
                f"K = {k:2d} -> "
                f"Accuracy = {current_accuracy * 100:.2f}%"
            )

            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                best_k = k

        print()
        print("Best K:", best_k)
        print(
            f"Best accuracy: "
            f"{best_accuracy * 100:.2f}%"
        )

        # Gráfico de K
        plt.plot(
            k_values,
            accuracies,
            marker="o"
        )

        plt.xlabel("K value")
        plt.ylabel("Accuracy (%)")
        plt.title("KNN Validation Accuracy")
        plt.grid()

        plt.savefig(EX05 / "KNN.png")
        plt.show()

        return

    # --------------------------------------------------
    # SI EL SEGUNDO CSV NO TIENE "knight"
    # HACEMOS LAS PREDICCIONES
    # --------------------------------------------------

    X_test = test[features].values

    # Normalizamos con los datos del Training
    X_test = (
        X_test - mean
    ) / std

    predictions = []

    for row in X_test:

        result = predict_knn(
            X_train,
            y_train,
            row,
            K
        )

        if result == 0:
            predictions.append("Jedi")

        else:
            predictions.append("Sith")

    # --------------------------------------------------
    # GUARDAMOS KNN.TXT
    # --------------------------------------------------

    with open(EX05 / "KNN.txt", "w") as file:

        for prediction in predictions:
            file.write(prediction + "\n")


if __name__ == "__main__":
    main()