import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


EX05 = Path(__file__).resolve().parent


# Calcula la distancia entre dos knights
def distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


# Predice un knight usando sus K vecinos más cercanos
def predict_knn(X_train, y_train, row, k):
    distances = []

    # Calculamos la distancia con todos los knights conocidos
    for i in range(len(X_train)):
        dist = distance(row, X_train[i])
        distances.append((dist, y_train[i]))

    # Ordenamos de más cercano a más lejano
    distances.sort(key=lambda x: x[0])

    jedi = 0
    sith = 0

    # Miramos solamente los K vecinos más cercanos
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
        print("Usage: python3 KNN.py Train_knight.csv Test_knight.csv")
        return

    # Leemos los dos CSV recibidos
    train = pd.read_csv(sys.argv[1])
    test = pd.read_csv(sys.argv[2])

    # Cogemos todas las características menos knight
    features = []

    for column in train.columns:
        if column != "knight":
            features.append(column)

    # --------------------------------------------------
    # PREPARAMOS TRAIN
    # --------------------------------------------------

    X = train[features].values

    y = []

    for knight in train["knight"]:
        if knight == "Jedi":
            y.append(0)
        else:
            y.append(1)

    y = np.array(y)

    # --------------------------------------------------
    # VALIDATION
    # --------------------------------------------------
    # Separamos temporalmente Train:
    # 80% para aprender
    # 20% para probar distintos valores de K

    data = train.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    split_index = int(len(data) * 0.8)

    training = data[:split_index]
    validation = data[split_index:]

    X_train = training[features].values
    X_validation = validation[features].values

    y_train = []

    for knight in training["knight"]:
        if knight == "Jedi":
            y_train.append(0)
        else:
            y_train.append(1)

    y_train = np.array(y_train)

    y_validation = []

    for knight in validation["knight"]:
        if knight == "Jedi":
            y_validation.append(0)
        else:
            y_validation.append(1)

    y_validation = np.array(y_validation)

    # --------------------------------------------------
    # NORMALIZACIÓN
    # --------------------------------------------------

    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)

    std[std == 0] = 1

    X_train = (X_train - mean) / std
    X_validation = (X_validation - mean) / std

    # --------------------------------------------------
    # BUSCAMOS EL MEJOR K
    # --------------------------------------------------

    k_values = []
    accuracies = []

    best_k = 1
    best_accuracy = 0

    # Probamos solamente K impares para evitar empates
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
        accuracies.append(current_accuracy * 100)

        print(
            f"K = {k:2d} -> "
            f"Accuracy = {current_accuracy * 100:.2f}%"
        )

        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            best_k = k

    print()
    print("Best K:", best_k)
    print(f"Best accuracy: {best_accuracy * 100:.2f}%")

    # --------------------------------------------------
    # GRÁFICO
    # --------------------------------------------------

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

    # --------------------------------------------------
    # AHORA USAMOS TODO TRAIN PARA EL TEST FINAL
    # --------------------------------------------------

    X_train = X
    y_train = y

    X_test = test[features].values

    # Volvemos a calcular normalización usando TODO Train
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)

    std[std == 0] = 1

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    # --------------------------------------------------
    # PREDICCIONES
    # --------------------------------------------------

    predictions = []

    for row in X_test:
        result = predict_knn(
            X_train,
            y_train,
            row,
            best_k
        )

        if result == 0:
            predictions.append("Jedi")
        else:
            predictions.append("Sith")

    # --------------------------------------------------
    # KNN.TXT
    # --------------------------------------------------

    with open(EX05 / "KNN.txt", "w") as file:
        for prediction in predictions:
            file.write(prediction + "\n")

    plt.show()


if __name__ == "__main__":
    main()