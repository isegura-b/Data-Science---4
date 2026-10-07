import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


EX04 = Path(__file__).resolve().parent


# Cada Node representa una caja del árbol
class Node:
    def __init__(
        self,
        feature=None,
        threshold=None,
        left=None,
        right=None,
        prediction=None
    ):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.prediction = prediction


# Calcula cuánto están mezclados Jedi y Sith
def gini(y):
    if len(y) == 0:
        return 0

    jedi = np.sum(y == 0)
    sith = np.sum(y == 1)

    total = len(y)

    jedi_ratio = jedi / total
    sith_ratio = sith / total

    return 1 - (jedi_ratio ** 2 + sith_ratio ** 2)


# Busca la mejor pregunta:
# por ejemplo: Strength <= 0.52
def best_split(X, y):
    best_feature = None
    best_threshold = None
    best_gini = float("inf")

    # Recorremos todas las características
    for feature in range(X.shape[1]):

        values = np.unique(X[:, feature])

        # Probamos cortes entre valores consecutivos
        for i in range(len(values) - 1):

            threshold = (values[i] + values[i + 1]) / 2

            left_mask = X[:, feature] <= threshold
            right_mask = X[:, feature] > threshold

            left_y = y[left_mask]
            right_y = y[right_mask]

            if len(left_y) == 0 or len(right_y) == 0:
                continue

            # Gini de cada lado
            left_gini = gini(left_y)
            right_gini = gini(right_y)

            # Gini total del corte
            split_gini = (
                (len(left_y) / len(y)) * left_gini
                +
                (len(right_y) / len(y)) * right_gini
            )

            # Nos quedamos con el corte que mejor separa
            if split_gini < best_gini:
                best_gini = split_gini
                best_feature = feature
                best_threshold = threshold

    return best_feature, best_threshold


# Construye el árbol
def build_tree(X, y, depth=0, max_depth=7):
    jedi = np.sum(y == 0)
    sith = np.sum(y == 1)

    # Clase mayoritaria de este nodo
    if jedi >= sith:
        prediction = 0
    else:
        prediction = 1

    # Si todos son de la misma clase, terminamos
    if jedi == 0 or sith == 0:
        return Node(prediction=prediction)

    # Limitamos la profundidad del árbol
    if depth >= max_depth:
        return Node(prediction=prediction)

    # Buscamos la mejor pregunta
    feature, threshold = best_split(X, y)

    if feature is None:
        return Node(prediction=prediction)

    # Dividimos los datos en izquierda y derecha
    left_mask = X[:, feature] <= threshold
    right_mask = X[:, feature] > threshold

    # Creamos las dos ramas
    left = build_tree(
        X[left_mask],
        y[left_mask],
        depth + 1,
        max_depth
    )

    right = build_tree(
        X[right_mask],
        y[right_mask],
        depth + 1,
        max_depth
    )

    return Node(
        feature=feature,
        threshold=threshold,
        left=left,
        right=right
    )


# Predice una sola fila
def predict(node, row):
    # Si estamos en una hoja, devolvemos Jedi o Sith
    if node.prediction is not None:
        return node.prediction

    # Seguimos el árbol según la pregunta
    if row[node.feature] <= node.threshold:
        return predict(node.left, row)

    return predict(node.right, row)


# Dibuja el árbol
def draw_tree(node, names, x, y, space, ax):
    if node.prediction is not None:

        if node.prediction == 0:
            text = "Jedi"
        else:
            text = "Sith"

        ax.text(
            x,
            y,
            text,
            ha="center",
            bbox=dict(boxstyle="round")
        )

        return

    # Pregunta del nodo
    text = f"{names[node.feature]}\n<= {node.threshold:.3f}"

    ax.text(
        x,
        y,
        text,
        ha="center",
        bbox=dict(boxstyle="round")
    )

    child_y = y - 1

    left_x = x - space
    right_x = x + space

    # Líneas hacia las ramas
    ax.plot([x, left_x], [y, child_y])
    ax.plot([x, right_x], [y, child_y])

    draw_tree(
        node.left,
        names,
        left_x,
        child_y,
        space / 2,
        ax
    )

    draw_tree(
        node.right,
        names,
        right_x,
        child_y,
        space / 2,
        ax
    )


def main():
    # El programa necesita:
    # Train_knight.csv Test_knight.csv
    if len(sys.argv) != 3:
        print("Usage: python3 Tree.py Train_knight.csv Test_knight.csv")
        return

    train = pd.read_csv(sys.argv[1])
    test = pd.read_csv(sys.argv[2])

    # Las características son todas menos knight
    features = []

    for column in train.columns:
        if column != "knight":
            features.append(column)

    # Datos de entrenamiento
    X_train = train[features].values

    # Convertimos Jedi/Sith a números
    y_train = []

    for knight in train["knight"]:
        if knight == "Jedi":
            y_train.append(0)
        else:
            y_train.append(1)

    y_train = np.array(y_train)

    # Datos que queremos predecir
    X_test = test[features].values

    # Entrenamos el árbol
    tree = build_tree(X_train, y_train)

    # Hacemos las predicciones
    predictions = []

    for row in X_test:
        result = predict(tree, row)

        if result == 0:
            predictions.append("Jedi")
        else:
            predictions.append("Sith")

    # Guardamos Tree.txt
    with open(EX04 / "Tree.txt", "w") as file:
        for prediction in predictions:
            file.write(prediction + "\n")

    # Dibujamos el árbol
    fig, ax = plt.subplots(figsize=(18, 10))

    draw_tree(
        tree,
        features,
        x=0,
        y=0,
        space=16,
        ax=ax
    )

    ax.axis("off")
    plt.title("Decision Tree")

    plt.savefig(
        EX04 / "Tree.png",
        bbox_inches="tight"
    )

    plt.show()


if __name__ == "__main__":
    main()