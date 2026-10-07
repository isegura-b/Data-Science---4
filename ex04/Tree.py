import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Carpeta donde está Tree.py.
# La usamos para guardar Tree.txt y Tree.png dentro de ex04.
EX04 = Path(__file__).resolve().parent


# ---------------------------------------------------------
# NODO DEL ÁRBOL
# ---------------------------------------------------------
# feature    -> qué característica usamos
# threshold  -> valor con el que comparamos
# left       -> rama izquierda si se cumple la condición
# right      -> rama derecha si NO se cumple
# prediction -> Jedi/Sith si este nodo ya es una hoja
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


# ---------------------------------------------------------
# GINI
# ---------------------------------------------------------
def gini(y):
    if len(y) == 0:
        return 0
    jedi = np.sum(y == 0)
    sith = np.sum(y == 1)

    total = len(y)

    # Calculamos qué porcentaje del grupo es Jedi y qué porcentaje es Sith
    jedi_ratio = jedi / total
    sith_ratio = sith / total

    # Fórmula del Gini.
    # Si todo el grupo es Jedi o todo es Sith, el resultado será 0.
    return 1 - (jedi_ratio ** 2 + sith_ratio ** 2)


# ---------------------------------------------------------
# BUSCAR LA MEJOR PREGUNTA
# ---------------------------------------------------------
# Aquí probamos muchos posibles "if".

# 1. dividimos los datos en izquierda y derecha
# 2. calculamos el Gini de ambos lados
# 3. nos quedamos con la pregunta que deje menos mezcla
def best_split(X, y):
    best_feature = None
    best_threshold = None
    best_gini = float("inf")

    # Recorremos todas las características
    for feature in range(X.shape[1]):

        # Cogemos todos los valores diferentes de esta característica
        values = np.unique(X[:, feature])

        for i in range(len(values) - 1):

            threshold = (values[i] + values[i + 1]) / 2

            # Grupo izquierdo:
            left_mask = X[:, feature] <= threshold

            # Grupo derecho:
            right_mask = X[:, feature] > threshold

            # Nos quedamos solo con las etiquetas Jedi/Sith de cada grupo
            left_y = y[left_mask]
            right_y = y[right_mask]

            if len(left_y) == 0 or len(right_y) == 0:
                continue

            left_gini = gini(left_y)
            right_gini = gini(right_y)

            # Calculamos el Gini total del corte.
            # Si un grupo tiene más datos, pesa más en el resultado.
            split_gini = (
                (len(left_y) / len(y)) * left_gini
                +
                (len(right_y) / len(y)) * right_gini
            )

            if split_gini < best_gini:
                best_gini = split_gini
                best_feature = feature
                best_threshold = threshold

    return best_feature, best_threshold


# ---------------------------------------------------------
# CONSTRUIR EL ÁRBOL
# ---------------------------------------------------------
# Esta función crea el árbol completo.
#
# Hace:
# 1. mira el grupo actual
# 2. busca la mejor pregunta
# 3. divide izquierda/derecha
# 4. vuelve a hacer lo mismo en cada lado
def build_tree(X, y, depth=0, max_depth=7):
    jedi = np.sum(y == 0)
    sith = np.sum(y == 1)

    # Guardamos cuál es la clase mayoritaria de este grupo.
    if jedi >= sith:
        prediction = 0
    else:
        prediction = 1

    # Si todos son Jedi o todos son Sith, ya no hace falta seguir dividiendo.
    if jedi == 0 or sith == 0:
        return Node(prediction=prediction)

    if depth >= max_depth:
        return Node(prediction=prediction)

    # Buscamos la mejor pregunta para separar este grupo
    feature, threshold = best_split(X, y)

    # Si no encontramos ninguna división válida,
    # convertimos este nodo en una hoja.
    if feature is None:
        return Node(prediction=prediction)

    # Dividimos los datos según la pregunta encontrada
    left_mask = X[:, feature] <= threshold
    right_mask = X[:, feature] > threshold

    # Construimos la rama izquierda.
    left = build_tree(
        X[left_mask],
        y[left_mask],
        depth + 1,
        max_depth
    )

    # Construimos la rama derecha.
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


# ---------------------------------------------------------
# HACER UNA PREDICCIÓN
# ---------------------------------------------------------
# Cogemos una sola fila del Test y recorremos el árbol.

def predict(node, row):

    if node.prediction is not None:
        return node.prediction

    if row[node.feature] <= node.threshold:
        return predict(node.left, row)

    return predict(node.right, row)


# ---------------------------------------------------------
# DIBUJAR EL ÁRBOL
# ---------------------------------------------------------
def draw_tree(node, names, x, y, space, ax):
    # Si es una hoja, escribimos Jedi o Sith
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


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():

    if len(sys.argv) != 3:
        print("Usage: python3 Tree.py Train_knight.csv Test_knight.csv")
        return

    train = pd.read_csv(sys.argv[1])
    test = pd.read_csv(sys.argv[2])

    features = []

    for column in train.columns:
        if column != "knight":
            features.append(column)

    # X_train contiene solo las características que el árbol usará para aprender.
    X_train = train[features].values

    # Jedi = 0
    # Sith = 1
    y_train = []

    for knight in train["knight"]:
        if knight == "Jedi":
            y_train.append(0)
        else:
            y_train.append(1)

    y_train = np.array(y_train)

    # X_test contiene los knights que queremos clasificar.
    X_test = test[features].values

    # -----------------------------------------------------
    # ENTRENAMIENTO
    # -----------------------------------------------------
    # Construimos el árbol usando los datos de Train.
    tree = build_tree(X_train, y_train)

    # -----------------------------------------------------
    # PREDICCIONES
    # -----------------------------------------------------
    predictions = []

    # Cogemos cada knight del Test
    for row in X_test:

        # Lo hacemos pasar por el árbol
        result = predict(tree, row)

        # Convertimos 0/1 otra vez a Jedi/Sith
        if result == 0:
            predictions.append("Jedi")
        else:
            predictions.append("Sith")

    # -----------------------------------------------------
    # GUARDAR TREE.TXT
    # -----------------------------------------------------
    with open(EX04 / "Tree.txt", "w") as file:
        for prediction in predictions:
            file.write(prediction + "\n")

    # -----------------------------------------------------
    # DIBUJAR Y GUARDAR EL ÁRBOL
    # -----------------------------------------------------
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