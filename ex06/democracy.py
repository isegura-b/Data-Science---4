from pathlib import Path


EX06 = Path(__file__).resolve().parent
ROOT = EX06.parent


TREE_FILE = ROOT / "ex04" / "Tree.txt"
KNN_FILE = ROOT / "ex05" / "KNN.txt"
LOGISTIC_FILE = EX06 / "Logistic.txt"

OUTPUT = EX06 / "Voting.txt"


# Lee un archivo de predicciones.
def read_predictions(file_path):

    predictions = []

    with open(file_path, "r") as file:

        for line in file:

            prediction = line.strip()

            if prediction != "":
                predictions.append(prediction)

    return predictions


def main():

    # Leemos las predicciones de los 3 modelos.
    tree_predictions = read_predictions(
        TREE_FILE
    )

    knn_predictions = read_predictions(
        KNN_FILE
    )

    logistic_predictions = read_predictions(
        LOGISTIC_FILE
    )

    # Los 3 archivos deben tener
    # exactamente el mismo número de predicciones.
    if (
        len(tree_predictions) != len(knn_predictions)
        or
        len(tree_predictions) != len(logistic_predictions)
    ):
        print("Error: prediction files have different sizes.")
        return

    final_predictions = []

    # Recorremos todas las predicciones.
    for i in range(len(tree_predictions)):

        tree_vote = tree_predictions[i]
        knn_vote = knn_predictions[i]
        logistic_vote = logistic_predictions[i]

        jedi = 0
        sith = 0

        # Voto del Tree.
        if tree_vote == "Jedi":
            jedi += 1
        elif tree_vote == "Sith":
            sith += 1
        else:
            print("Error: invalid prediction in Tree.txt")
            return

        # Voto del KNN.
        if knn_vote == "Jedi":
            jedi += 1
        else:
            sith += 1

        # Voto de Logistic Regression.
        if logistic_vote == "Jedi":
            jedi += 1
        else:
            sith += 1

        # Mayoría.
        if jedi > sith:
            final_predictions.append("Jedi")
        else:
            final_predictions.append("Sith")

    # Guardamos Voting.txt.
    with open(OUTPUT, "w") as file:

        for prediction in final_predictions:
            file.write(prediction + "\n")


if __name__ == "__main__":
    main()