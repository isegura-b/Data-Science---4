import sys
from pathlib import Path
import matplotlib.pyplot as plt


EX00 = Path(__file__).resolve().parent


def main():
    if len(sys.argv) != 3:
        print("Usage: python3 Confusion_Matrix.py predictions.txt truth.txt")
        return

    # Guardamos las rutas que nos pasan por terminal
    predictions_file = sys.argv[1]
    truth_file = sys.argv[2]

    # Predictions.txt
    try:
        with open(predictions_file, "r") as file:
            predictions = file.read().splitlines()
    except FileNotFoundError:
        print("Error: predictions file not found")
        return

    # Truth.txt
    try:
        with open(truth_file, "r") as file:
            truth = file.read().splitlines()
    except FileNotFoundError:
        print("Error: truth file not found")
        return

    # Ambos archivos tienen que tener el mismo número de líneas
    if len(predictions) != len(truth):
        print("Error: files have different number of predictions")
        return

    # Contadores de la matriz
    jedi_jedi = 0
    jedi_sith = 0
    sith_jedi = 0
    sith_sith = 0

    # Recorremos todas las predicciones
    for i in range(len(truth)):
        real = truth[i]
        predicted = predictions[i]

        if real not in ["Jedi", "Sith"]:
            print("Error: invalid value in truth.txt:", real)
            return

        if predicted not in ["Jedi", "Sith"]:
            print("Error: invalid value in predictions.txt:", predicted)
            return

        # REAL Jedi - PREDICCIÓN Jedi
        if real == "Jedi" and predicted == "Jedi":
            jedi_jedi += 1

        # REAL Jedi - PREDICCIÓN Sith
        elif real == "Jedi" and predicted == "Sith":
            jedi_sith += 1

        # REAL Sith - PREDICCIÓN Jedi
        elif real == "Sith" and predicted == "Jedi":
            sith_jedi += 1

        # REAL Sith - PREDICCIÓN Sith
        elif real == "Sith" and predicted == "Sith":
            sith_sith += 1

    # JEDI
    # Precision
    jedi_predicted = jedi_jedi + sith_jedi

    if jedi_predicted != 0:
        jedi_precision = jedi_jedi / jedi_predicted
    else:
        jedi_precision = 0

    # Recall
    jedi_total = jedi_jedi + jedi_sith

    if jedi_total != 0:
        jedi_recall = jedi_jedi / jedi_total
    else:
        jedi_recall = 0

    # F1-score
    if jedi_precision + jedi_recall != 0:
        jedi_f1 = (
            2 * jedi_precision * jedi_recall
            / (jedi_precision + jedi_recall)
        )
    else:
        jedi_f1 = 0

    # SITH
    # Precision
    sith_predicted = sith_sith + jedi_sith

    if sith_predicted != 0:
        sith_precision = sith_sith / sith_predicted
    else:
        sith_precision = 0

    # Recall
    sith_total = sith_sith + sith_jedi

    if sith_total != 0:
        sith_recall = sith_sith / sith_total
    else:
        sith_recall = 0

    # F1-score
    if sith_precision + sith_recall != 0:
        sith_f1 = (
            2 * sith_precision * sith_recall
            / (sith_precision + sith_recall)
        )
    else:
        sith_f1 = 0

    # ACCURACY
    total = len(truth)
    correct = jedi_jedi + sith_sith

    if total != 0:
        accuracy = correct / total
    else:
        accuracy = 0

    # Print
    print("             precision recall f1-score total")

    print(
        f"Jedi         {jedi_precision:.2f}      "
        f"{jedi_recall:.2f}   {jedi_f1:.2f}     {jedi_total}"
    )

    print(
        f"Sith         {sith_precision:.2f}      "
        f"{sith_recall:.2f}   {sith_f1:.2f}     {sith_total}"
    )

    print(f"accuracy                      {accuracy:.2f}     {total}")

    # Creamos la matriz
    matrix = [
        [jedi_jedi, jedi_sith],
        [sith_jedi, sith_sith]
    ]

    print()
    print(matrix)

    # Mostramos la matriz como heatmap
    plt.imshow(matrix)

    plt.xticks([0, 1], ["Jedi", "Sith"])
    plt.yticks([0, 1], ["Jedi", "Sith"])

    plt.xlabel("Predicted")
    plt.ylabel("Truth")
    plt.title("Confusion Matrix")

    # Escribimos los números dentro de cada cuadrado
    for i in range(2):
        for j in range(2):
            plt.text(
                j,
                i,
                matrix[i][j],
                ha="center",
                va="center"
            )

    plt.savefig(EX00 / "Confusion_Matrix.png")
    plt.show()


if __name__ == "__main__":
    main()