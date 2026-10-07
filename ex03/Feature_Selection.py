import pandas as pd
import numpy as np
from pathlib import Path

#VIF = Variance Inflation Factor → Factor de Inflación de la Varianza

EX03 = Path(__file__).resolve().parent
SUBJECT = EX03.parent / "subject"


def calculate_vif(data):
    vif_values = []

    for i in range(data.shape[1]):

        y = data[:, i]
        X = np.delete(data, i, axis=1)

        # Añadimos una columna de 1 para el término independiente
        # Para R = a * Recovery + b * Stims + c * Empowered + d * ...
        ones = np.ones((X.shape[0], 1))
        X = np.hstack((ones, X))

        # Regresión lineal
        coefficients = np.linalg.lstsq(X, y, rcond=None)[0]

        # Predicción de y usando las otras variables
        y_pred = X @ coefficients

        # Calculamos R²
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum((y - y_pred) ** 2)

        r_squared = 1 - (ss_residual / ss_total)

        # Calculamos VIF
        if r_squared >= 1:
            vif = float("inf")
        else:
            vif = 1 / (1 - r_squared)

        vif_values.append(vif)

    return vif_values


def main():

    data = pd.read_csv(SUBJECT / "Train_knight.csv")
    numeric_data = data.select_dtypes(include="number")

    while True:
        values = numeric_data.values

        # vif = 1 / (1 - r_squared)
        vif_values = calculate_vif(values)

        print()
        print("Feature                 VIF")

        for i in range(len(numeric_data.columns)):
            print(
                f"{numeric_data.columns[i]:20} "
                f"{vif_values[i]:.2f}"
            )

        # Buscamos el VIF más alto
        max_vif = max(vif_values)

        # Si todos están por debajo de 5, terminamos
        if max_vif < 5:
            break

        # Buscamos qué columna tiene el VIF más alto
        max_index = vif_values.index(max_vif)
        column_to_remove = numeric_data.columns[max_index]

        print()
        print("Removing:", column_to_remove)

        # Eliminamos esa columna
        numeric_data = numeric_data.drop(columns=[column_to_remove])

    print()
    print("Final features:")

    for column in numeric_data.columns:
        print(column)


if __name__ == "__main__":
    main()