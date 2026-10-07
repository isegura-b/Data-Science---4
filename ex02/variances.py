import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

#PCA = Principal Component Analysis → Análisis de Componentes Principales

EX02 = Path(__file__).resolve().parent
SUBJECT = EX02.parent / "subject"


def main():
    data = pd.read_csv(SUBJECT / "Train_knight.csv")

    numeric_data = data.select_dtypes(include="number")
    values = numeric_data.values

    # --------------------------------------------------
    # 1. ESTANDARIZAMOS LOS DATOS
    # --------------------------------------------------
    means = np.mean(values, axis=0)
    stds = np.std(values, axis=0)

    scaled_data = (values - means) / stds

    # --------------------------------------------------
    # 2. MATRIZ DE COVARIANZA
    # --------------------------------------------------
    # La covarianza nos dice cómo se mueven las variables entre ellas.

    # Con esto PCA puede entender qué columnas contienen información parecida o relacionada.
    covariance_matrix = np.cov(scaled_data, rowvar=False)

    # --------------------------------------------------
    # 3. EIGENVECTORS Y EIGENVALUES
    # --------------------------------------------------
    # Los eigenvectors representan las nuevas direcciones que encuentra PCA.

    # Los eigenvalues nos dicen cuánta información (varianza) contiene cada una de esas direcciones.
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # --------------------------------------------------
    # 4. ORDENAMOS LOS COMPONENTES
    # --------------------------------------------------
    eigenvalues = eigenvalues[::-1]

    # --------------------------------------------------
    # 5. PASAMOS LOS EIGENVALUES A PORCENTAJE
    # --------------------------------------------------
    # Sumamos toda la varianza disponible.
    total_variance = np.sum(eigenvalues)

    # Calculamos qué porcentaje de información
    # explica cada componente.

    variances = (eigenvalues / total_variance) * 100

    # --------------------------------------------------
    # 6. VARIANZA ACUMULADA
    # --------------------------------------------------

    cumulative = []
    total = 0

    for variance in variances:
        total += variance
        cumulative.append(total)

    # Mostramos la información que aporta cada componente
    print("Variances (Percentage):")
    print(variances)

    print()

    # Mostramos la suma acumulada
    print("Cumulative Variances (Percentage):")
    print(cumulative)

    # --------------------------------------------------
    # 7. BUSCAMOS CUÁNTOS COMPONENTES NECESITAMOS
    # --------------------------------------------------
    # Vamos recorriendo la varianza acumulada hasta
    # llegar o superar el 90%.
    components_needed = 0

    for value in cumulative:
        components_needed += 1

        if value >= 90:
            break

    print()
    print("Components needed to reach 90%:", components_needed)

    # --------------------------------------------------
    # 8. GRÁFICO
    # --------------------------------------------------
    # Eje X = número de componentes utilizados
    # Eje Y = porcentaje de información acumulada
    plt.plot(
        range(1, len(cumulative) + 1),
        cumulative,
        marker="o"
    )
    plt.axhline(y=90, linestyle="--")

    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Variance (%)")
    plt.title("Cumulative Explained Variance")

    plt.grid()

    # Guardamos el gráfico
    plt.savefig(EX02 / "variances.png")
    plt.show()


if __name__ == "__main__":
    main()