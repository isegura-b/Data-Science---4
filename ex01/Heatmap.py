import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


EX01 = Path(__file__).resolve().parent
SUBJECT = EX01.parent / "subject"


def main():
    data = pd.read_csv(SUBJECT / "Train_knight.csv")
    numeric_data = data.select_dtypes(include="number")

    # Calculamos la correlación entre todas las columnas
    correlation = numeric_data.corr()

    # Creamos el heatmap
    plt.figure(figsize=(12, 12))
    plt.imshow( correlation, vmin=-1, vmax=1 )

    # Ponemos los nombres de las columnas en los ejes
    plt.xticks(
        range(len(correlation.columns)),
        correlation.columns,
        rotation=90
    )
    plt.yticks(
        range(len(correlation.columns)),
        correlation.columns
    )

    plt.title("Correlation Heatmap")

    # Barra lateral de colores
    plt.colorbar()
    # Ajustamos para que no se corten los nombres
    plt.tight_layout()

    plt.savefig(EX01 / "Heatmap.png")
    plt.show()


if __name__ == "__main__":
    main()