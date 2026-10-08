import pandas as pd
import sys
from pathlib import Path


EX05 = Path(__file__).resolve().parent


def main():
    # Comprobamos que nos pasen el CSV
    if len(sys.argv) != 2:
        print("Usage: python3 split.py Train_knight.csv")
        return

    # Leemos el archivo pasado por argumento
    file_path = Path(sys.argv[1])
    data = pd.read_csv(file_path)

    # Mezclamos las filas aleatoriamente
    data = data.sample(frac=1).reset_index(drop=True)

    # 80% para entrenamiento
    split_index = int(len(data) * 0.8)

    training = data[:split_index]
    validation = data[split_index:]

    # Guardamos Training completo
    training.to_csv(
        EX05 / "Training_knight.csv",
        index=False
    )

    # Guardamos Validation completo
    validation.to_csv(
        EX05 / "Validation_knight.csv",
        index=False
    )

    # Guardamos solo las respuestas reales de Validation
    # para comprobar después el F1
    validation["knight"].to_csv(
        EX05 / "validation_truth.txt",
        index=False,
        header=False
    )

    # Guardamos Validation sin la respuesta "knight"
    # para usarlo como si fuera un Test
    validation.drop(columns=["knight"]).to_csv(
        EX05 / "Validation_test.csv",
        index=False
    )

    print("Training:", len(training))
    print("Validation:", len(validation))


if __name__ == "__main__":
    main()