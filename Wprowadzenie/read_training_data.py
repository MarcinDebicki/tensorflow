import sys
import numpy as np
from io import StringIO

def read_training_data():
    X = []
    Y = []

    for line in sys.stdin:

        line = line.strip()  # Usuń białe znaki z początku i końca linii

        if ';' not in line:
            print(f"Błąd w linii '{line}': Brak średnika.")
            exit()
            continue

        # Walidacja 1: Sprawdzenie czy jest średnik
        if ';' not in line:
            print(f"Błąd w linii '{line}': Brak średnika.")
            exit()
            continue
        
        inputs_str, output_str = line.split(';')

        # Walidacja 2: Sprawdzenie czy dane wejściowe są poprawne
        try:
            sequences = [list(map(float, seq.split())) for seq in inputs_str.split(',')]
            if not all(len(seq) == 1 for seq in sequences):
                raise ValueError
        except ValueError:
            print(f"Błąd w linii '{line} \r\n---{inputs_str}---': Błędny format danych wejściowych.")
            exit()
            continue

        # Walidacja 3: Sprawdzenie czy dane wyjściowe są liczbą
        try:
            elements = output_str.split(" ")
            output = [float(x) for x in elements]
        except ValueError:
            print(f"Błąd w linii '{line}***{output_str}***': Błędny format danych wyjściowych.")
            exit()
            continue

        X.append(sequences)
        Y.append(output)

    return X, Y



