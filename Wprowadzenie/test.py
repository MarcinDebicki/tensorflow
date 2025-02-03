# Program do testowania naszej sieci
# Sposób uruchomienia:
# python3 test.py
# po uruchomieniu podajesz "z palca" co zapodać sieci

import sys
import numpy as np
import tensorflow as tf
import os
import argparse

from tensorflow.keras.models import load_model

from read_training_data import read_training_data
from sklearn.model_selection import train_test_split

from keras_tuner.tuners import BayesianOptimization
from keras_tuner import HyperParameters as hp
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import cross_val_score
from scikeras.wrappers import KerasClassifier

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input

def main():
    # Parsowanie argumentów wiersza poleceń
    parser = argparse.ArgumentParser(description="Wczytuje model i wykonuje predykcje na danych wejściowych.")
    parser.add_argument('model_path', type=str, help='Ścieżka do pliku z zapisanym modelem')
    args = parser.parse_args()

    # Wczytanie modelu
    print(f"Ładowanie modelu z {args.model_path}...")
    model = load_model(args.model_path)
    print("Model wczytany.")

    # Pobieranie danych wejściowych ze strumienia (stdin)
    print("Podaj dane wejściowe w formacie: cecha1,cecha2,... (CTRL+D, aby zakończyć):")
    input_lines = sys.stdin.read().strip().split('\n')

    # Przetwarzanie danych wejściowych
    data = [list(map(float, line.split(','))) for line in input_lines]
    X = np.array(data)

    # Wykonanie predykcji
    predictions = model.predict(X)

    # Wyświetlenie wyników
    print("Wyniki predykcji:")
    for pred in predictions:
        print(pred)

if __name__ == "__main__":
    main()
