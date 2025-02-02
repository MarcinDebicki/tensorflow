# Dostosowanie hiperparametrów sieci metodą optymalizacji bayesowskiej
# Jest biblioteka ogólnego przeznaczenia scikit-optimize, którą wównież można użyć, ale tutaj używamy wbudowanej optymalizacji bayesowskiej
# Na końcu wyświetlamy najlepsze znalezione parametry modelu
# uruchomienie:
# python3 05_dostosowanie_hiperparametrow_sieci.py < dane_treningowe.txt

plik = "05_model_sieci_zapisany.keras"

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from keras_tuner.tuners import BayesianOptimization  # Poprawiony import
import numpy as np
import os
from read_training_data import read_training_data


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

X, Y = read_training_data()
X = np.array(X)
Y = np.array(Y)

num_features = X.shape[-1]  # liczba cech wejściowych (np. 17)
num_outputs = Y.shape[-1]  # liczba wyjść (np. 7)

print(num_features)
print(num_outputs)

def build_model(hp):
    model = Sequential([
        tf.keras.layers.Input(shape=(num_features,)),

        Dense(
            units=hp.Int('n_neurons_layer1', min_value=10, max_value=250, step=10),  # 10-250, skok co 10
            activation='sigmoid',
            kernel_regularizer=l2(0.01)
        ),
        Dropout(0.2),
        Dense(
            units=hp.Int('n_neurons_layer2', min_value=20, max_value=120, step=5),  # 20-120, skok co 5
            activation='relu',
            kernel_regularizer=l2(0.01)
        ),
        Dropout(0.2),
        Dense(num_outputs, activation='linear')
    ])
    
    # Kompilacja modelu
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Definiowanie tunera optymalizacji bayesowskiej
tuner = BayesianOptimization(
    build_model,
    objective='val_loss',
    max_trials=50,  # Liczba prób tuningowania modelu
    num_initial_points=10,  # Liczba początkowych punktów do zbadania losowo
    seed=42,
)

# Trenowanie modelu i przeszukiwanie przestrzeni hiperparametrów
tuner.search(
    X, Y,
    epochs=50,
    validation_split=0.2,
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)],
    #,verbose=0
)

# Uzyskanie najlepszego modelu i hiperparametrów
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"Najlepsza liczba neuronów w warstwie 1: {best_hps.get('n_neurons_layer1')}")
print(f"Najlepsza liczba neuronów w warstwie 2: {best_hps.get('n_neurons_layer2')}")
