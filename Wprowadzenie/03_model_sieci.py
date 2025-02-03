# Dokładamy metodę callbacku, która mówi, że jeśli funkcja straty rośnie to należy zakończyć trening
# Na końcu zapisujemy model do pliku
# uruchomienie:
# python3 01_model_sieci.py < dane_treningowe.txt

plik = "03_model_sieci_zapisany.keras"

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from kerastuner.tuners import BayesianOptimization
import numpy as np
import os
from read_training_data import read_training_data


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

X, Y = read_training_data()
X = np.array(X)
Y = np.array(Y)

num_features = X.shape[-1]  # liczba cech wejściowych (np. 17)
num_outputs = Y.shape[-1]  # liczba wyjść (np. 7)

print(num_features)
print(num_outputs)

# Definicja modelu
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(num_features,)),  # Warstwa wejściowa
    
    tf.keras.layers.Dense(20, activation='linear'),  # Pierwsza warstwa ukryta (liniowa aktywacja)
    tf.keras.layers.Dropout(0.2),  # 20% neuronów losowo wyłączanych

    tf.keras.layers.Dense(37, activation='relu'),    # Druga warstwa ukryta (ReLU)
    tf.keras.layers.Dropout(0.2),  # 20% neuronów losowo wyłączanych

    tf.keras.layers.Dense(num_outputs, activation='linear')  # Warstwa wyjściowa (regresja)
])

# Kompilacja modelu
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Podgląd architektury modelu
model.summary()

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',    # Monitorowanie straty na danych walidacyjnych
    patience=10,            # Ile epok poczekać na poprawę (w tym przypadku 2)
    restore_best_weights=True  # Przywraca wagi z najlepszej epoki
)

# Trenowanie modelu
history = model.fit(X, Y,
    epochs=50,# Maksymalna liczba epok
    batch_size=32,            # Rozmiar batcha
    validation_split=0.2,     # 20% danych na walidację
    callbacks=[early_stopping]  # Dodanie mechanizmu EarlyStopping
)

import os

# Sprawdzenie, czy plik istnieje i jego usunięcie
if os.path.isfile(plik):
    os.remove(plik)

model.save(plik)
