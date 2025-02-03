# pierwsza przymiarka, takie hello world dla tensorflow
# uruchomienie:
# python3 01_model_sieci.py < dane_treningowe.txt

import tensorflow as tf
import os
import numpy as np

from read_training_data import read_training_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

X, Y = read_training_data()
X = np.array(X)
Y = np.array(Y)

num_features = X.shape[-1]  # liczba cech wejściowych (np. 17)
num_outputs = Y.shape[-1]  # liczba wyjść (np. 7)

print(num_features)
print(num_outputs)

# Więcej funkcji aktywacji, albo ich paramtryzacja znajduje się na stronach tensorflow https://www.tensorflow.org/api_docs/python/tf/keras/activations
# Definicja modelu
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(num_features,)),  # Warstwa wejściowa
    tf.keras.layers.Dense(20, activation='linear'),  # Pierwsza warstwa ukryta (liniowa aktywacja)
    tf.keras.layers.Dense(37, activation='relu'),    # Druga warstwa ukryta (ReLU)
    tf.keras.layers.Dense(num_outputs, activation='linear')  # Warstwa wyjściowa (regresja)
])

# Kompilacja modelu
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Podgląd architektury modelu
model.summary()

# Trenowanie modelu
history = model.fit(X, Y,
    epochs=50,# Maksymalna liczba epok
    batch_size=32,            # Rozmiar batcha
    validation_split=0.2,     # 20% danych na walidację
)