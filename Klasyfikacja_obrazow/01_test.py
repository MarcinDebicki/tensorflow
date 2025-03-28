import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import os

# Nazwa pliku modelu
model_file_name = "01_model_sieci_zapisany.keras"

# Wczytanie modelu
model = keras.models.load_model(model_file_name)

# Klasy (załóżmy, że model trenował na "cats" i "dogs")
class_names = ["cat", "dog"]  # Musisz upewnić się, że są w tej samej kolejności, co w zbiorze treningowym

model.summary()

# Ścieżka do obrazu do klasyfikacji
image_path = "cat2.png"  # Podmień na swoją ścieżkę

# Parametry zgodne z trenowaniem modelu
img_size = (300, 300)  # Musi być zgodne z tym, co było używane przy trenowaniu

# Wczytanie i przetworzenie obrazu
img = image.load_img(image_path, target_size=img_size)
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Dodanie wymiaru batch
img_array = img_array / 255.0  # Normalizacja jak w trenowaniu

# Klasyfikacja
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions)  # Pobranie indeksu klasy
confidence = np.max(predictions)  # Pewność predykcji

probabilities = model.predict(img_array)[0]

# Wynik
print(f"Model przewiduje: cat z pewnością {probabilities[0]:.2%}")
print(f"Model przewiduje: dog z pewnością {probabilities[1]:.2%}")
