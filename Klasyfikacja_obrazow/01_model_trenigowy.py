import tensorflow as tf
import os
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom

# dane: https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset

AUTOTUNE = tf.data.AUTOTUNE
TF_CPP_MIN_LOG_LEVEL = "2"

model_file_name = "01_model_sieci_zapisany.keras"

# Wczytanie danych z katalogów "cats" i "dogs"
data_dir = "dane"  # Podmień na prawidłową ścieżkę
img_scale = 300
img_size = (img_scale, img_scale)  # Ujednolicony rozmiar
batch_size = 32

train_ds = image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=img_size,
    batch_size=batch_size,
    label_mode="categorical",
)

class_names = train_ds.class_names
print(class_names)

val_ds = image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=img_size,
    batch_size=batch_size,
    label_mode="categorical",
)

# Prefetch do przyspieszenia przetwarzania
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# Normalizacja pikseli do zakresu [0,1]
normalization_layer = layers.Rescaling(1.0 / 255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# Augmentacja tylko dla zbioru treningowego
data_augmentation = keras.Sequential([
    RandomFlip("horizontal"),
    RandomRotation(0.2),
    RandomZoom(0.2),
])
# Dodanie augmentacji do zbioru treningowego
train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))

# Definiowanie modelu CNN
model = keras.Sequential([
    keras.Input(shape=(img_scale, img_scale, 3)),  # Ujednolicony rozmiar wejścia

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(2, activation="sigmoid")  # 2 wyjścia dla [P_dog, P_cat]
])

# Kompilacja modelu
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Trenowanie modelu
epochs = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# Zapisanie modelu
if os.path.isfile(model_file_name):
    os.remove(model_file_name)

model.save(model_file_name)
