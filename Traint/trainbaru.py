import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Tentukan path dataset
base_dir = 'E:\Kuliah\Semester 5\Big Data Analytics\Coba Detect\Eksperiment\dataset'

# Membuat path untuk setiap subset data
train_dir = 'E:/Kuliah/Semester 5/Big Data Analytics/Coba Detect/Eksperiment/dataset'
test_dir = 'E:/Kuliah/Semester 5/Big Data Analytics/Coba Detect/Eksperiment/test'

# Membuat generator data untuk augmentasi gambar pada data latih
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Membuat generator data uji tanpa augmentasi
test_datagen = ImageDataGenerator(rescale=1./255)

# Membuat generator data latih dan data uji
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

# Membangun model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

# Mengompilasi model
model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.RMSprop(lr=1e-4),
              metrics=['accuracy'])

# Melatih model dengan data latih
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=10
)

# Evaluasi model dengan data uji
test_loss, test_acc = model.evaluate(test_generator, steps=len(test_generator))
print(f'Test accuracy: {test_acc}')

# Menampilkan parameter dan metrik dari pelatihan
history_dict = history.history
epochs = range(1, len(history_dict['accuracy']) + 1)

# Plot akurasi pelatihan
plt.plot(epochs, history_dict['accuracy'], 'bo', label='Training acc')
plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
