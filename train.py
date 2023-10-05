import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
train_dir = 'E:\Kuliah\Semester 5\Big Data Analytics\Coba Detect\Eksperiment\dataset'

# Tentukan generator data untuk augmentasi gambar
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalisasi nilai piksel gambar
    rotation_range=20,  # Rotasi gambar dalam rentang 20 derajat
    width_shift_range=0.2,  # Geser gambar secara horizontal sebesar 20% lebar gambar
    height_shift_range=0.2,  # Geser gambar secara vertikal sebesar 20% tinggi gambar
    shear_range=0.2,  # Melakukan shear transformation sebesar 20%
    zoom_range=0.2,  # Melakukan zoom gambar sebesar 20%
    horizontal_flip=True,  # Membalikkan gambar secara horizontal
    fill_mode='nearest'  # Mengisi piksel yang kosong dengan piksel terdekat
)
# Membuat generator data latih
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),  # Menyesuaikan ukuran gambar menjadi 150x150 piksel
    batch_size=32,  # Jumlah gambar yang digunakan dalam setiap iterasi pelatihan
    class_mode='categorical'  # Mode kelas untuk tugas klasifikasi multikelas
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
    tf.keras.layers.Dropout(0.5),  # Menggunakan dropout untuk mengurangi overfitting
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')  # Menggunakan softmax untuk output tiga kelas
])

# Mengompilasi model
model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.RMSprop(lr=1e-4),
              metrics=['accuracy'])
# Melatih model
model.fit(
    train_generator,
    steps_per_epoch=100,  # Jumlah batch yang akan dieksekusi dalam setiap epoch
    epochs=10  # Jumlah epoch pelatihan
)
# Tentukan path ke gambar yang akan diuji
test_image_path = 'E:\\Kuliah\\Semester 5\\Big Data Analytics\\Coba Detect\\Eksperiment\\test\\tesmobil.jpg'
# Load gambar dan ubah ukurannya menjadi 150x150 piksel
test_image = image.load_img(test_image_path, target_size=(150, 150))
# Ubah gambar menjadi array numerik
test_image_array = image.img_to_array(test_image)
# Normalisasi nilai piksel gambar
test_image_array = test_image_array / 255.0
# Ubah dimensi gambar menjadi batch dengan dimensi tambahan
test_image_array = np.expand_dims(test_image_array, axis=0)

# Melakukan prediksi menggunakan model
predictions = model.predict(test_image_array)

# Mengambil indeks kelas dengan probabilitas tertinggi
predicted_class_index = np.argmax(predictions)

# Mengurutkan label kelas
class_labels = ['mobil', 'motor', 'sepeda']
# Mendapatkan label kelas yang diprediksi
predicted_class_label = class_labels[predicted_class_index]
# Menampilkan gambar hasil testing
plt.imshow(test_image)
plt.axis('off')
plt.title('Predicted label: ' + predicted_class_label)
plt.show()

# # Simpan model ke file
model.save('model.h5')