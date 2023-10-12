import cv2
import numpy as np
import h5py

# Nama file untuk menyimpan data pelatihan
train_data_file = 'model.h5'

# Membuat objek detektor kendaraan
detektor = cv2.createBackgroundSubtractorMOG2()

# Membuka video
video = cv2.VideoCapture('1menit.mp4')

# Mengatur ukuran frame
video.set(cv2.CAP_PROP_FRAME_WIDTH, 400)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)

# Inisialisasi penghitung kendaraan
total_kendaraan = 0

# Dictionary untuk menyimpan data kendaraan
data_kendaraan = {}

# List untuk menyimpan data pelatihan
data_pelatihan = []

while True:
    # Membaca frame video
    ret, frame = video.read()

    # Keluar dari loop jika video selesai
    if not ret:
        break

    # Mengubah frame menjadi grayscale
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Menggunakan detektor untuk menghapus background
    mask = detektor.apply(grayscale)

    # Memfilter noise menggunakan operasi opening
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Mendeteksi kendaraan dalam frame
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Mengabaikan kontur kecil
        if cv2.contourArea(contour) < 500:
            continue

        # Mendapatkan ID kendaraan berdasarkan koordinat kontur
        kendaraan_id = tuple(contour[0][0])

        # Jika ID kendaraan belum ada dalam dictionary data kendaraan, tambahkan
        if kendaraan_id not in data_kendaraan:
            data_kendaraan[kendaraan_id] = total_kendaraan
            total_kendaraan += 1

        # Menggambar bounding box pada kendaraan dengan ID yang sesuai dan menampilkan ID
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {data_kendaraan[kendaraan_id]}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Menambahkan data pelatihan
        data_pelatihan.append((frame[y:y+h, x:x+w], data_kendaraan[kendaraan_id]))  # ID kendaraan sebagai label

    # Menampilkan total kendaraan pada frame
    cv2.putText(frame, f"Total Kendaraan: {total_kendaraan}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Menampilkan frame
    cv2.imshow('Deteksi Kendaraan', frame)

    # Menghentikan deteksi dengan menekan tombol 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Melepaskan sumber daya video
video.release()
cv2.destroyAllWindows()

# Simpan data pelatihan dalam format .h5
with h5py.File(train_data_file, 'w') as hf:
    hf.create_dataset("train_images", data=[data[0] for data in data_pelatihan])
    hf.create_dataset("train_labels", data=[data[1] for data in data_pelatihan])