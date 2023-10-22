import cv2
import numpy as np
from time import sleep, time, ctime, strptime, strftime
import os
import glob
import requests



def detect_cars(video_path):
    cap = cv2.VideoCapture(video_path)
    car_cascade = cv2.CascadeClassifier('cars.xml')
    delay = 600
    detec = []
    pos_line = 200
    offset = 5
    car = 0

    # Inisialisasi ID kendaraan
    next_car_id = 1
    car_ids = {}  # Dictionary untuk melacak ID kendaraan yang terdeteksi

    # Fungsi center_object() dan loop deteksi kendaraan
    def center_object(x, y, w, h):
        x1 = int(w / 2)
        y1 = int(h / 2)
        cx = x + x1
        cy = y + y1
        return cx, cy
    
    def send_detection_data(car_id, timestamp):
        data = {
            'car_id': car_id,
            'timestamp': timestamp
        }
        url = 'https://wkf6l4sh-4000.asse.devtunnels.ms/save_data'

        response = requests.post(url, json=data)

        if response.status_code == 200:
            print("Data terkirim dengan sukses")
        else:
            print("Gagal mengirim data")

    while True:
        ret, img = cap.read()
        sleep(1 / delay)  # Menggunakan sleep untuk mengatur waktu
        if type(img) == type(None):
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cars = car_cascade.detectMultiScale(gray, 1.1, 1)
        angle = -2  # Sudut kemiringan garis
        radian = angle * np.pi / 180
        x2 = int(1200 * np.cos(radian))
        y2 = int(1200 * np.sin(radian))

        cv2.line(img, (25, pos_line), (25 + x2, pos_line - y2), (255, 127, 0), 3)  # Membuat garis kemiringan

        for (x, y, w, h) in cars:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            center = center_object(x, y, w, h)
            detec.append((center, next_car_id))  # Menambahkan ID kendaraan
            cv2.circle(img, center, 4, (0, 0, 255), -1)

            if center[1] < (pos_line + offset) and center[1] > (pos_line - offset):
                car += 1
                cv2.line(img, (25, pos_line), (25 + x2, pos_line - y2), (0, 127, 255), 3)

                # Catat ID kendaraan yang melewati garis beserta waktu
                car_ids[next_car_id] = (center, ctime(time()))  # Pencatatan waktu

                # Cetak ID kendaraan dan waktu di terminal
                print(f"ID Kendaraan {next_car_id} terdeteksi pada {ctime(time())}")

            # Increment ID kendaraan untuk kendaraan berikutnya
            next_car_id += 1

        for car_id, (center, timestamp) in car_ids.items():
            cv2.putText(img, f"ID Kendaraan: {car_id}, Waktu Terdeteksi: {timestamp}", (center[0], center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.putText(img, "Kendaraan Lewat : " + str(car), (0, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 250), 3)
        cv2.imshow('video', img)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    # Menghitung total kendaraan
    total_kendaraan = len(car_ids)
    print(f"Total Kendaraan Terdeteksi: {total_kendaraan}")

    # Menentukan sesi berdasarkan waktu machine learning
    current_time = int(ctime(time()).split()[3].split(':')[0])
    sesi = "pagi" if 6 <= current_time < 10 else "siang" if 11 <= current_time < 14 else "sore"
    print(f"Sesi: {sesi}")

    # Menentukan status kendaraan berdasarkan jumlah kendaraan
    status_kendaraan = ""
    if total_kendaraan <= 30:
        status_kendaraan = "sepi"
    elif total_kendaraan <= 60:
        status_kendaraan = "renggang"
    else:
        status_kendaraan = "padat"
    print(f"Status Kendaraan: {status_kendaraan}")

    # Menentukan hari berdasarkan waktu machine learning
    current_time = ctime(time())
    current_time_struct = strptime(current_time, "%a %b %d %H:%M:%S %Y")
    hari = strftime("%A", current_time_struct)
    print(f"Hari: {hari}")

    # Menutup video setelah selesai
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Mencari video terbaru dalam folder "uploads"
    upload_dir = 'uploads'
    video_files = glob.glob(os.path.join(upload_dir, '*.mp4'))
    if video_files:
        video_files.sort(key=os.path.getctime, reverse=True)
        latest_video = video_files[0]
        detect_cars(latest_video)
    else:
        print("Tidak ada file video dalam folder 'uploads'.")
