from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from train import detect_cars


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:nafijoko@localhost:3307/trafficgate' # Sesuaikan dengan konfigurasi database Anda
db = SQLAlchemy(app)
# Model untuk tabel Video 
class Video(db.Model):
    id_video = db.Column(db.Integer, primary_key=True, autoincrement=True)
    video = db.Column(db.String(255), nullable=False)
    waktu = db.Column(db.DateTime, default=datetime.utcnow)

# Model untuk tabel Kendaraan
class Kendaraan(db.Model):
    id_kendaraan = db.Column(db.Integer, primary_key=True, autoincrement=True)
    jumlah_kendaraan = db.Column(db.Integer)
    status_kendaraan = db.Column(db.String(20))
    sesi = db.Column(db.String(10))

# Model untuk tabel Analisis
class Analisis(db.Model):
    id_analisis = db.Column(db.Integer, primary_key=True, autoincrement=True)
    id_video = db.Column(db.Integer, db.ForeignKey('id_video'))
    id_kendaraan = db.Column(db.Integer, db.ForeignKey('id_kendaraan'))
    hari = db.Column(db.String(20))
    waktu = db.Column(db.DateTime, default=datetime.utcnow)
    jumlah_kendaraan = db.Column(db.Integer)
    sesi = db.Column(db.String(10))
    status_kendaraan = db.Column(db.String(20))

# Tentukan direktori penyimpanan file video
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Folder untuk hasil deteksi
DETECTED_FOLDER = 'detected'
if not os.path.exists(DETECTED_FOLDER):
    os.makedirs(DETECTED_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mkv', 'mov'}  # Hanya izinkan beberapa ekstensi file video
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file uploaded'}), 400
    
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if not allowed_file(video_file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    filename = secure_filename(video_file.filename)
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    video_file.save(video_path)
    
    # Panggil fungsi deteksi kendaraan dengan video_path sebagai argumen
    detect_cars(video_path)
    
    return jsonify({'message': 'Video uploaded successfully'}), 200  # Mengembalikan respons JSON yang mengindikasikan video berhasil diunggah

# Route untuk menyimpan data kendaraan dan analisis
@app.route('/save_data', methods=['POST'])
def save_data():
    data = request.json
    jumlah_kendaraan = data['jumlah_kendaraan']
    status_kendaraan = data['status_kendaraan']
    status_arus = data['status_arus']
    sesi = data['sesi']
    hari = data['hari']
    id_video = data['id_video']

    # Simpan data video ke database
    waktu = datetime.now()
    video_data = Video(video=filename, waktu=waktu)
    db.session.add(video_data)
    db.create_all()
    db.session.commit()

    # Simpan data kendaraan ke database
    kendaraan_data = Kendaraan(jumlah_kendaraan=jumlah_kendaraan, status_kendaraan=status_kendaraan, sesi=sesi)
    db.session.add(kendaraan_data)
    db.create_all()
    db.session.commit()

    # Simpan data analisis ke database
    id_kendaraan = kendaraan_data.id
    analisis_data = Analisis(id_video=id_video, id_kendaraan=id_kendaraan, hari=hari,jumlah_kendaraan=jumlah_kendaraan, status_arus=status_arus, sesi=sesi)
    db.session.add(analisis_data)
    db.create_all()
    db.session.commit()

    detected_path = os.path.join(DETECTED_FOLDER, f'detected_{video_data.id_video}.txt')
    with open(detected_path, 'w') as file:
        file.write(f'Jumlah Kendaraan: {jumlah_kendaraan}\n')
        file.write(f'Status Kendaraan: {status_kendaraan}\n')
        file.write(f'Sesi: {sesi}\n')
        file.write(f'Hari: {hari}\n')

    return jsonify({'message': 'Data saved successfully'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000)

