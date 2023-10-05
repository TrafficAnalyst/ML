from flask import Flask, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np

# Load model
model = tf.keras.models.load_model('model.h5')

# Initialize Flask app
app = Flask(_name_)

# Define API endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
   # Menerima gambar dari permintaan POST

    file = request.files['image']
    img = Image.load_img(file, target_size=(150, 150))
    
    # Mengubah gambar menjadi array numpy
    img_array = Image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalisasi

    result = train.predict(img_array)
    
    # Memprediksi kelas gambar
    predictions = model.predict(img_array)
    class_names = ['car', 'motorcycle', 'bycycle']  # Ganti dengan kelas yang sesuai
    predicted_class = class_names[np.argmax(predictions)]
    
    # Mengembalikan hasil prediksi dalam format JSON
    result = {'prediction': predicted_class}
    return jsonify(result)

# Run the Flask app
if _name_ == '_main_':
    app.run()