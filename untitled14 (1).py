# -*- coding: utf-8 -*-
"""Untitled14.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1lSc-cgrQQ2d1V6f43tMgmC4Chr7hypwB
"""

import requests

url = "https://drive.google.com/uc?id=1mJQGU90goT8sHdn5Yo_6_xxR8QmICTj_&export=download"
response = requests.get(url)
with open("output_file.ext", "wb") as f:
    f.write(response.content)
    
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Fungsi untuk memuat model (gantilah 'model_path' dengan path model Anda)
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('VGG16Terbaru.h5')
    return model

# Memuat model
model = load_model()

# Fungsi untuk memprediksi gambar menggunakan model CNN
def predict(image, model):
    # Mengubah ukuran gambar sesuai dengan input model
    image = image.resize((224, 224))  # Sesuaikan dengan ukuran input model Anda
    # Mengubah gambar menjadi array numpy dan menormalkan
    image = np.array(image) / 255.0
    # Menambahkan dimensi batch
    image = np.expand_dims(image, axis=0)
    # Membuat prediksi
    predictions = model.predict(image)
    return predictions

# Judul aplikasi
st.title("Alat Deteksi Penyakit Mata dengan CNN")

# Mengunggah gambar
uploaded_file = st.file_uploader("Unggah gambar untuk deteksi", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Menampilkan gambar yang diunggah
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar yang diunggah', use_column_width=True)

    # Membuat prediksi jika model sudah dimuat
    if model is not None:
        predictions = predict(image, model)

        # Menampilkan hasil prediksi
        st.write("Hasil Prediksi:", predictions)

        # Mendapatkan indeks kelas yang diprediksi
        predicted_class_index = np.argmax(predictions[0])

        # Menampilkan hasil prediksi dengan label
        if predicted_class_index == 0:
            st.write('Kelas Prediksi: armd')
        elif predicted_class_index == 1:
            st.write('Kelas Prediksi: cataract')
        elif predicted_class_index == 2:
            st.write('Kelas Prediksi: diabetic_retinopathy')
        elif predicted_class_index == 3:
            st.write('Kelas Prediksi: glaucoma')
        elif predicted_class_index == 4:
            st.write('Kelas Prediksi: normal')
        else:
            st.write('Kelas tidak dikenali')
    else:
        st.write("Mohon tunggu, model sedang dimuat...")
