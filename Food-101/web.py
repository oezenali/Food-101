from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
import tensorflow as tf
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

# Model yolu
model_path = "C:\\Users\\oezen\\OneDrive\\Masaüstü\\PROJE01\\EfficientNetV2\\food101_model_efficientnetv2_finetuned_epoch_25.h5"
model = load_model(model_path)

# Sınıf isimlerini dosyadan oku
class_path = "C:\\Users\\oezen\\OneDrive\\Masaüstü\\PROJE01\\classes.txt"
with open(class_path, 'r', encoding='utf-8') as file:
    class_names = [line.strip() for line in file.readlines()]

@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    if request.method == 'POST':
        image_file = request.files['file']
        if image_file:
            filename = secure_filename(image_file.filename)
            filepath = os.path.join('uploads', filename)
            image_file.save(filepath)

            # Görüntüyü model için hazırla
            image = tf.io.read_file(filepath)
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.image.resize(image, [320, 320])
            image = tf.expand_dims(image, 0)
            image = image / 255.0

            # Tahmini yap
            prediction = model.predict(image)
            predicted_class = tf.argmax(prediction, axis=1).numpy()[0]
            predicted_class_name = class_names[predicted_class]

            return render_template('index.html', prediction=predicted_class_name)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
