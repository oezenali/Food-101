import tensorflow as tf
import numpy as np
import os
import tensorflow_datasets as tfds
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt


# Veri ön işleme ve artırma fonksiyonu
def on_isleme(goruntu, etiket):
    goruntu = tf.image.resize(goruntu, [299, 299])
    goruntu = tf.image.random_flip_left_right(goruntu)
    goruntu = tf.image.random_brightness(goruntu, max_delta=0.2)
    goruntu = tf.image.random_contrast(goruntu, lower=0.75, upper=1.25)
    return goruntu / 255.0, etiket


# Model yapılandırmasını oluştur
def build_model_xception(num_classes):
    base_model = tf.keras.applications.Xception(include_top=False, input_shape=(299, 299, 3), weights='imagenet')
    base_model.trainable = True
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# Veri seti yükleme ve ön işleme
veri_dizini = "C:\\Users\\oezen\\OneDrive\\Masaüstü\\PROJE01\\"
test_data, info = tfds.load('food101', split='validation', as_supervised=True, with_info=True, data_dir=veri_dizini)
test_data = test_data.map(on_isleme).batch(16)

# Model dosyasının yolu
model_path = "C:\\Users\\oezen\\OneDrive\\Masaüstü\\PROJE01\\Xception\\food101_model_xception_finetuned_epoch_20.h5"

if os.path.exists(model_path):
    model = build_model_xception(info.features['label'].num_classes)
    model.load_weights(model_path)
    y_true = []
    y_pred = []
    for images, labels in test_data:
        predictions = model.predict(images)
        y_pred.extend(np.argmax(predictions, axis=1))
        y_true.extend(labels.numpy())

    # Metriklerin hesaplanması
    accuracy = np.mean(np.array(y_true) == np.array(y_pred))
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred)

    print(f"Processed Model - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

    # Karışıklık matrisini görselleştirme
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(info.features['label'].names))
    plt.xticks(tick_marks, info.features['label'].names, rotation=45, fontsize=2)  # Font boyutu
    plt.yticks(tick_marks, info.features['label'].names, fontsize=4)  # Font boyutu
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

else:
    print(f"Model dosyası bulunamadı: {model_path}")
