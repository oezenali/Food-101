import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds

# Veri seti
veri_dizini = "C:\\Users\\oezen\\OneDrive\\Masaüstü\\PROJE01\\"
(egitim_verisi, test_verisi), veri_bilgisi = tfds.load(
    'food101',
    split=['train', 'validation'],
    as_supervised=True,
    with_info=True,
    shuffle_files=True,
    data_dir=veri_dizini,
    download=True
)

# Veri ön işleme ve artırma fonksiyonu
def on_isleme(goruntu, etiket):
    goruntu = tf.image.resize(goruntu, [380, 380])  # EfficientNetB4 için boyut
    goruntu = tf.image.random_flip_left_right(goruntu)
    goruntu = tf.image.random_brightness(goruntu, max_delta=0.2)
    goruntu = tf.image.random_contrast(goruntu, lower=0.75, upper=1.25)
    return goruntu / 255.0, etiket

# Veri setlerini işleme ve batch'lere ayırma
egitim_verisi = egitim_verisi.map(on_isleme).batch(8).prefetch(tf.data.experimental.AUTOTUNE)
test_verisi = test_verisi.map(on_isleme).batch(8).prefetch(tf.data.experimental.AUTOTUNE)

# EfficientNetB4 modeli
temel_model = tf.keras.applications.EfficientNetB4(include_top=False, input_shape=(380, 380, 3), weights='imagenet')
temel_model.trainable = True

# Yeni katmanlar ekleyerek modeli özelleştirme
model = tf.keras.Sequential([
    temel_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.6),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.6),
    tf.keras.layers.Dense(veri_bilgisi.features['label'].num_classes, activation='softmax')
])

# Modeli derle
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Her epoch sonunda modeli kaydetme
kontrol_noktasi = tf.keras.callbacks.ModelCheckpoint(
    'C:/Users/oezen/OneDrive/Masaüstü/PROJE01/EfficientNetB4/food101_model_efficientnetb4_finetuned_epoch_{epoch:02d}.h5',
    save_weights_only=True,
    verbose=1
)

# Model eğitimi ve erken durdurma
erken_durdurma = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
gecmis = model.fit(
    egitim_verisi,
    epochs=50,
    validation_data=test_verisi,
    callbacks=[kontrol_noktasi, erken_durdurma]
)

# Eğitim ve doğrulama sonuçlarını görselleştirme
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(gecmis.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(gecmis.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.title('Model Doğruluğu')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(gecmis.history['loss'], label='Eğitim Kaybı')
plt.plot(gecmis.history['val_loss'], label='Doğrulama Kaybı')
plt.title('Model Kaybı')
plt.legend()
plt.show()

# Eğitim sonrası modeli değerlendirme
test_sonuc = model.evaluate(test_verisi)
print("Test Seti - Kayıp (Loss):", test_sonuc[0])
print("Test Seti - Doğruluk (Accuracy):", test_sonuc[1])
