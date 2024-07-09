# Food101 Image Classification Project

This project aims to classify food images using the Food101 dataset, which consists of 101 food classes and a total of 101,000 images. Various deep learning models were employed to achieve high accuracy rates.

## Dataset

- **Food101**: A dataset containing 101 different food classes, with a total of 101,000 images. Each class has 750 training and 250 test images.

![image](https://github.com/oezenali/Food-101/assets/65864130/9911921e-c574-4720-9a8e-27ffc8962401)

## Models Used

1. **EfficientNetV2**
2. **MobileNetV2**
3. **Xception**
4. **EfficientNetB0**
5. **EfficientNetB4**

## Model Performance Evaluation

Models were evaluated using metrics such as accuracy, precision, recall, and F1 score.

## Methods

### Data Preprocessing

Images were resized and normalized to fit the input dimensions of the models.

### Data Augmentation

Various data augmentation techniques were used to increase the diversity of the training data.

### Model Training

Models were initialized with pretrained ImageNet weights and fine-tuned on the Food101 dataset.

## Results

- **EfficientNetV2**: Emerged as the best-performing model.
- **MobileNetV2 and Xception**: Lightweight models suitable for mobile devices and low-power hardware.
- **EfficientNetB0 and EfficientNetB4**: Successfully handled complex classification tasks with deeper structures.

## Usage

### Requirements

- Python 3.x
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Scikit-Learn




### Training

```python
# Creating a new model based on EfficientNetV2
base_model = tf.keras.applications.EfficientNetV2S(include_top=False, input_shape=(320, 320, 3), weights='imagenet')
base_model.trainable = True

# Customizing the model by adding new layers
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(101, activation='softmax')
])

# Compiling the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Training the model
history = model.fit(train_data, epochs=50, validation_data=test_data, callbacks=[checkpoint_callback, early_stopping])
```
![image](https://github.com/oezenali/Food-101/assets/65864130/d4347b65-d6ef-4098-8407-a2e67145d4c4)

### Evaluation

```python
# Evaluating the model
results = model.evaluate(test_data)
print(f"Test Loss: {results[0]}")
print(f"Test Accuracy: {results[1]}")
```

![image](https://github.com/oezenali/Food-101/assets/65864130/d6d0c592-85c9-4d77-8fbe-24da41902407)

![image](https://github.com/oezenali/Food-101/assets/65864130/300c7fdf-4b18-4a70-8e67-7a717c296acf)

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---
