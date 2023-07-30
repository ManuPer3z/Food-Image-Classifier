# Food Image Classifier

A simple convolutional neural network to classify food images using TensorFlow and Keras.

## Setup

Ensure you have TensorFlow installed:

```bash
pip install tensorflow

Loading and Data Preparation:

Assuming you have the Food-101 dataset:

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Directories for data (adjust paths accordingly)
train_dir = 'path_to_food101_dataset/train'
validation_dir = 'path_to_food101_dataset/validation'

# Image preprocessing
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='categorical')

Building the Model:

from tensorflow.keras import layers
from tensorflow.keras import models

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(101, activation='softmax'))  # 101 classes in Food-101 dataset

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
Training:

history = model.fit(
    train_generator,
    steps_per_epoch=100,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50)

Evaluation and Prediction:

# Evaluation
test_loss, test_acc = model.evaluate(validation_generator, steps=50)
print('Test accuracy:', test_acc)

# Prediction
from tensorflow.keras.preprocessing import image
import numpy as np

img_path = 'path_to_test_image.jpg'
img = image.load_img(img_path, target_size=(150, 150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.

predictions = model.predict(img_tensor)
predicted_class = np.argmax(predictions[0])


