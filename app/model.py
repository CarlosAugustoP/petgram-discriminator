import tensorflow as tf
from keras import layers, models
import tensorflow_datasets as tfds
import os

MODEL_PATH = "cats_dogs_model.h5"

def create_and_train_model():
    dataset, info = tfds.load('cats_vs_dogs', with_info=True, as_supervised=True)
    
    train_dataset = dataset['train'].take(20000) 
    test_dataset = dataset['train'].skip(20000)  

    def preprocess(image, label):
        image = tf.image.resize(image, (128, 128))
        image = image / 255.0
        return image, label

    train_dataset = train_dataset.map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)
    
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])

    # Treinar e salvar modelo
    model.fit(train_dataset, epochs=10, validation_data=test_dataset)
    model.save(MODEL_PATH)
    
    return model

def load_model():
    return tf.keras.models.load_model(MODEL_PATH)