import tensorflow as tf
from tensorflow.keras import layers, models

def LeNet(num_classes=10):
    model = models.Sequential([
        layers.Conv2D(6, (5, 5), activation='relu', input_shape=(32, 32, 1)),
        layers.AveragePooling2D(),
        
        layers.Conv2D(16, (5, 5), activation='relu'),
        layers.AveragePooling2D(),
        
        layers.Flatten(),
        layers.Dense(120, activation='relu'),
        layers.Dense(84, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model
