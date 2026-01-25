import tensorflow as tf
from tensorflow.keras import layers, models

def MobileNetV2(num_classes=1000):
    base = tf.keras.applications.MobileNetV2(
        input_shape=(224,224,3),
        include_top=False,
        weights=None
    )

    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return models.Model(base.input, outputs)
