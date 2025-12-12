import tensorflow as tf

def load_image_dataset(
        directory,
        img_size=(224, 224),
        batch_size=32,
        shuffle=True
    ):
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        directory,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=shuffle
    )
    return dataset.prefetch(buffer_size=tf.data.AUTOTUNE)


def normalize_dataset(dataset):
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    return dataset.map(lambda x, y: (normalization_layer(x), y))
