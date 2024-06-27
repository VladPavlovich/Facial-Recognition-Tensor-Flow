import tensorflow as tf
from tensorflow.keras.layers import Layer, Input, Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Model


# L1 Distance layer definition
class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__()

    def call(self, input_embedding, validation_embedding):
        # Convert inputs to tensors if they are not already
        input_embedding = tf.convert_to_tensor(input_embedding)
        validation_embedding = tf.convert_to_tensor(validation_embedding)
        return tf.math.abs(input_embedding - validation_embedding)


def make_embedding():
    inp = Input(shape=(100, 100, 3), name='input_image')
    c1 = Conv2D(64, (10, 10), activation='relu')(inp)
    m1 = MaxPooling2D(64, (2, 2), padding='same')(c1)
    c2 = Conv2D(128, (7, 7), activation='relu')(m1)
    m2 = MaxPooling2D(64, (2, 2), padding='same')(c2)
    c3 = Conv2D(128, (4, 4), activation='relu')(m2)
    m3 = MaxPooling2D(64, (2, 2), padding='same')(c3)
    c4 = Conv2D(256, (4, 4), activation='relu')(m3)
    f1 = Flatten()(c4)
    d1 = Dense(4096, activation='sigmoid')(f1)

    return Model(inputs=[inp], outputs=[d1], name='embedding')


def make_siamese_model():
    embedding = make_embedding()

    # Anchor and positive image inputs
    input_image = Input(name='input_img', shape=(100, 100, 3))
    validation_image = Input(name='validation_img', shape=(100, 100, 3))

    # Generate embeddings
    inp_embedding = embedding(input_image)
    val_embedding = embedding(validation_image)

    # Siamese Layer
    siamese_layer = L1Dist()
    distances = siamese_layer(inp_embedding, val_embedding)

    # Classification Layer
    classifier = Dense(1, activation='sigmoid')(distances)

    return Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')


# Example of calling the model
if __name__ == "__main__":
    model = make_siamese_model()
    model.summary()
