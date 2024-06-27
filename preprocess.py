import os
import tensorflow as tf

# Define dataset paths
ANC_PATH = "/Users/vladpavlovich/Downloads/FaceDataSet/Original Images/Original Images"
POS_PATH = "data/positive"
NEG_PATH = "data/negative"

def preprocess(file_path):
    byte_img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(byte_img, channels=3)
    img = tf.image.resize(img, (100, 100))
    img = img / 255.0
    return img

def preprocess_twin(input_img, validation_img, label):
    return (preprocess(input_img), preprocess(validation_img), label)

def run():
    anchor = tf.data.Dataset.list_files(os.path.join(ANC_PATH, '*/*.jpg')).take(3000)
    positive = tf.data.Dataset.list_files(os.path.join(POS_PATH, '*.jpg')).take(3000)
    negative = tf.data.Dataset.list_files(os.path.join(NEG_PATH, '*.jpg')).take(3000)

    positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
    negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
    data = positives.concatenate(negatives)

    data = data.map(preprocess_twin)
    data = data.cache()
    data = data.shuffle(buffer_size=10000)

    train_data = data.take(int(0.7 * 6000))
    train_data = train_data.batch(16)
    train_data = train_data.prefetch(8)

    test_data = data.skip(int(0.7 * 6000))
    test_data = test_data.take(int(0.3 * 6000))
    test_data = test_data.batch(16)
    test_data = test_data.prefetch(8)

    return anchor, positive, negative, train_data, test_data

if __name__ == "__main__":
    anchor, positive, negative, train_data, test_data = run()
