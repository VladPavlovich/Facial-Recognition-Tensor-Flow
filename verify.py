import os
import numpy as np
import cv2
import tensorflow as tf
from models.siamese_model import L1Dist


def preprocess(file_path):
    byte_img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(byte_img)
    img = tf.image.resize(img, (100, 100))
    img = img / 255.0
    return img


def recognize_face(model, input_img_path, anchor_dir, known_face_names, detection_threshold=0.8):
    results = []
    for person_name in known_face_names:
        person_dir = os.path.join(anchor_dir, person_name)
        anchor_images = [os.path.join(person_dir, img) for img in os.listdir(person_dir) if img.endswith('.jpg')]

        if not anchor_images:
            print(f"No anchor images found for {person_name} in {person_dir}")
            continue

        input_img = preprocess(input_img_path)
        person_results = []
        for anchor_img_path in anchor_images:
            anchor_img = preprocess(anchor_img_path)
            result = model.predict([np.expand_dims(input_img, axis=0), np.expand_dims(anchor_img, axis=0)])
            person_results.append(result[0][0])

        best_result = max(person_results)
        results.append((best_result, person_name))

    if results:
        best_match_score, best_match_name = max(results)
        recognized_person = best_match_name if best_match_score > detection_threshold else "Unknown"
        print(f"Best match score: {best_match_score}, Detection threshold: {detection_threshold}")
        return recognized_person
    else:
        print("No results obtained from model predictions.")
        return "Unknown"


def run():
    model_path = '/Users/vladpavlovich/PycharmProjects/FacialRecTensorFlow/siamesemodelv2.h5'
    print(f"Loading model from {model_path}...")
    siamese_model = tf.keras.models.load_model(model_path, custom_objects={'L1Dist': L1Dist,
                                                                           'BinaryCrossentropy': tf.losses.BinaryCrossentropy})
    print(f"Model loaded from {model_path}")

    anchor_dir = '/Users/vladpavlovich/PycharmProjects/FacialRecTensorFlow/anchorImages'
    known_face_names = [name for name in os.listdir(anchor_dir) if os.path.isdir(os.path.join(anchor_dir, name))]

    if not os.path.exists('application_data/input_image'):
        os.makedirs('application_data/input_image')

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        cv2.imshow('Facial Recognition', frame)
        key = cv2.waitKey(10) & 0xFF
        if key == ord('v'):
            print("Verification key pressed.")
            input_image_path = os.path.join('application_data', 'input_image', 'input_image.jpg')
            cv2.imwrite(input_image_path, frame)
            print(f"Saved input image to {input_image_path}")

            recognized_person = recognize_face(siamese_model, input_image_path, anchor_dir, known_face_names)
            print(f"Recognized Person: {recognized_person}")

        if key == ord('q'):
            print("Quit key pressed.")
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
