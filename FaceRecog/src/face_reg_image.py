from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import argparse
import facenet
import align.detect_face
import numpy as np
import cv2
import pickle
import collections
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', help='Path of the image you want to test on.')
    args = parser.parse_args()

    MINSIZE = 20
    THRESHOLD = [0.6, 0.7, 0.7]
    FACTOR = 0.709
    IMAGE_SIZE = 182
    INPUT_IMAGE_SIZE = 160
    CLASSIFIER_PATH = 'Models/facemodel.pkl'
    FACENET_MODEL_PATH = 'Models/20180402-114759.pb'

    # Load The Custom Classifier
    with open(CLASSIFIER_PATH, 'rb') as file:
        model, class_names = pickle.load(file)
    print("Custom Classifier, Successfully loaded")

    with tf.compat.v1.Graph().as_default():

        # Set GPU options if available
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

        with sess.as_default():

            # Load the model
            print('Loading feature extraction model')
            facenet.load_model(FACENET_MODEL_PATH)

            # Get input and output tensors
            images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, "src/align")

            image = cv2.imread(args.image_path)
            frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert image to RGB format
            bounding_boxes, _ = align.detect_face.detect_face(frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)

            faces_found = bounding_boxes.shape[0]
            try:
                for i in range(faces_found):
                    det = bounding_boxes[i, 0:4]
                    bb = np.zeros((4,), dtype=np.int32)
                    bb[0] = det[0]
                    bb[1] = det[1]
                    bb[2] = det[2]
                    bb[3] = det[3]

                    cropped = frame[bb[1]:bb[3], bb[0]:bb[2], :]
                    scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
                    scaled = facenet.prewhiten(scaled)
                    scaled_reshape = scaled.reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)
                    feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
                    emb_array = sess.run(embeddings, feed_dict=feed_dict)
                    predictions = model.predict_proba(emb_array)
                    best_class_indices = np.argmax(predictions, axis=1)
                    best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                    best_name = class_names[best_class_indices[0]]

                    print("Name: {}, Probability: {}".format(best_name, best_class_probabilities))

                    if best_class_probabilities > 0.6:
                        cv2.rectangle(image, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 2)
                        text_x = bb[0]
                        text_y = bb[3] + 20

                        name = class_names[best_class_indices[0]]
                        cv2.putText(image, name, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                    1, (255, 255, 255), thickness=1, lineType=2)
                        cv2.putText(image, str(round(best_class_probabilities[0], 3)), (text_x, text_y + 17),
                                    cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                    1, (255, 255, 255), thickness=1, lineType=2)
            except:
                pass
            input_file_name = os.path.splitext(os.path.basename(args.image_path))[0]
            output_path = 'src/TestImage/After/{}.png'.format(input_file_name)
            cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

