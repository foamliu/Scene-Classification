import argparse
import json
import os

import cv2 as cv
import keras.backend as K
import numpy as np
from keras.applications.inception_resnet_v2 import preprocess_input
from tqdm import tqdm

from config import img_width, img_height, best_model
from model import build_model

if __name__ == '__main__':
    # Parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--testsuite", help="name of test suite (e.g. test_a or test_b")
    args = vars(ap.parse_args())
    test_suite = args["testsuite"]

    model = build_model()
    model_weights_path = os.path.join('models', best_model)
    model.load_weights(model_weights_path)

    test_a = 'data/ai_challenger_scene_{}_20180103'.format(test_suite)
    image_folder = 'data/ai_challenger_scene_{0}_20180103/scene_{0}_images_20180103'.format(test_suite)
    annotations = 'data/ai_challenger_scene_{0}_20180103/scene_{0}_annotations_20180103.json'.format(test_suite)
    with open(annotations, 'r') as f:
        data = json.load(f)

    num_samples = len(data)
    print('num_samples: ' + str(num_samples))

    num_correct = 0
    for i in tqdm(range(num_samples)):
        image_id = data[i]['image_id']
        label_id = int(data[i]['label_id'])
        filename = os.path.join(image_folder, image_id)
        image = cv.imread(filename)
        image = cv.resize(image, (img_width, img_height), cv.INTER_CUBIC)
        rgb_img = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        rgb_img = np.expand_dims(rgb_img, 0).astype(np.float32)
        rgb_img = preprocess_input(rgb_img)
        preds = model.predict(rgb_img)
        top3 = np.argsort(preds)[0][::-1][:3]
        if label_id in top3:
            num_correct += 1

    print(num_correct / num_samples)
    K.clear_session()
