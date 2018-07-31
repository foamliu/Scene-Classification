# import the necessary packages
import csv
import os
import random

import cv2 as cv
import keras.backend as K
import numpy as np
from keras.applications.inception_resnet_v2 import preprocess_input

from config import best_model
from model import build_model
from utils import draw_str

if __name__ == '__main__':
    model = build_model()
    model_weights_path = os.path.join('models', best_model)
    model.load_weights(model_weights_path)

    with open('scene_classes.csv') as file:
        reader = csv.reader(file)
        scene_classes_list = list(reader)

    scene_classes_dict = dict()
    for item in scene_classes_list:
        scene_classes_dict[int(item[0])] = item[2]

    test_path = 'data/test_a/'
    test_images = [f for f in os.listdir(test_path) if
                   os.path.isfile(os.path.join(test_path, f)) and f.endswith('.jpg')]
    num_samples = 20
    samples = random.sample(test_images, num_samples)

    if not os.path.exists('images'):
        os.makedirs('images')

    for i in range(len(samples)):
        image_name = samples[i]
        filename = os.path.join(test_path, image_name)
        print('Start processing image: {}'.format(filename))
        image = cv.imread(filename)
        rgb_img = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        rgb_img = np.expand_dims(rgb_img, 0).astype(np.float32)
        rgb_img = preprocess_input(rgb_img)
        preds = model.predict(rgb_img)
        prob = np.max(preds)
        class_id = np.argmax(preds)
        print(scene_classes_dict[class_id])
        text = ('{}, prob: {:.4}'.format(scene_classes_dict[class_id], prob))
        draw_str(image, (5, 15), text)
        cv.imwrite('images/{}_out.png'.format(i), image)

    K.clear_session()
