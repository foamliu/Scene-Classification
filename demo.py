# import the necessary packages
import csv
import os
import random

import cv2 as cv
import keras.backend as K
import numpy as np

from resnet_50 import resnet50_model_new
from utils import draw_str

if __name__ == '__main__':
    img_width, img_height = 640, 640
    num_channels = 3
    num_classes = 80

    model = resnet50_model_new(img_rows=img_height, img_cols=img_width, color_type=num_channels,
                               num_classes=num_classes)
    model.load_weights('models/model.07-0.7091.hdf5')

    with open('scene_classes.csv') as file:
        reader = csv.reader(file)
        scene_classes_list = list(reader)

    scene_classes_dict = dict()
    for item in scene_classes_list:
        scene_classes_dict[int(item[0])] = item[2]

    test_path = 'data/ai_challenger_scene_test_a_20180103/scene_test_a_images_20180103/'
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
        image = cv.resize(image, (640, 640), cv.INTER_CUBIC)
        rgb_img = cv.cvtColor(image, cv.COLOR_BGR2RGB) / 255.
        rgb_img = np.expand_dims(rgb_img, 0)
        preds = model.predict(rgb_img)
        prob = np.max(preds)
        class_id = np.argmax(preds)
        print(scene_classes_dict[class_id])
        text = ('Predict: {}, prob: {}'.format(scene_classes_dict[class_id], prob))
        draw_str(image, (20, 20), text)
        cv.imwrite('images/{}_out.png'.format(i), image)

    K.clear_session()
