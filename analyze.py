import argparse
import json
import os

import cv2 as cv
import numpy as np
from console_progressbar import ProgressBar

from config import img_width, img_height, num_channels, num_classes
from densenet121 import densenet121_model

if __name__ == '__main__':
    # Parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--testsuite", help="name of test suite (e.g. test_a or test_b")
    args = vars(ap.parse_args())
    test_suite = args["testsuite"]

    model = densenet121_model(img_rows=img_height, img_cols=img_width, color_type=num_channels,
                              num_classes=num_classes)
    model.load_weights('models/model.85-0.7657.hdf5')

    test_a = 'data/ai_challenger_scene_{}_20180103'.format(test_suite)
    image_folder = 'data/ai_challenger_scene_{0}_20180103/scene_{0}_images_20180103'.format(test_suite)
    annotations = 'data/ai_challenger_scene_{0}_20180103/scene_{0}_annotations_20180103.json'.format(test_suite)
    with open(annotations, 'r') as f:
        data = json.load(f)

    num_samples = len(data)
    pb = ProgressBar(total=num_samples, prefix='Processing images', suffix='', decimals=3, length=50, fill='=')
    num_correct = 0
    for i in range(num_samples):
        image_id = data[i]['image_id']
        label_id = int(data[i]['label_id'])
        filename = os.path.join(image_folder, image_id)
        image = cv.imread(filename)
        image = cv.resize(image, (img_width, img_height), cv.INTER_CUBIC)
        rgb_img = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        rgb_img = np.expand_dims(rgb_img, 0)
        preds = model.predict(rgb_img)
        top3 = np.argsort(preds)[0][::-1][:3]
        if label_id in top3:
            num_correct += 1
        pb.print_progress_bar(i + 1)

    print(num_correct / num_samples)
