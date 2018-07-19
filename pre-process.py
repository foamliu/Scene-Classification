import json
import os
import zipfile

import cv2 as cv
from tqdm import tqdm

from config import img_height, img_width


def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def extract(usage, package, image_path, json_path):
    filename = 'data/{}.zip'.format(package)
    print('Extracting {}...'.format(filename))
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall('data')

    if not os.path.exists('data/{}'.format(usage)):
        os.makedirs('data/{}'.format(usage))
    with open('data/{}/{}'.format(package, json_path)) as json_data:
        data = json.load(json_data)
    num_samples = len(data)
    print("num_samples: " + str(num_samples))
    for i in tqdm(range(num_samples)):
        item = data[i]
        image_name = item['image_id']
        label_id = item['label_id']
        src_folder = 'data/{}/{}'.format(package, image_path)
        src_path = os.path.join(src_folder, image_name)
        dst_folder = 'data/{}'.format(usage)
        label = "%02d" % (int(label_id),)
        dst_path = os.path.join(dst_folder, label)
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)
        dst_path = os.path.join(dst_path, image_name)
        src_image = cv.imread(src_path)
        dst_image = cv.resize(src_image, (img_height, img_width), cv.INTER_CUBIC)
        cv.imwrite(dst_path, dst_image)


def extract_test(usage, package, image_path, json_path):
    filename = 'data/{}.zip'.format(package)
    print('Extracting {}...'.format(filename))
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall('data')

    if not os.path.exists('data/{}'.format(usage)):
        os.makedirs('data/{}'.format(usage))
    with open('data/{}/{}'.format(package, json_path)) as json_data:
        data = json.load(json_data)
    num_samples = len(data)
    print("num_samples: " + str(num_samples))
    label_dict = dict()
    for i in tqdm(range(num_samples)):
        item = data[i]
        image_name = item['image_id']
        label_id = item['label_id']
        src_folder = 'data/{}/{}'.format(package, image_path)
        src_path = os.path.join(src_folder, image_name)
        dst_folder = 'data/{}'.format(usage)
        label = "%02d" % (int(label_id),)
        label_dict[image_name] = label
        dst_path = os.path.join(dst_folder, image_name)
        src_image = cv.imread(src_path)
        dst_image = cv.resize(src_image, (img_height, img_width), cv.INTER_CUBIC)
        cv.imwrite(dst_path, dst_image)
    with open('label_dict.txt', 'w') as outfile:
        json.dump(label_dict, outfile, indent=4, sort_keys=True)


if __name__ == '__main__':
    # parameters
    ensure_folder('data')

    extract('train', 'ai_challenger_scene_train_20170904', 'scene_train_images_20170904',
            'scene_train_annotations_20170904.json')

    extract('valid', 'ai_challenger_scene_validation_20170908', 'scene_validation_images_20170908',
            'scene_validation_annotations_20170908.json')

    extract_test('test_a', 'ai_challenger_scene_test_a_20180103', 'scene_test_a_images_20180103',
                 'scene_test_a_annotations_20180103.json')

    extract_test('test_b', 'ai_challenger_scene_test_b_20180103', 'scene_test_b_images_20180103',
                 'scene_test_b_annotations_20180103.json')
