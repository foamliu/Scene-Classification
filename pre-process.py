import zipfile
import os
import json
import os
import cv2 as cv
from console_progressbar import ProgressBar


def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


if __name__ == '__main__':
    # parameters
    ensure_folder('data')

    if not os.path.exists('data/ai_challenger_scene_train_20170904'):
        filename = 'data/ai_challenger_scene_train_20170904.zip'
        print('Extracting {}...'.format(filename))
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall('data')
            zip_ref.close()

    if not os.path.exists('data/ai_challenger_scene_validation_20170908'):
        filename = 'data/ai_challenger_scene_validation_20170908.zip'
        print('Extracting {}...'.format(filename))
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall('data')
            zip_ref.close()

    pb = ProgressBar(total=100, prefix='Save train data', suffix='', decimals=3, length=50, fill='=')
    if not os.path.exists('data/train'):
        os.makedirs('data/train')
        json_data = open('data/ai_challenger_scene_train_20170904/scene_train_annotations_20170904.json')
        data = json.load(json_data)
        num_samples = len(data)
        for i in range(num_samples):
            item = data[i]
            image_name = item['image_id']
            label_id = item['label_id']
            src_folder = 'data/ai_challenger_scene_train_20170904/scene_train_images_20170904'
            src_path = os.path.join(src_folder, image_name)
            dst_folder = 'data/train'
            label = "%02d" % (int(label_id),)
            dst_path = os.path.join(dst_folder, label)
            if not os.path.exists(dst_path):
                os.makedirs(dst_path)
            dst_path = os.path.join(dst_path, image_name)
            src_image = cv.imread(src_path)
            dst_image = cv.resize(src_image, (224, 224), cv.INTER_CUBIC)
            cv.imwrite(dst_path, dst_image)
            pb.print_progress_bar((i + 1) * 100 / num_samples)

    pb = ProgressBar(total=100, prefix='Save valid data', suffix='', decimals=3, length=50, fill='=')
    if not os.path.exists('data/valid'):
        os.makedirs('data/valid')
        json_data = open('data/ai_challenger_scene_validation_20170908/scene_validation_annotations_20170908.json')
        data = json.load(json_data)
        num_samples = len(data)
        for i in range(num_samples):
            item = data[i]
            image_name = item['image_id']
            label_id = item['label_id']
            src_folder = 'data/ai_challenger_scene_validation_20170908/scene_validation_images_20170908'
            src_path = os.path.join(src_folder, image_name)
            dst_folder = 'data/valid'
            label = "%02d" % (int(label_id),)
            dst_path = os.path.join(dst_folder, label)
            if not os.path.exists(dst_path):
                os.makedirs(dst_path)
            dst_path = os.path.join(dst_path, image_name)
            src_image = cv.imread(src_path)
            dst_image = cv.resize(src_image, (224, 224), cv.INTER_CUBIC)
            cv.imwrite(dst_path, dst_image)
            pb.print_progress_bar((i + 1) * 100 / num_samples)


