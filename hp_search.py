from __future__ import print_function

import os
from math import log

import keras
from hyperas import optim
from hyperas.distributions import loguniform
from hyperas.distributions import uniform
from hyperopt import Trials, STATUS_OK, tpe
from keras import regularizers
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.layers import GlobalAveragePooling2D
from keras.layers.core import Dense, Dropout
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator

from config import img_width, img_height, num_classes, batch_size, train_data, valid_data, num_train_samples, \
    num_valid_samples, best_model


def data():
    train_datagen = ImageDataGenerator(shear_range=0.2,
                                       rotation_range=20.,
                                       width_shift_range=0.3,
                                       height_shift_range=0.3,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       preprocessing_function=preprocess_input)
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_generator = train_datagen.flow_from_directory(train_data, (img_width, img_height), batch_size=batch_size,
                                                        class_mode='categorical', shuffle=True)
    validation_generator = test_datagen.flow_from_directory(valid_data, (img_width, img_height), batch_size=batch_size,
                                                            class_mode='categorical', shuffle=True)

    return train_generator, validation_generator


def create_model(train_generator, validation_generator):
    l2_reg = regularizers.l2({{loguniform(log(1e-6), log(1e-2))}})
    base_model = InceptionResNetV2(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout({{uniform(0, 1)}})(x)
    x = Dense(1024, activation='relu', kernel_regularizer=l2_reg, activity_regularizer=l2_reg)(x)
    x = Dropout({{uniform(0, 1)}})(x)
    predictions = Dense(num_classes, activation='softmax', kernel_regularizer=l2_reg, activity_regularizer=l2_reg)(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    model_weights_path = os.path.join('models', best_model)
    model.load_weights(model_weights_path)

    for i in range(int(len(base_model.layers) * {{uniform(0, 1)}})):
        layer = base_model.layers[i]
        layer.trainable = False

    adam = keras.optimizers.Adam(lr={{loguniform(log(1e-6), log(1e-3))}})
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam)

    # print(model.summary())

    model.fit_generator(
        train_generator,
        steps_per_epoch=num_train_samples // batch_size,
        validation_data=validation_generator,
        validation_steps=num_valid_samples // batch_size)

    score, acc = model.evaluate_generator(validation_generator)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':
    train_generator, validation_generator = data()
    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=15,
                                          trials=Trials())

    print("Evalutation of best performing model:")
    print(best_model.evaluate_generator(validation_generator))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
