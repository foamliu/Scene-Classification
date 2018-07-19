from __future__ import print_function

from hyperas import optim
from hyperas.distributions import choice, uniform
from hyperopt import Trials, STATUS_OK, tpe
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.layers import GlobalAveragePooling2D
from keras.layers.core import Dense, Dropout
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator

from config import img_width, img_height, num_classes, batch_size, train_data, valid_data, num_train_samples, \
    num_valid_samples


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
    base_model = InceptionResNetV2(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout({{uniform(0, 1)}})(x)
    x = Dense({{choice([512, 1024, 2048])}}, activation='relu')(x)
    x = Dropout({{uniform(0, 1)}})(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    if {{choice(['three', 'four'])}} == 'four':
        for layer in base_model.layers:
            layer.trainable = False

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
                  optimizer={{choice(['rmsprop', 'adam', 'sgd', 'nadam'])}})

    model.fit_generator(
        train_generator,
        steps_per_epoch=num_train_samples / batch_size,
        validation_data=validation_generator,
        validation_steps=num_valid_samples / batch_size)

    score, acc = model.evaluate(validation_generator, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':
    train_generator, validation_generator = data()
    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=20,
                                          trials=Trials())

    print("Evalutation of best performing model:")
    print(best_model.evaluate(validation_generator))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
