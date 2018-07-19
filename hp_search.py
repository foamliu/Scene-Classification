from __future__ import print_function

from hyperas import optim
from hyperas.distributions import choice, uniform
from hyperopt import Trials, STATUS_OK, tpe
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.layers import GlobalAveragePooling2D
from keras.layers.core import Dense, Dropout
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator

from config import img_width, img_height, num_classes, batch_size


def data():
    train_data_dir = 'column_data/train'
    validation_data_dir = 'column_data/validation'

    test_datagen = ImageDataGenerator(
        rescale=None)

    train_datagen = ImageDataGenerator(
        rescale=None,
        shear_range=0.2,
        horizontal_flip=False)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

    return train_generator, validation_generator


def create_model(x_train, y_train, x_test, y_test):
    base_model = InceptionResNetV2(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout({{uniform(0, 1)}})(x)
    x = Dense({{choice([512, 1024, 1048])}}, activation='relu')(x)
    x = Dropout({{uniform(0, 1)}})(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
                  optimizer={{choice(['rmsprop', 'adam', 'sgd', 'nadam'])}})

    model.fit(x_train, y_train,
              batch_size={{choice([32, 64, 128])}},
              epochs=1,
              verbose=2,
              validation_data=(x_test, y_test))
    score, acc = model.evaluate(x_test, y_test, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=5,
                                          trials=Trials())
    X_train, Y_train, X_test, Y_test = data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
