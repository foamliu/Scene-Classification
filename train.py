import argparse

import keras
import tensorflow as tf
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import multi_gpu_model

from config import img_height, img_width, batch_size, patience, train_data, valid_data, \
    num_train_samples, num_valid_samples, num_epochs, verbose
from model import build_model
from utils import get_available_gpus, get_available_cpus

if __name__ == '__main__':
    # Parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--pretrained", help="path to save pretrained model files")
    args = vars(ap.parse_args())
    pretrained_path = args["pretrained"]

    # prepare data augmentation configuration
    train_data_gen = ImageDataGenerator(shear_range=0.2,
                                        rotation_range=20.,
                                        width_shift_range=0.3,
                                        height_shift_range=0.3,
                                        zoom_range=0.2,
                                        horizontal_flip=True,
                                        preprocessing_function=preprocess_input)
    valid_data_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

    # generators
    train_generator = train_data_gen.flow_from_directory(train_data, (img_width, img_height), batch_size=batch_size,
                                                         class_mode='categorical', shuffle=True)
    valid_generator = valid_data_gen.flow_from_directory(valid_data, (img_width, img_height), batch_size=batch_size,
                                                         class_mode='categorical', shuffle=True)


    class MyCbk(keras.callbacks.Callback):
        def __init__(self, model):
            keras.callbacks.Callback.__init__(self)
            self.model_to_save = model

        def on_epoch_end(self, epoch, logs=None):
            fmt = 'models/model.%02d-%.4f.hdf5'
            self.model_to_save.save(fmt % (epoch, logs['val_acc']))


    # Callbacks
    tensor_board = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
    # early_stop = EarlyStopping('val_acc', patience=patience)
    # reduce_lr = ReduceLROnPlateau('val_acc', factor=0.5, patience=int(patience / 4), verbose=1)
    trained_models_path = 'models/model'
    model_names = trained_models_path + '.{epoch:02d}-{val_loss:.4f}.hdf5'
    model_checkpoint = ModelCheckpoint(model_names, monitor='val_acc', verbose=1, save_best_only=True)

    num_gpu = len(get_available_gpus())
    if num_gpu >= 2:
        with tf.device("/cpu:0"):
            model = build_model()
            if pretrained_path is not None:
                model.load_weights(pretrained_path)

        new_model = multi_gpu_model(model, gpus=num_gpu)
        # rewrite the callback: saving through the original model and not the multi-gpu model.
        model_checkpoint = MyCbk(model)
    else:
        new_model = build_model()
        if pretrained_path is not None:
            new_model.load_weights(pretrained_path)

    adam = keras.optimizers.Adam(lr=1e-6)
    new_model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    callbacks = [tensor_board, model_checkpoint]

    # fine tune the model
    new_model.fit_generator(
        train_generator,
        steps_per_epoch=num_train_samples / batch_size,
        validation_data=valid_generator,
        validation_steps=num_valid_samples / batch_size,
        shuffle=True,
        epochs=num_epochs,
        callbacks=callbacks,
        verbose=verbose,
        use_multiprocessing=True,
        workers=int(get_available_cpus() * 0.80)
    )
