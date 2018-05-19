import keras.backend as K

from resnet_152 import resnet152_model

img_cols, img_rows = 320, 320
num_channels = 3
num_classes = 80


def migrate_model(new_model):
    old_model = resnet152_model(224, 224, num_channels, num_classes)
    # print(old_model.summary())
    old_layers = [l for l in old_model.layers]
    new_layers = [l for l in new_model.layers]

    for i in range(2, len(old_layers)):
        old_layer = old_layers[i]
        new_layer = new_layers[i]
        new_layer.set_weights(old_layer.get_weights())

    del old_model


if __name__ == '__main__':
    model = resnet152_model(img_rows, img_cols, num_channels, num_classes)
    migrate_model(model)
    print(model.summary())
    model.save_weights('models/model_weights.h5')

    K.clear_session()
