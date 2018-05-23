import keras.backend as K

from config import img_height, img_width
from resnet_50 import resnet50_model_old, resnet50_model_new


def migrate_model(new_model):
    old_model = resnet50_model_old(224, 224, 3)
    # print(old_model.summary())
    old_layers = [l for l in old_model.layers]
    new_layers = [l for l in new_model.layers]

    for i in range(2, len(old_layers) - 3):
        old_layer = old_layers[i]
        new_layer = new_layers[i]
        new_layer.set_weights(old_layer.get_weights())

    del old_model


if __name__ == '__main__':
    model = resnet50_model_new(img_height, img_width, 3, num_classes=20)
    migrate_model(model)
    print(model.summary())

    K.clear_session()
