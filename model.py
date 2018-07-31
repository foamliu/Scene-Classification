from keras import regularizers
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model

from config import num_classes


def build_model():
    l2_reg = regularizers.l2(2e-6)
    base_model = InceptionResNetV2(weights='imagenet', include_top=False)
    x = base_model.output
    for i in range(int(len(base_model.layers) * 0.73)):
        layer = base_model.layers[i]
        layer.trainable = False

    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.29)(x)
    x = Dense(1024, activation='relu', kernel_regularizer=l2_reg, activity_regularizer=l2_reg)(x)
    x = Dropout(0.25)(x)
    predictions = Dense(num_classes, activation='softmax', kernel_regularizer=l2_reg, activity_regularizer=l2_reg)(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model
