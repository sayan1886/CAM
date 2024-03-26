from keras.models import Model

from utils.data import INPUT_WIDTH, INPUT_HEIGHT

import ssl

from model.vgg16 import VGG16Model
from training.testing import test
from utils.callback import StopByAccuracyCallback
from utils.data import get_generators

ssl._create_default_https_context = ssl._create_unverified_context

MODEL_NAME = 'vgg16_cancer_classifier.weights.h5'


def train_vgg16(train_images, train_labels, num_classes = 4) -> Model:
    vgg16 = VGG16Model(INPUT_WIDTH, INPUT_HEIGHT, num_classes)
    model = vgg16.model

    train_generator, validation_generator = get_generators(train_images, train_labels)

    model.fit(
        train_generator,
        steps_per_epoch=40,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=10,
        callbacks=[StopByAccuracyCallback()]
    )
    print(f"Save weights into {MODEL_NAME}")
    model.save_weights(MODEL_NAME)

    test(model)

    return model


def restore_vgg16(num_classes = 4, test_model=False) -> Model:
    vgg16 = VGG16Model(INPUT_WIDTH, INPUT_HEIGHT, num_classes)
    model = vgg16.model

    model.load_weights(MODEL_NAME)

    if test_model:
        test(model)

    return model