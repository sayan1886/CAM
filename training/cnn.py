import numpy as np
from keras import Model
from keras.callbacks import EarlyStopping

from model.cnn import CNNModel
from training.testing import test
from utils.callback import StopByAccuracyCallback
from utils.data import get_generators, INPUT_HEIGHT, INPUT_WIDTH

MODEL_NAME = 'cnn_cancer_classifier.weights.h5'


def train_cnn(train_images, train_labels, num_classes = 4) -> Model:
    model = CNNModel(num_classes)
    train_generator, validation_generator = get_generators(train_images, train_labels)
    # model.build((None, INPUT_WIDTH, INPUT_HEIGHT, 3))
    model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

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


def restore_cnn(num_classes = 4, test_model = False) -> Model:
    model = CNNModel(num_classes)
    model.load_weights(MODEL_NAME)
    # train_generator, validation_generator = get_generators(train_images, train_labels)
    # model.build((None, INPUT_WIDTH, INPUT_HEIGHT, 3))
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    if test_model:
        test(model)

    return model