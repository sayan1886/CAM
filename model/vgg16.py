from keras.layers import Dense, Dropout, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


class VGG16Model:
    def __init__(self, input_width, input_height, num_classes = 4):
        self.input_shape = (input_width, input_height, 3)
        self.classes = num_classes
        self.model = None
        self.build_model()

    def build_model(self):
        vgg16 = VGG16(weights='imagenet', input_shape=self.input_shape, classes=self.classes, include_top = False)

        for layer in vgg16.layers:
            layer.trainable = False

        x = Flatten()(vgg16.output)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(self.classes, activation='softmax')(x)

        self.model = Model(inputs=vgg16.input, outputs=predictions)
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])