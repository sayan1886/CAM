from tensorflow import keras
from keras import Model
from keras.layers import Flatten, Dropout, Dense, Conv2D, MaxPooling2D
from utils.data import INPUT_HEIGHT, INPUT_WIDTH

MODEL_NAME = 'cnn_cancer_classifier'

class CNNModel(Model):
    def __init__(self, num_classes = 4):
        super(CNNModel, self).__init__()
        # Block 1
        self.conv1 = Conv2D(32, (3, 3), padding = 'same', activation = 'relu', name = 'CONV_1')
        self.pool1 = MaxPooling2D((2, 2), name = 'MAX_POOL_1')
        # Block 2
        self.conv2 = Conv2D(64, (3, 3), activation = 'relu', name = 'CONV_2')
        self.pool2 = MaxPooling2D((2, 2), name = 'MAX_POOL_2')
        # Block 3
        self.conv3 = Conv2D(128, (3, 3), activation = 'relu', name = 'CONV_3')
        self.pool3 = MaxPooling2D((2, 2), name = 'MAX_POOL_3')
        # Block 4
        self.conv4 = Conv2D(256, (3, 3), activation = 'relu', name = 'CONV_4')
        self.pool4 = MaxPooling2D((2, 2), name = 'MAX_POOL_4')
        # FC
        self.flatten = Flatten()
        self.dropout = Dropout(0.5)
        self.dense1 = Dense(512, activation = 'relu', name = 'DENSE_1')
        self.dense2 = Dense(256, activation = 'relu', name = 'DENSE_2')
        self.dense3 = Dense(num_classes, activation = 'softmax', name = 'DENSE_3')

        inputs = keras.layers.Input(shape = (INPUT_WIDTH, INPUT_HEIGHT, 3))
        x = inputs
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.pool4(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.dense1(x)
        x = self.dense2(x)
        output = self.dense3(x)

        self.model = keras.Model(inputs, output)
        self.model.name = "CNN_CUSTOM"
        self.inputs = inputs
        self.outputs = output

    # def build(self) :
    #     self.output = self.model.

    def call(self, input_tensor):
        return self.model(input_tensor)
    
    def summary(self):
        self.model.summary()