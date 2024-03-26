# import the necessary packages
import tensorflow as tf
from tensorflow import keras
from keras.applications import ResNet50
from keras.applications import VGG16
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.applications import imagenet_utils

import matplotlib.pyplot as plt

from cam.gradcam import GradCAM
from training.training import restore_model
from utils.data import CATEGORIES

from PIL import Image
from os.path import dirname, realpath

import numpy as np
import argparse
import imutils
import cv2
import os



# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
ap.add_argument("-m", "--model", type=str, default="vgg",
	choices=("vgg", "resnet", "cnn", "cvgg"),
	help="model to be used")
args = vars(ap.parse_args())

# initialize the model to be VGG16
Model = VGG16
# check to see if we are using ResNet
if args["model"] == "resnet" :
    Model = ResNet50

# load the pre-trained CNN from disk
print("[INFO] loading " + args["model"] + " model...")
if args["model"] == "cnn" or args["model"] == "cvgg" :
      model = restore_model(args["model"])
else:
     model = Model(weights = "imagenet", input_shape = (224, 224, 3), include_top=True)

model.summary()
print(model.inputs)
print(model.outputs)

if args["model"] == "resnet" or args["model"] == "vgg"  or args['model'] == '':
    # load the input image from disk (in Keras/TensorFlow format) and
    # preprocess it
    image = load_img(args["image"], target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    # load the original image from disk (in OpenCV format) and then
    # resize the image to its target dimensions
    orig = cv2.imread(args["image"])
    resized = cv2.resize(orig, (224, 224))

    # use the network to make predictions on the input image and find
    # the class label index with the largest corresponding probability
    preds = model.predict(image)
    result = np.argmax(preds[0])

    # decode the ImageNet predictions to obtain the human-readable label
    decoded = imagenet_utils.decode_predictions(preds)
    (imagenetID, label, prob) = decoded[0][0]
    output_image_label = label + ".jpg"
    label = "{}: {:.2f}%".format(label, prob * 100)
    print("[INFO] {}".format(label))

    # initialize our gradient class activation map and build the heatmap
    cam = GradCAM(model, result)
    heatmap = cam.compute_heatmap(image)

    # resize the resulting heatmap to the original input image dimensions
    # and then overlay heatmap on top of the image
    heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
    (heatmap, output) = cam.overlay_heatmap(heatmap, orig, alpha = 0.5)

    # draw the predicted label on the output image
    cv2.rectangle(output, (0, 0), (340, 40), (0, 0, 0), -1)
    cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
        0.8, (255, 255, 255), 2)

    # find contours of heatmap
    heatmap_max = np.max(heatmap)
    print("Max of heatmap: %.2f" % heatmap_max)

    boundary = heatmap_max * 0.3
    print("Cut-off value: %.2f" % boundary)

    gray = cv2.cvtColor(heatmap, cv2.COLOR_RGB2GRAY)
    bbox_heatmap = gray.copy()
    bbox_heatmap = np.where(bbox_heatmap <= boundary, 0, 255)

    bbox_img = output.copy()

    cnts = cv2.findContours(bbox_heatmap.astype('uint8'),
                            cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    offset = 10
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(bbox_img, (x+offset, y+offset),
                    (x+offset + w, y + h), (100,255,0), 3)

    # display the original image and resulting heatmap and output image
    # to our screen
    output = np.vstack([orig, heatmap, bbox_img])
    output = imutils.resize(output, height=700)
    cv2.imshow("Output", output)

    # saving image
    root_dir = dirname(realpath(__file__))
    output_dir = root_dir + "/images/predictions"
    print(output_dir)
    os.chdir(output_dir)
    print(os.listdir(output_dir)) 
    print(label)
    cv2.imwrite(output_image_label, output)
    cv2.waitKey(100)
    cv2.destroyAllWindows()
else:
    # https://stackoverflow.com/questions/66182884/how-to-implement-grad-cam-on-a-trained-network/66189774#66189774

    # load the original image from disk (in OpenCV format) and then
    # resize the image to its target dimensions
    orig = cv2.imread(args["image"])
    orig = cv2.resize(orig, (224, 224))
    image = orig.astype('float32') / 255
    
    image = np.expand_dims(image, axis=0)
    preds = model.predict(image) 
    result = np.argmax(preds[0])

    category_label = CATEGORIES[result]
    text = f'Found: {category_label}'
    print(text)
    plt.title("Prediction:" + category_label)
    plt.imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
    
    cam = GradCAM(model, result, customModel = True)
    heatmap = cam.compute_heatmap(image)
    heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
    (heatmap, output) = cam.overlay_heatmap(heatmap, orig, alpha = 0.5)
    
    print(heatmap.shape, image.shape)

    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(heatmap)
    ax[1].imshow(orig)
    ax[2].imshow(output)
    plt.show()





