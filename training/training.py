from training.cnn import train_cnn, restore_cnn
from training.vgg16 import train_vgg16, restore_vgg16
from utils.data import get_merged_dataset, NUM_CATEGORIES

import os.path

def restore_model(modelName = "cvgg"):
    train_images_merged, train_labels_merged, test_images_merged, test_labels_merged = get_merged_dataset()
    train_images = train_images_merged
    train_labels = train_labels_merged
    if modelName == "cvgg":
        check_file = os.path.isfile("./vgg16_cancer_classifier.weights.h5")
        if check_file :
            model = restore_vgg16(NUM_CATEGORIES, False)
        else : 
            model = train_vgg16(train_images, train_labels, NUM_CATEGORIES)
    elif modelName == "cnn":
        check_file = os.path.isfile("./cnn_cancer_classifier.weights.h5")
        if check_file :
            model = restore_cnn(num_classes = NUM_CATEGORIES, test_model = False)
        else : 
            model = train_cnn(train_images, train_labels, NUM_CATEGORIES)
    else :
        model = None
    return model