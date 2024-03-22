import matplotlib.pyplot as plt
import seaborn as sns
import math
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout 
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

from sklearn.metrics import classification_report,confusion_matrix

import tensorflow as tf

import cv2
import os

import numpy as np

import argparse

import imghdr

import shutil

def get_files_from_folder(path):
    files = os.listdir(path)
    return np.asarray(files)

def prep_train_and_test_data(origin_path, train_ratio):
    # Too many values to unpack if I don't add next, but why
    _, dir, _ = next(os.walk(origin_path))


    train = []
    test = []


    for i in range(len(dir)):
        label = dir[i]
        img_path = os.path.join(origin_path, label)

        files = get_files_from_folder(img_path)

        train_count = math.floor(len(files) * train_ratio)

        local_train = []
        local_test = []

        for file in files:
            file_path = os.path.join(img_path, file)

            if (imghdr.what(file_path) == 'jpeg'):
                cv_img_read = cv2.imread(file_path)[...,::-1]
                cv_resize_img = cv2.resize(cv_img_read, (225, 225))

                if len(local_train) < train_count:
                    local_train.append([cv_resize_img, label])
                else:
                    local_test.append([cv_resize_img, label])

        train.extend(local_train)
        test.extend(local_test)
    return train, test

        
def plot_data(data_arr):
    l = []
    for [item, label] in data_arr:
        l.append(label)
        
    sns.countplot(x=l)

    plt.show()

    

def parse_arguments():
    parser = argparse.ArgumentParser(description="Dataset divider")
    parser.add_argument("--data_origin_path", required=True,
        help="Path to data")
    parser.add_argument("--test_data_path", required=True,
        help="Path in which the test data should be saved")
    parser.add_argument("--train_ratio", required=True,
        help="Train ratio - 0.7 means splitting data in 70 % train and 30 % test")
    return parser.parse_args()


args = parse_arguments()

# Generate training and test arrays
train, test = prep_train_and_test_data(args.data_origin_path, float(args.train_ratio))




