import matplotlib.pyplot as plt
import seaborn as sns
import math
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout 
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

from sklearn.metrics import classification_report,confusion_matrix

import tensorflow as tf

import cv2
import os

import numpy as np

import argparse

import imghdr

import shutil

import filetype

img_size = 224
epoch_val = 5

batch_size = 64


def get_files_from_folder(path):
    files = os.listdir(path)
    return np.asarray(files)

def prep_train_and_test_data(origin_path, train_ratio):
    # Too many values to unpack if I don't add next, but why
    _, dir, _ = next(os.walk(origin_path))

    print(len(dir))


    train_data = []
    test_data = []
    train_labels = []
    test_labels = []

    confusion_matr_labels = []


    for i in range(len(dir)):
        label = dir[i]

        confusion_matr_labels.append(f"{label} - Class {i}")

        img_path = os.path.join(origin_path, label)

        files = get_files_from_folder(img_path)

        train_count = math.floor(len(files) * train_ratio)

        local_train_data = []
        local_test_data = []
        local_train_labels = []
        local_test_labels = []

        

        for file in files:
            file_path = os.path.join(img_path, file)

            if (filetype.is_image(file_path)):
                try:
                    cv_img_read = cv2.imread(file_path)[...,::-1]
                    cv_resize_img = cv2.resize(cv_img_read, (img_size, img_size))


                    if len(local_train_data) < train_count:
                        local_train_data.append(cv_resize_img)
                        local_train_labels.append(i)
                    else:
                        local_test_data.append(cv_resize_img)
                        local_test_labels.append(i)
                except Exception as e:
                    print(e)


        train_data.extend(local_train_data)
        train_labels.extend(local_train_labels)
        
        test_data.extend(local_test_data)
        test_labels.extend(local_test_labels)
    

    return np.array(train_data), np.array(train_labels), np.array(test_data), np.array(test_labels), len(dir), confusion_matr_labels

        
def plot_data(data_arr):
    l = []
    for [_, label] in data_arr:
        l.append(label)
        
    sns.countplot(x=l)

    plt.show()

def data_preprocess(train_data, train_labels, te bst_data, test_labels):
    x_train = []
    y_train = []
    x_val = []
    y_val = []


    # for feature, label in train:
    #     x_train.append(feature)
    #     y_train.append(label)

    # for feature, label in test:
    #     x_val.append(feature)
    #     y_val.append(label)

    # Normalize the data
    x_train = np.array(train_data) / 255
    x_val = np.array(test_data) / 255

    x_train.reshape(-1, img_size, img_size, 1)
    y_train = np.array(train_labels)

    x_val.reshape(-1, img_size, img_size, 1)
    y_val = np.array(test_labels)

    # for feature, label in train:
    #     normalizedTrainFeature = np.zeros((img_size, img_size))
    #     normalizedTrainFeature = cv2.normalize(feature, normalizedTrainFeature, 0, 255, cv2.NORM_MINMAX)
    #     x_train.append(normalizedTrainFeature)
    #     y_train.append(label)

    # for feature, label in test:
    #     normalizedTestFeature = np.zeros((img_size, img_size))
    #     normalizedTestFeature = cv2.normalize(feature, normalizedTestFeature, 0, 255, cv2.NORM_MINMAX)
    #     x_test.append(normalizedTestFeature)
    #     y_test.append(label)


    

    # Normalize the data
    # x_train = np.array(x_train) / 255
    # x_test = np.array(x_test) / 255

    # x_train = np.array(x_train).reshape(-1, img_size, img_size, 1)
    # y_train = np.array(y_train)


    # x_test = np.array(x_test).reshape(-1, img_size, img_size, 1)
    # y_test = np.array(y_test)

    # return x_train, y_train, x_test, y_test
    return x_train, y_train, x_val, y_val

    

def parse_arguments():
    parser = argparse.ArgumentParser(description="Dataset divider")
    parser.add_argument("--data_origin_path", required=True,
        help="Path to data")
    parser.add_argument("--train_ratio", required=True,
        help="Train ratio - 0.7 means splitting data in 70 % train and 30 % test")
    return parser.parse_args()


args = parse_arguments()


# Enable eager execution
tf.config.run_functions_eagerly(True)

# Generate training and test arrays
train_data, train_labels, test_data, test_labels, label_count, conf_labels = prep_train_and_test_data(args.data_origin_path, float(args.train_ratio))

x_train_arr, y_train_arr, x_test_arr, y_test_arr = data_preprocess(train_data, train_labels, test_data, test_labels)

# # ??
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range = 30,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.2, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip = True,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(x_train_arr)


model = Sequential()
model.add(Conv2D(32,3,padding="same", activation="relu", input_shape=(224,224,3)))
model.add(MaxPool2D())

model.add(Conv2D(32, 3, padding="same", activation="relu"))
model.add(MaxPool2D())

model.add(Conv2D(64, 3, padding="same", activation="relu"))
model.add(MaxPool2D())
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(128,activation="relu"))
model.add(Dense(label_count, activation="softmax"))

model.summary()

opt = Adam(learning_rate=0.000001)


train_generator = datagen.flow(x_train_arr, y_train_arr, batch_size)
# Questioning my life decisions????
model.compile(optimizer = opt , loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True) , metrics = ['accuracy'])


# history = model.fit(x_train_arr, y_train_arr, epochs = epoch_val, validation_data=(x_test_arr, y_test_arr))

history = model.fit_generator(
    train_generator,
    steps_per_epoch=len(x_train_arr) // batch_size,  # Number of batches per epoch
    epochs=epoch_val,
    validation_data=(x_test_arr, y_test_arr)
)

predictions = model.predict(x_test_arr)

# Get the class indices with argmax
predictions = predictions.argmax(axis=-1)
predictions = predictions.reshape(1,-1)[0]
print(classification_report(y_test_arr, predictions, target_names = conf_labels))


model.save("animal_model.h5")
model.save("animal_keras_model.keras")