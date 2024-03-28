import matplotlib.pyplot as plt
import seaborn as sns
import math
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

from sklearn.metrics import classification_report,confusion_matrix

import tensorflow as tf

import cv2
import os

import numpy as np

import argparse

import shutil

import filetype

img_size = 150
epoch_val = 1

batch_size = 32


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



# Example: Convert integer labels to binary arrays
def reshape_labels(y_labels, num_classes):
    num_samples = len(y_labels)
    y_binary = np.zeros((num_samples, num_classes), dtype=np.int32)  # Initialize binary label array
    
    for i, label in enumerate(y_labels):
        # Assuming label is an integer representing the class index
        # Set the corresponding position to 1 to indicate the presence of the class
        y_binary[i, label] = 1
    
    return y_binary



def data_preprocess(train_data, train_labels, test_data, test_labels, num_classes):
    # # Convert list of images to NumPy arrays
    # x_train = np.array(train_data)
    # x_val = np.array(test_data)
    
    # # Normalize pixel values
    # x_train = x_train / 255.0
    # x_val = x_val / 255.0
    
    # # Reshape data arrays for CNNs
    # x_train = x_train.reshape(-1, img_size, img_size, 3)  # Assuming 3 channels for RGB images
    # x_val = x_val.reshape(-1, img_size, img_size, 3)      # Assuming 3 channels for RGB images
    
    # y_train = np.array(train_labels)
    # y_val = np.array(test_labels)

    # Normalize pixel values
    x_train = train_data / 255.0
    x_test = test_data / 255.0

    x_train = x_train.reshape(-1, img_size, img_size, 3)  # Assuming 3 channels for RGB images
    x_test = x_test.reshape(-1, img_size, img_size, 3)      # Assuming 3 channels for RGB images


    y_train = reshape_labels(train_labels, num_classes)
    y_test = reshape_labels(test_labels, num_classes)

    return x_train, y_train, x_test, y_test

    

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

x_train_arr, y_train_arr, x_test_arr, y_test_arr = data_preprocess(train_data, train_labels, test_data, test_labels, label_count)

# # ??
# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=30,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=False
)


# datagen.fit(x_train_arr)


# Create model
model = Sequential([
    Conv2D(16, 3, padding="same", activation="relu", input_shape=(img_size,img_size,3)),
    MaxPool2D(),
    Conv2D(32, 3, padding="same", activation="relu"),
    MaxPool2D(),
    Flatten(),
    Dense(32, activation="relu"),
    Dense(label_count, activation="sigmoid")  # Using sigmoid activation for multi-label classification
])

model.summary()

opt = Adam(learning_rate=0.000001)

train_generator = datagen.flow(x_train_arr, y_train_arr, batch_size=batch_size)

# Questioning my life decisions????
model.compile(optimizer = opt , loss='binary_crossentropy', metrics = ['accuracy'])

# So for some odd reason, this array is empty...
print(x_test_arr)

# history = model.fit(x_train_arr, y_train_arr, epochs = epoch_val, validation_data=(x_test_arr, y_test_arr))

# Calculate the number of steps per epoch
steps_per_epoch = len(x_train_arr) // batch_size

# If there are remaining samples, adjust the steps_per_epoch to include them in the last partial batch
if len(x_train_arr) % batch_size != 0:
    steps_per_epoch += 1

history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
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