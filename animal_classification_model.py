import matplotlib.pyplot as plt
import seaborn as sns

from keras.models import Sequential
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, SGD, RMSprop
from keras.regularizers import l2

from sklearn.metrics import classification_report  

import tensorflow as tf

import cv2
import os

import numpy as np

import argparse


import filetype

from sklearn.model_selection import train_test_split


img_size = 224
epoch_val = 550

batch_size = 32


def get_files_from_folder(path):
    files = os.listdir(path)
    return np.asarray(files)

def prep_train_and_test_data(origin_path, train_ratio):
    _, dir, _ = next(os.walk(origin_path))


    train_data = []
    test_data = []
    train_labels = []
    test_labels = []

    classification_labels = []


    for i, label in enumerate(dir):
        classification_labels.append(f"{label} - Class {i}")

        img_path = os.path.join(origin_path, label)
        files = get_files_from_folder(img_path)

        # Split files into training and testing sets
        train_files, test_files = train_test_split(files, train_size=train_ratio, random_state=42)

        for file in train_files:
            file_path = os.path.join(img_path, file)
            if (filetype.is_image(file_path)):
                try:
                    cv_img_read = cv2.imread(file_path)[...,::-1]
                    cv_resize_img = cv2.resize(cv_img_read, (img_size, img_size))
                    train_data.append(cv_resize_img)
                    train_labels.append(i)
                except Exception as e:
                    print(e)

        for file in test_files:
            file_path = os.path.join(img_path, file)
            if (filetype.is_image(file_path)):
                try:
                    cv_img_read = cv2.imread(file_path)[...,::-1]
                    cv_resize_img = cv2.resize(cv_img_read, (img_size, img_size))
                    test_data.append(cv_resize_img)
                    test_labels.append(i)
                except Exception as e:
                    print(e)

    return np.array(train_data), np.array(train_labels), np.array(test_data), np.array(test_labels), len(dir), classification_labels

        
def plot_data(data_arr):
    l = []
    for [_, label] in data_arr:
        l.append(label)
        
    sns.countplot(x=l)

    plt.show()



def reshape_labels(y_labels, num_classes):
    num_samples = len(y_labels)
    y_binary = np.zeros((num_samples, num_classes), dtype=np.int32)  # Initialize binary label array
    
    for i, label in enumerate(y_labels):
        # Assuming label is an integer representing the class index
        # Set the corresponding position to 1 to indicate the presence of the class
        y_binary[i, label] = 1
    
    return y_binary



def data_preprocess(train_data, train_labels, test_data, test_labels, num_classes):
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
train_data, train_labels, test_data, test_labels, label_count, class_labels = prep_train_and_test_data(args.data_origin_path, float(args.train_ratio))

x_train_arr, y_train_arr, x_test_arr, y_test_arr = data_preprocess(train_data, train_labels, test_data, test_labels, label_count)

# Data augmentation
train_datagen = ImageDataGenerator(
    rotation_range=30,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator()


train_datagen.fit(x_train_arr)



# Transfer Learning with a Pre-trained Model (VGG16 in this example)
base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))

# Freeze convolutional layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom classification head
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(label_count, activation='sigmoid')
])

# Implement learning rate scheduler
# lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)


# Define the learning rate and momentum
learning_rate = 0.00125  # Starting learning rate

# Adjust learning rate based on dataset size
# Rule of thumb: Smaller datasets may require larger learning rates
optimizer = RMSprop(learning_rate=learning_rate)
# optimizer = Adam(learning_rate=initial_learning_rate)

# Compile the model 
model.compile(optimizer=optimizer,
              loss='binary_crossentropy',  # for multilabel classification
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_datagen.flow(x_train_arr, y_train_arr, batch_size=batch_size),
    epochs=epoch_val,
    validation_data=test_datagen.flow(x_test_arr, y_test_arr, batch_size=batch_size)
)

# history = model.fit(x_train_arr, y_train_arr, epochs = epoch_val, validation_data=(x_test_arr, y_test_arr))

predictions = model.predict(x_test_arr)
# Convert predicted probabilities to binary predictions using a threshold
threshold = 0.5  # Adjust threshold as needed
binary_predictions = (predictions > threshold).astype(int)


# Show classification report
print(classification_report(y_test_arr, binary_predictions, target_names=class_labels))

model.save("animal_keras_model.keras")