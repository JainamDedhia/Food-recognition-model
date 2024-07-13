import tensorflow as tf
import numpy as np
import os
import random
from collections import defaultdict
from shutil import copy, copytree, rmtree
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras import regularizers

# Helper function to prepare data
def prepare_data(filepath, src, dest):
    classes_images = defaultdict(list)
    with open(filepath, 'r') as txt:
        paths = [read.strip() for read in txt.readlines()]
        for p in paths:
            food = p.split('/')
            classes_images[food[0]].append(food[1] + '.jpg')

    for food in classes_images.keys():
        print("\nCopying images into ", food)
        if not os.path.exists(os.path.join(dest, food)):
            os.makedirs(os.path.join(dest, food))
        for i in classes_images[food]:
            copy(os.path.join(src, food, i), os.path.join(dest, food, i))
    print("Copying Done!")

# Helper function to create dataset subsets
def dataset_mini(food_list, src, dest):
    if os.path.exists(dest):
        rmtree(dest)
    os.makedirs(dest)
    for food_item in food_list:
        print("Copying images into", food_item)
        copytree(os.path.join(src, food_item), os.path.join(dest, food_item))

# Function to plot accuracy
def plot_accuracy(history, title):
    plt.title(title)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train_accuracy', 'validation_accuracy'], loc='best')
    plt.show()

# Function to plot loss
def plot_loss(history, title):
    plt.title(title)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train_loss', 'validation_loss'], loc='best')
    plt.show()

# Function to predict class of images
def predict_class(model, images, food_list, show=True):
    for img in images:
        img = tf.keras.preprocessing.image.load_img(img, target_size=(224, 224))
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img /= 255.

        pred = model.predict(img)
        index = np.argmax(pred)
        food_list.sort()
        pred_value = food_list[index]
        if show:
            plt.imshow(img[0])
            plt.axis('off')
            plt.title(pred_value)
            plt.show()

# Helper function to select n random food classes
def pick_n_random_classes(n, foods_sorted):
    food_list = []
    random_food_indices = random.sample(range(len(foods_sorted)), n)
    for i in random_food_indices:
        food_list.append(foods_sorted[i])
    food_list.sort()
    return food_list

# Main code starts here
if __name__ == '__main__':
    # List of food classes
    foods_sorted = ['apple_pie', 'beef_carpaccio', 'bibimbap', 'cup_cakes', 'foie_gras',
                    'french_fries', 'garlic_bread', 'pizza', 'spring_rolls', 'spaghetti_carbonara', 'strawberry_shortcake']

    # Paths
    src_train = 'train'
    dest_train = 'train_mini/'
    src_test = 'test'
    dest_test = 'test_mini/'

    # Prepare mini datasets
    dataset_mini(foods_sorted, src_train, dest_train)
    dataset_mini(foods_sorted, src_test, dest_test)

    # Model parameters
    n_classes = len(foods_sorted)
    img_width, img_height = 224, 224
    nb_train_samples = 8250
    nb_validation_samples = 2750
    batch_size = 16

    # Data generators
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        'train_mini',
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        'test_mini',
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')

    # Model architecture (using ResNet50)
    resnet50 = ResNet50(weights='imagenet', include_top=False)
    x = resnet50.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    predictions = Dense(n_classes, kernel_regularizer=regularizers.l2(0.005), activation='softmax')(x)
    model = Model(inputs=resnet50.input, outputs=predictions)
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

    # Callbacks
    checkpointer = ModelCheckpoint(filepath='best_model_11class.hdf5', verbose=1, save_best_only=True)
    csv_logger = CSVLogger('history_11class.log')

    # Training the model
    history_11class = model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size,
        epochs=30,
        verbose=1,
        callbacks=[csv_logger, checkpointer])

    # Save the trained model
    model.save('model_trained_11class.hdf5')

    # Mapping of classes
    class_map_11 = train_generator.class_indices

    # Plot accuracy and loss
    plot_accuracy(history_11class, 'FOOD101-ResNet50')
    plot_loss(history_11class, 'FOOD101-ResNet50')
