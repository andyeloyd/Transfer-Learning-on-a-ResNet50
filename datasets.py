import tensorflow as tf
import tensorflow_addons as tfa
import cv2 as cv
import os
import numpy as np
import tqdm
import random

def load_dataset_from_dir(dataset_dir):
    # Dataset directory must contain a subdirectory for each class. Inside each subdirectory, all images for
    # such class must be contained.
    if not os.path.isdir(dataset_dir):
        print('Invalid directory.')
        return
    print('Loading dataset from %s' % dataset_dir)
    x = list()
    y = list()
    id_list = os.listdir(dataset_dir)
    id_list.sort(key=int)
    id_index = 0
    for subdir in tqdm.tqdm(id_list):
        subdir_path = os.path.join(dataset_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue
        image_list = list()
        for image in os.listdir(subdir_path):
            image_path = os.path.join(subdir_path, image)
            img = cv.imread(image_path)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            #img = subtract_resnet_mean(img).astype(np.float32)
            image_list.append(img)
        labels = [id_index for file in range(len(image_list))]
        x.extend(image_list)
        y.extend(labels)
        id_index += 1

    #np.random.shuffle(x)
    y = np.reshape(y, newshape=(len(y),1))
    return x, y

def subtract_resnet_mean(x,y):
    mean = (91.4953, 103.8827, 131.0912)
    x = tf.cast(x, dtype=tf.float32)
    x = x[:, :, ::-1] - mean
    return x,y


def split_dataset_list(x, y, split=0.4):
    zipped_data = list(zip(x, y))
    random.shuffle(zipped_data)
    x, y = list(zip(*zipped_data))
    split_point = int((1 - split) * len(x))
    x_train = np.asarray(x[:split_point])
    y_train = np.asarray(y[:split_point])
    x_val = np.asarray(x[split_point:])
    y_val = np.asarray(y[split_point:])
    return x_train, y_train, x_val, y_val


def deterministic_brightness_augmentation(x, y):
    # Aumento de datos al alterar aleatoriamente la saturation y brillo
    # Los datos aumentados se agregan a los datos originales, duplicando la cantidad de ellos

    augmented_x = []
    augmented_y = []

    n_samples = range(len(x))

    for i in n_samples:
        img = x[i]
        label = y[i]
        value = np.random.uniform(0.2, 2)
        # HSV = Hue, saturation, value
        #hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        hsv_img = cv.cvtColor(img, cv.COLOR_RGB2HSV)
        hsv_img = np.array(hsv_img, dtype=np.float64)

        hsv_img[:, :, 1] = hsv_img[:, :, 1] * value
        hsv_img[:, :, 1][hsv_img[:, :, 1] > 255] = 255
        hsv_img[:, :, 2] = hsv_img[:, :, 2] * value
        hsv_img[:, :, 2][hsv_img[:, :, 2] > 255] = 255
        hsv_img = np.array(hsv_img, dtype=np.uint8)
        #img = cv.cvtColor(hsv_img, cv.COLOR_HSV2BGR)
        img = cv.cvtColor(hsv_img, cv.COLOR_HSV2RGB)

        augmented_x.append(img)
        augmented_y.append(label)
        # img = np.expand_dims(img, axis=0)
        # label = np.expand_dims(label, axis=0)
        # np.append(augmented_x, img, axis=0)
        # np.append(augmented_y, label, axis=0)
    augmented_x = np.array(augmented_x)
    augmented_y = np.array(augmented_y)

    x = np.append(x, augmented_x, axis=0)
    y = np.append(y, augmented_y, axis=0)
    return x, y


def flip_image(x, y):
    x = tf.image.random_flip_left_right(x)
    return x, y


def rotate_img(img, y):
    angle = np.random.uniform(low=-np.pi / 18, high=np.pi / 18)
    img = tfa.image.rotate(img, angle, fill_mode='nearest')
    return img, y

def standardize_per_image(x, y):
    x = tf.image.per_image_standardization(x)
    return x, y

def rescale(x, y):
    x = x / 255.0
    return x, y

trainAug = tf.keras.models.Sequential([
           tf.keras.layers.RandomFlip(mode="horizontal"),
           tf.keras.layers.RandomRotation(0.05),
           #tf.keras.layers.Rescaling(scale=1.0 / 255)
           tf.keras.layers.Rescaling(scale=1./127.5, offset=-1)
])

data_rescaling = tf.keras.models.Sequential([
    tf.keras.layers.Rescaling(scale=1.0 / 255)
])

def arcface_data_format(x, y):
    return (x, y), y


def generate_train_dataset(x, y, batch_size=64, arcface_format=False):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.map(subtract_resnet_mean)

    dataset = dataset.map(flip_image).map(rotate_img)
    dataset = dataset.shuffle(batch_size * 100)
    dataset = dataset.batch(batch_size)

    dataset = dataset.map(lambda x, y: (data_rescaling(x), y))
    if arcface_format:
        dataset = dataset.map(arcface_data_format)

    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def generate_val_dataset(x, y, batch_size=64, arcface_format=False):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.map(subtract_resnet_mean)

    dataset = dataset.shuffle(batch_size * 100)
    dataset = dataset.batch(batch_size)

    dataset = dataset.map(lambda x, y: (data_rescaling(x), y))
    if arcface_format:
        dataset = dataset.map(arcface_data_format)

    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def get_dataset_from_directory(dataset_path, batch_size=64, bright_augment=True, split=0.33, arcface_format=False):
    x, y = load_dataset_from_dir(dataset_path)

    if bright_augment:
        x, y = deterministic_brightness_augmentation(x, y)
    x_train, y_train, x_val, y_val = split_dataset_list(x, y, split=split)

    train_dataset = generate_train_dataset(x_train, y_train, batch_size=batch_size, arcface_format=arcface_format)
    val_dataset = generate_val_dataset(x_val, y_val, batch_size=batch_size, arcface_format=arcface_format)

    print('Loaded a training dataset of %d samples.' %len(x_train))
    print('Loaded a validation dataset of %d samples.' % len(x_val))
    return train_dataset, val_dataset