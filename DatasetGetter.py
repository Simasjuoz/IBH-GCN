import numpy as np
import tensorflow as tf
import tensorflow_datasets.image_classification as tfds
import os.path

def getCifar10():
    cifar = tf.keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = cifar.load_data()

    x_train, M, S = zscoreNorm(x_train, M=None, S=None)
    x_test, _, _ = zscoreNorm(x_test, M=M, S=S)

    x_train = np.array(x_train, dtype=np.float32)
    x_test = np.array(x_test, dtype=np.float32)
    return (np.array(x_train), y_train), (np.array(x_test), y_test)

def getCifar100():
    cifar100 = tf.keras.datasets.cifar100
    (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode="fine")

    x_train, M, S = zscoreNorm(x_train, M=None, S=None)
    x_test, _, _ = zscoreNorm(x_test, M=M, S=S)

    x_train = np.array(x_train, dtype=np.float32)
    x_test = np.array(x_test, dtype=np.float32)
    return (np.array(x_train), y_train), (np.array(x_test), y_test)

def getMnist():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    x_train = np.expand_dims(x_train, axis=-1)
    x_train = tf.image.resize(x_train, [32,32])

    x_train = zscoreNormMnist(np.array(x_train))
    x_test, y_test = x_train[40000:50000], y_train[40000:50000]
    x_train, y_train = x_train[0:40000], y_train[0:40000]

    x_train = np.array(x_train, dtype=np.float32)
    x_test = np.array(x_test, dtype=np.float32)
    return (np.array(x_train), y_train), (np.array(x_test), y_test)

def getSvhn():
    svhn_builder = tfds.SvhnCropped()
    svhn_builder.download_and_prepare()

    train_dataset = svhn_builder.as_dataset(split="train")
    test_dataset = svhn_builder.as_dataset(split="test")

    trainImgs = []
    trainLabels = []
    file_exists = os.path.exists('svend/data.npy')

    if file_exists:
        with open('svend/data.npy', 'rb') as f:
            trainImgs = np.load(f)
            trainLabels = np.load(f)
    else:
        for a_train_example in train_dataset.take(73257):                
            image, label = a_train_example["image"], a_train_example["label"]
            trainImgs.append(image)
            if label == 9:
                label = 6
            trainLabels.append(label)
        for a_test_example in test_dataset.take(26032):
            image, label = a_test_example["image"], a_test_example["label"]
            trainImgs.append(image)
            if label == 9:
                label = 6
            trainLabels.append(label)
        with open('svend/data.npy', 'wb') as f:
            np.save(f, trainImgs)
            np.save(f, trainLabels)

    x_train = np.array(trainImgs, dtype=np.float32)
    y_train = np.array(trainLabels, dtype=np.float32)
    x_test, y_test = x_train[73257:73257+26032], y_train[73257:73257+26032]
    x_train, y_train = x_train[0:73257], y_train[0:73257]

    x_train, M, S = zscoreNorm(x_train, M=None, S=None)
    x_test, _, _ = zscoreNorm(x_test, M=M, S=S)
        
    return (x_train, y_train), (x_test, y_test)


def zscoreNorm(images, *, M, S):
    images = (images.astype(np.float32) / 255.)
    if ( M is None ) or (S is None):
        M, S = np.mean(images, axis=(0,1,2), keepdims=True), np.std(images, axis=(0,1,2), keepdims=True)
    images = (images-M) / S
    return images, M, S

def zscoreNormMnist(images):
    images = (images.astype(np.float32) / 255.)
    images = (images-0.1307) / 0.3081
    return images
