from matplotlib.colors import Colormap
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Reshape, Concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import scipy.sparse as sp
from spektral.data import Dataset, DisjointLoader, Graph, Loader, MixedLoader
from spektral.layers import GCSConv, GlobalAvgPool, GlobalMaxPool, GlobalSumPool
from spektral.transforms.normalize_adj import NormalizeAdj
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import sparse_categorical_accuracy
from pylab import *
import math
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import random_rotation, apply_affine_transform
from enum import Enum
class ModelnameEnum(Enum):
    IBH = "IBH"
    ROBIN = "Robin"
class PaddingEnum(Enum):
    REFLECTIVE = "REFLECT"
    SYMMETRIC = "SYMMETRIC"
    CONSTANT = "CONSTANT"
class DatasetEnum(Enum):
    C10 = "Cifar10"
    C100 = "Cifar100"
    MNIST = "Mnist"
    SVHN = "Svhn"

model_used = ModelnameEnum.IBH.value
dataset_name = DatasetEnum.SVHN.value


block_size = 4
stride = 4
channel_amount = 3
padding_type = PaddingEnum.REFLECTIVE.value

if dataset_name == DatasetEnum.MNIST.value:
    padding_type = PaddingEnum.CONSTANT.value
    channel_amount = 1
if model_used == ModelnameEnum.ROBIN.value:
    block_size = 2
    stride = 2

LEARNING_RATE = 5e-4  # Learning rate
EPOCHS = 10000  # Number of training epochs
ES_PATIENCE = 15  # Patience for early stopping
BATCH_SIZE = 32  # Batch size
STEPS_PER_EPOCHS_TRAIN = 20
STEPS_PER_EPOCHS_EVAL = 5
IMAGE_SIZE = 32 #28*28 images in mnist
PADDING = math.ceil((math.sqrt(IMAGE_SIZE ** 2 + IMAGE_SIZE ** 2) - IMAGE_SIZE) / 2)
CROP = IMAGE_SIZE / (IMAGE_SIZE + 2*PADDING)

block_amount =  ((IMAGE_SIZE - block_size)/stride + 1) #14*14 number of blocks pr image

model_name = "/" + dataset_name  +"/" + "t4" +model_used+dataset_name
model_folder = "./"+model_used+"/"
model_path = model_folder + model_name

DROPOUT = 0.12
L2REG = 5e-4

if (block_amount % 1) > 0:
    print("block amount is float my guy, you did a woopsie ", block_amount)
else:
    block_amount = int(block_amount)

# BLOCK PREP
mnist = tf.keras.datasets.mnist
cifar = tf.keras.datasets.cifar10



def splitImage(im):
    return [im[x:x+block_size,y:y+block_size] for x in range(0,im.shape[0],block_size) for y in range(0,im.shape[1],block_size)]

def createPatch(im, x, y):
	patch = np.zeros((block_size, block_size, channel_amount))
	for i in range(block_size):
		for j in range(block_size):
			patch[i][j] = im[y+i][x+j]
	return patch

def makePatches(im, x, y, array):
    if x+block_size > IMAGE_SIZE or y + block_size > IMAGE_SIZE:
        print("woops")
    array.append(createPatch(im, x, y))
    if x+block_size == IMAGE_SIZE:
        if y+block_size == IMAGE_SIZE:
            return
        else:	
            x = 0
            y = y + stride
    else:
        x = x + stride
    makePatches(im, x, y, array)

def showImg(tiles):
    plt.figure(figsize=(10, 10))
    count = 0
    for r in range(block_amount):
        for c in range(block_amount):
            ax = plt.subplot(block_amount, block_amount, count+1)
            plt.imshow(tiles[count])
            count += 1
    plt.show()


def rotateImages(imgs, rotation_degree = 0, testing = False):
    temp = []
    for i in imgs:
        i = tf.reshape(i, [IMAGE_SIZE, IMAGE_SIZE, channel_amount])
        paddings = np.array(tf.constant([[PADDING,PADDING],[PADDING,PADDING],[0,0]]), dtype=int32)
        padded = np.array(tf.pad(i, paddings, padding_type))
        if testing:
            rotated = apply_affine_transform(padded, theta=rotation_degree)
        else:
            rotated = random_rotation(padded, 180, channel_axis=2)
        a = tf.image.central_crop(rotated,CROP)
        temp.append(a)
    x_test = np.array(temp)
    return x_test

def showFeaturesNonRGB(features, dim):
    values = features
    npArr = np.array(list(values))
    npMatrix = npArr.reshape((dim,dim))
    plt.imshow((npMatrix), interpolation='nearest')
    plt.show()

def getFeatures(images):
    size = np.array(images[0]).shape[0]
    NormImages=images
    result = []
    index = 0
    for img in tqdm(range(len(images))):
        features = {}
        for y in range(size):
            for x in range(size):
                features[index] = [float(NormImages[img][y][x])]
                index = index+1
        features = np.array(list(features.values()))
        result.append(features)
    return result


def rgb_to_gray(img):
        zeroes = np.zeros((IMAGE_SIZE, IMAGE_SIZE))
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                zeroes[i][j] = 0.299 * img[i][j][0] + 0.587 * img[i][j][1] + 0.114 * img[i][j][2]
        return np.array(zeroes)