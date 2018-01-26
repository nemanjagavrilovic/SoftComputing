import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np
import os
import cv2

# keras
from keras.models import Sequential
from keras.layers.core import Dense,Activation
from keras.optimizers import SGD
from keras.utils import np_utils


def matrix_to_vector(image):
    return image.flatten()


def scale_to_range(image): # skalira elemente slike na opseg od 0 do 1
    return image/255


def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def resize_region(region):
    '''Transformisati selektovani region na sliku dimenzija 28x28'''
    return cv2.resize(region,(28,28), interpolation = cv2.INTER_NEAREST)


def select_region(image_bin):
    x = 1130;
    y = 1280;
    h = 340;
    w = 200;
    region = image_bin[y:y + h + 1, x:x + w + 1]
    region=resize_region(region)
    return  region


ann=Sequential('NeuralNetwork.h5')
image_color1 = load_image('images/12 tref.jpg')
region = select_region(image_color1)

result=ann.predict(np.array(matrix_to_vector(scale_to_range(region))))
print( result)
print( display_result(result,labels))
