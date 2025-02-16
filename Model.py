import cv2
import os
import random 
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.models import Model #type: ignore
from tensorflow.keras.layers import Layer, Conv2D ,Dense, MaxPooling2D , Input ,Flatten #type: ignore
import tensorflow as tf 
from PIL import Image 

class L1Dist(Layer):
    def __init__(self):
        super(L1Dist, self).__init__()

    def call(self,input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)




