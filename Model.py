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

gpus =tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus :
        tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU available : {gpu}")


POS_PATH=os.path.join('data','positive')
NEG_PATH=os.path.join('data','negitive')
ANC_PATH=os.path.join('data','anchor')

os.makedirs(POS_PATH,exist_ok=True)
os.makedirs(NEG_PATH,exist_ok=True)
os.makedirs(ANC_PATH,exist_ok=True)

BASE_DIR= r'C:\Users\hp\Desktop\MODEL\data\negative\lfw-deepfunneled\lfw-deepfunneled'
