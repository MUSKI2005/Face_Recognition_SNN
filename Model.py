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

def print_dataset_structure(BASE_DIR):
    print("DATASET STUCTURE:")
    for person_name in os.listdir(BASE_DIR):
        person_folder=os.path.join(BASE_DIR,person_name)
        if os.path.isdir(person_folder):
            print(f"Person: {person_name}")
            for image_name in os.listdir(person_folder):
                print(f"Image: {image_name}")

def display_sample_images(BASE_DIR, num_samples=5):
    print("Displaying the sample images of each person:\n")
    count=0
    for person_name in os.listdir(BASE_DIR):
        person_folder=os.path.join(BASE_DIR,person_name)
        if os.path.isdir(person_folder):
            for image_name in os.listdir(person_folder):
                image_path=os.path.join(person_folder,image_name)
                img=Image.open(image_path)
                plt.imshow(img)
                plt.title(f"{person_name} : {image_name}")
                plt.axis('off')
                plt.show()
                count+=1
                if count >=num_samples:
                    return 


import cv2
import uuid

os.path.join(ANC_PATH,'{}.jpg'.format(uuid.uuid1()))
uuid.uuid1()
for i in range(10):
   cap=cv2.VedioCapture(i)
   if cap.isOpened():
       while cap.isOpened():
           ret,frame=cap.read()
           if not ret:
               print("failed to read the frame, existing...")
               break
           resized_frame=cv2.resize(frame,(250,250))

            #collecting anchors
           if cv2.waitkey(1) &0xFF ==ord('a'):
            imgname=os.path.join(ANC_PATH,'{}.jpg'.format(uuid.uuid1()))
            cv2.imwrite(imgname,resized_frame)

            # collecting positives
            if cv2.waitkey(1) &0xFF == ord('p'):
                imgname=os.path.join(POS_PATH,'{}.jpg'.format(uuid.uuid1()))
                cv2.imwrite(imgname,resized_frame)

            cv2.imshow('Resized image collection is this',resized_frame)

            if cv2.waitkey(1) & 0xFF == ord('q'):
                break
            print("Camera not forund at index {i}")

       else:
         print(f"camera not found at index {i}")

cap.release()
cv2.destroyAllWindows()

import tensorflow as tf
import os
import glob
import matplotlib.pyplot as plt
import uuid

# DATA AUGMENTATION
def data_aug(img):
    data=[]
    for i in range(4):
        img=tf.stateless_random_brightness(img,max_delta=0.02,seed=(1,2))
        img=tf.stateless_random_contrast(img,lower=0.6,upper=1,seed=(1,3))
        # img=tf.image.stateless_random_crop(img,size=(20,20,3),seed=(1,3))
        img=tf.stateless_random_flip_left_right(img,seed=(np.random.radiant(100),np.random.radiant(100)))
        img=tf.image.stateless_random_jpeg_quality(img,min_jpeg_quality=90,max_jpeg_quality=100,seed=(np.random.randint(100),np.random.randint(100)))
        img=tf.stateless_random_saturation(img,lower=0.9,upper=1,seed=(np.random.radiant(100),np.random.radiant(100)))
        data.append(img)
    return data

img_path=os.path.join(ANC_PATH,r"C:\Users\hp\Desktop\MODEL\data\anchor\00bbe314-dbf7-11ef-9aec-b2c3c39739c6.jpg")
img=cv2.imread(img_path)
augmented_images=data_aug(img)

for image in augmented_images:
    cv2.imwrite(os.path.join(POS_PATH,'{}.jpg'.format(uuid.uuid1())),image.numpy())

for file_name in os.listdir(os.path.join(POS_PATH)):
    img_path=os.path.join(POS_PATH,file_name)
    img=cv2.imread(img_path)
    augmented_images=data_aug(img)

    for image in augmented_images:
       cv2.imwrite(os.path.join(POS_PATH,'{}.jpg'.format(uuid.uuid1())),image.numpy())






