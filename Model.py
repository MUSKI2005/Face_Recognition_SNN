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

# get our image directories
anchor = tf.data.Dataset.list_files(rf"{ANC_PATH}/*.jpg").shuffle(buffer_size=1000)
positive = tf.data.Dataset.list_files(rf"{POS_PATH}/*.jpg").shuffle(buffer_size=1000)

# Use glob to find all negative images recursively
negative_file_paths=glob.glob(f"{NEG_PATH}/**/*.jpg",recursive=True)
if not negative_file_paths:
    raise ValueError("No files found in the {NEG_PATH} matching pattern '*.jpg'")
negative=tf.data.Dataset.from_tensor_slices(negative_file_paths).shuffle(buffer_size=1000)

# grabbing the anchors now
dir_test=anchor.as_numpy_iterator()
print(dir_test.next())

# grabbing the positives now
dir_test=positive.as_numpy_iterator()
print(dir_test.next())

# grabbing the nagatives now
dir_test=negative.as_numpy_iterator()
print(dir_test.next())

# now writing the preprocessing function for THE SNN Model 
def preprocess(file_path):
    # reading the image for the file path
    byte_img=tf.io.read_file(file_path)
    # loading the image
    img=tf.decode_jpeg(byte_img)
    # preprocessing the image to be 100*100*3
    img=tf.image.resize(img,(100,100))
    # scanning our image to bet 0 to 1
    img=img/255.0
    # returm the image

    img=preprocess(r'C:data\\anchor\\53d9b23d-d568-11ef-8fff-b2c3c39739c6.jpg') #here giving the file path of any random image
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    print(img.numpy().min())
    print(img.numpy().max())

# Create labelled dataset
positives=tf.data.Dataset.zip(anchor,positive,tf.data.Dataset.from_tensor_slices(tf.ones(len(list(anchor)))))  #when anchor+positive
negatives=tf.data.Dataset.zip(anchor,negative,tf.data.Dataset.from_tensor_slices(tf.ones(len(list(anchor)))))   #when anchor+negative
data=positives.concatenate(negatives)

print(data)
print(tf.ones(len(anchor))) #printing the number of ones
print(tf.zeros(len(anchor))) #printing the no. of zeros

# creating the class.
class_labels=tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))
iterator_labs=class_labels.as_numpy_iterator()
# now looping through each one of the labels in the class, and printing it
print(iterator_labs.next())

# printing the path for the anchors~positives shuffled
samples=data.as_numpy_iterator()
example=samples.next()
print(samples.next())

# now preprocessing the image directories with the twin function
def preprocess_twin(input_img,validation_img,label):
    return (preprocess(input_img),preprocess(validation_img),label)

res=preprocess_twin(*example)
print(res)
print(len(res))

# printing the image in the output
plt.imshow(res[0].numpy())
plt.show()

# Now making the dataloader pipeline
data=data.map(preprocess_twin,num_parallel_calls=tf.data.experimental.AUTOTUNE)
data=data.cache()

# shuffling the data
data=data.shuffle(buffer_size=3000)

print(data)
samples=data.as_numpy_iterator()
print(samples.next())

# printing the length of the array
print(len(samples.next()))

# Visualization of the array of ANCHOR IMAGES
samp=samples.next()
print(samp[0])

# displaying the anchors
plt.imshow(samp[0])
plt.show()

# either the negative or positive
print(samp[1])
plt.imshow(samp[1])
plt.show()

# checking the prior samples our labels will vary upto which,if 2nd label is -ve then it'll be ideally 0
print("Depending upon the 1st and the 2nd sample images the value of the label is: \n",samp[2]) 
if samp[2]==1: print("That is the positive sample!")
else: print("That is a negative sample!")

# Creating training partition here
train_data=data.take(round(len(data)*0.7))
train_data=train_data.batch(16)
train_data=train_data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE) #instead of doing it manually do it by using AUTOTUNE

val_data=data.skip(round(len(data)*0.7))
val_data=val_data.take(round(len(data)*0.3))
val_data=val_data.batch(16)
val_data=val_data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# printing the length of our train_data
print(round(len(data) * 0.7)) #fetching 1st 480 samples

# Comparing our previous data with train_data
print(data)
# Along with it dispalying the status of our train_data
print(train_data) #slightly changed

# now getting the array of our created batches
train_samples=train_data.as_numpy_itrerator()
train_sample=train_samples.next()
print(train_sample)

# getting the length of the train_data
print(len(train_sample))

# Getting the length of the 1st sample
print(len(train_sample[0])) #having 16 images of each sample from negative, positive, anchor from all three channels

# Testing the data
test_data=data.skip(round(len(data)*0.7)) #here we skipped 1st 480 observations
test_data=test_data.take(round(len(data)*0.3))
test_data=test_data.batch(16)
# prefetching
test_data=test_data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# printing the length of our test_data
print(round(len(data)*0.7))

# now printing the length of last remaining 30% of our data ~206 images
print(round(len(data) * 0.3))

# Now building the embedding layer
# create an L1 distance layer
# compile the Siamese network
# SAI- MESE

# so basically,
# convolutional 
# relu
# max pooling
# then again repeat these three
inp=Input(shape=(100,100,3),name='input_image')
print(inp)
c1=Conv2D(64,(10,10),activation='relu')(inp)
c1

m1=MaxPooling2D(64,(2,2),padding='same')(c1)
m1

c2=Conv2D(128,(7,7),activation='relu')(m1)
m2=MaxPooling2D(128,(2,2),padding='same')(c2)

c3=Conv2D(128,(4,4),activation='relu')(m2)
m3=MaxPooling2D(64,(2,2),padding='same')(c3)

# c4=Conv2D(256,(4,4),activation='relu')(m3)
# f1=Flatten()(c4) #flattening all of the elements
# d1=Dense(4096,activation='sigmoid')(f1)

def make_embedding():
    inp=Input(shape=(100,100,3),name='input_image')

     # 1st BLOCK
    # 1st layer convolutional +relu activation in this 64 filters in the convo. and for the shape 10*10 pixals
    # Then impliment the max pooling layer passing 64 units
    c1=Conv2D(64,(10,10),activation='relu')(inp)
    m1=MaxPooling2D(64,(2,2),padding='same')(c1)

     # 2nd BLOCK
    #now implimenting the next block of convolutional+ relu
    c2=Conv2D(128,(7,7),activation='relu')(m1)
    m2=MaxPooling2D(128,(2,2),padding='same')(c2)

    # 3rd BLOCK
    c3=Conv2D(128,(4,4),activation='relu')(m2)
    m3=MaxPooling2D(64,(2,2),padding='same')(c3)

    # now it's gonna be a convolutional +relu of 256 filters/units and 4*4
    # then aft that adding the fully connected layer+ sigmoid 
    c4=Conv2D(256,(4,4),activation='relu')(m3)
    f1=Flatten()(c4) #flattening all of the elements
    d1=Dense(4096,activation='sigmoid')(f1)

    return Model(inputs=inp,outputs=d1,name='embedding')

embedding=make_embedding() 
embedding.summary() #this will give us the compelete summary of our model that is named as the variable mod.


















