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

c4=Conv2D(256,(4,4),activation='relu')(m3)
f1=Flatten()(c4) #flattening all of the elements
d1=Dense(4096,activation='sigmoid')(f1)

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

# now we are going have 2 segeggated parts out of which one is going to be the anchors and another will be either positive or negative
# L1 siamese dist. basically adds them tog. but actually gotta be subtract them

# Siamese L1 distance class
class L1Dist(Layer):
    # init methord -inheritence
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

# magic happens here -similarity calculation
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding) #this gonna return the absolute value

# declaring our input_image
input_image=Input(name='input_img',shape=(100,100,3))

# declaring our validation_image as well
validation_image=Input(name='validation_img',shape=(100,100,3))

# Now about to make the make_siamese_model
def make_siamese_model():
    #Anchor image input in the network
    input_image=Input(name='input_img',shape=(100,100,3))
    # validation image in the network
    validation_image=Input(name='validation_img',shape=(100,100,3))
    # now we gonna take these raw  input images and pass them through our embedding model
    # Combine siamese distance components
    siamese_layer=L1Dist()
    siamese_layer._name='distance'
    distances=siamese_layer(embedding(input_image),embedding(validation_image))
    
    #Now checking whether the embeddings are similar or not.
    classifier=Dense(1,activation='sigmoid')(distances) #combining the distances with the sigmoid activation

inp_embedding=embedding(input_image)
# now we are printing the input image status
print(inp_embedding)

val_embedding=embedding(validation_image)
# printing the validation iamge status
print(val_embedding)

# now taking our siamese layer
siamese_layer=L1Dist()
print(siamese_layer(inp_embedding,val_embedding))

# defining distances outside the function
distances=siamese_layer(embedding(input_image),embedding(validation_image))

# defining classifier outside the function
classifier=Dense(1,activation='sigmoid')(distances) 
print(classifier)

# Lastly printing our siamese network
siamese_network= Model(inputs=[input_image,validation_image],outputs=classifier,name='SiameseNetwork')
print(siamese_network)

# then getting the summary of our siamese network
siamese_network.summary()

# even we can do
siamese_model=make_siamese_model()
siamese_model.summary() #this also gonna give the same thing likewise siamese_network did

# Starting training the model
# Setup the Loss function
# Set up the optimizer  #gonna help us to backpropogate through our neural network
# Build the custom Training step
# Create the training loop
# Train the MODEL
binary_cross_loss=tf.losses.BinaryCrossentropy()
# find the optimizer ~adam optimizer
opt=tf.keras.optimizers.Adam(1e-4)

checkpoint_dir=r'C:\Users\hp\Desktop\MODEL\training_checkpoints'
checkpoint_prefix=os.path.join(checkpoint_dir,'ckpt') #this ckpt means that we'll have all our checkpoints in a consistent  format
checkpoint=tf.train.Checkpoint(opt=opt,siamese_model=siamese_model)

# Build the train step function used to train 1 batch of our data
# make prediction
# calculate loss
# Derive the gradiant
# Calculate new weighs and apply

# Now calculating the 1st batch
test_batch=train_data.as_numpy_iterator()
batch_1=test_batch.next()
print(batch_1)

# then printing the length of the batch of the anchor, positive/negative, label
print(len(batch_1)) #1st component gonna be anchor image

print("the batch of anchor images\n:",batch_1[0]) #representation of the anchor images

# Doing the same thing for the anchor and the negative/positive images
print(batch_1[1])

# Then for the lables
print(batch_1[2])

# Now grabbing our features (X)
X=batch_1[:2] #we stored batch_1 in variable X
print(X)

# Now printing the length of the features
print(X)
# now printing the shape of X 
X_shape=np.array(X).shape

# Now grabbing our labels
y=batch_1[2]
print(y)

@tf.function #this compiles the function into a callable tensorflow graph
def train_step(batch):
    #recording all the operations here
    with tf.GradiantTape() as tape:
        X=batch[:2]
        y=batch[2]
        # Forward pass
        # The predicted outcome comes here
        y_hat=siamese_model(X,training=True) #making prediction, and setting training =True mandotory

        # calculating loss
        loss=binary_cross_loss(y,y_hat)
    print(loss)
    # calculating the gradiants
    grad=tape.gradient(loss,siamese_model.trainable_variables)

    # calculate updated weights and apply to siamese model
    opt.apply_gradients(zip(grad,siamese_model.trainable_variables))

    # return the loss
    return loss

# Building our training loop
#  while the train function is focused on training for one batch, the loop here will be used to iterate over each and evaery batch in the dataset.
def train(data,val_data,EPOCHS):
    # loop through the epochs
    for epoch in range(1,EPOCHS+1):
        print('\n Epoch{}/{}'.format(epoch,EPOCHS))
        progbar=tf.keras.utils.Progbar(len(data)) #defining the progreass bar

        # Creating metrics object
        r=Recall()
        p=Precision()

        # loop through each batch
        for idx,batch in enumerate(data):
            # Run train step here
            loss=train_stepy_hat=siamese_model.predict(batch[:2])
            r.update_state(batch[2],y_hat)
            p.update_state(batch[2],y_hat)
            progbar.update(idx+1)
        print(loss.numpy(),r.result().numpy(),p.result().numpy())

        # Evaluate less frequently ,e.g, every 5 epochs
        if epoch%5==0:
           evaluate_model(val_data)

        #    Disabling checkpoint saving for quick testing
        if epoch% 10==0: #save checkpoints at every 10 epochs
           checkpoint.save(file_prefix=checkpoint_prefix)

# Now evaluating our model
# Evaluate performance
# Import metric calculations ,here we'll use precision and recall( these are the metrices in tensorflow)
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy #type:ignore
# Precision demonstrates what proportion of the positives were actually correct.
# Recall shows the actual proportion of actual positives were identified correctly.

# Get a batch of test data
test_data_iterator = test_data.as_numpy_iterator()

# Fetch the first batch
test_var = test_data_iterator.next()

# Unpack the batch into input, validation, and labels
test_input, test_val, y_true = test_var  #here we unpacked the values , here y_true is effectively our LABELS
# displaying the batch
print("The array of test_var:\n",test_var) 

# dispalying the length of the batch
print("This is the length of the outer structure of test_var:\n",len(test_var)) #3 

# ~one is input images,validation images, labels
# printing the array of input images
print("This gives the array of the input images means the anchor:\n",test_var[0])

# length of input iamges
print("Length of the input images:\n",len(test_var[0])) #16

# Now doing the same for the 2nd example that is either negative/positive ~validation
print(test_var[1])
print(len(test_var[1])) #again 16

# now getting the labels
print("This is the array of the labels in test_var:", test_var[2])

# this also does means that our test_input=test_var[0], test_val=test_var[1], 

# now checking whether we enemurated right or not
# Verify that the unpacked variables match the elements in test_var
print("Verification:")
print("test_input == test_var[0]:", np.array_equal(test_input, test_var[0]))
print("test_val == test_var[1]:", np.array_equal(test_val, test_var[1]))
print("y_true == test_var[2]:", np.array_equal(y_true, test_var[2]))

def evaluate_model(val_data):
    precision=tf.keras.mertics.Precision()
    recall=tf.keras.Recall()
    binary_accuracy = tf.keras.metrics.BinaryAccuracy()

    
    for val_batch in val_data:
        val_X, val_y = val_batch[:2], val_batch[2]
        val_y_hat = siamese_model(val_X, training=False)
        precision.update_state(val_y, val_y_hat)
        recall.update_state(val_y, val_y_hat)
        binary_accuracy.update_state(val_y, val_y_hat)





















