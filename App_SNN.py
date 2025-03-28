# Here the code for app will be written here 
# We'ld be using 4 different kinds of widgets that are often being used in the app developement.

# importing kivy dependencies
from kivy.app import App
from kivy.uix.boxLayout import BoxLayout

# Import kivy UX components
from kivy.uix.image import Image 
from kivy.uix.button import Button
from kivy.uix.label import Label

# Import oth kivy dependencies
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger

# iMPORT other dependencies
import cv2
import tensorflow as tf
from layers import L1Dist
import os
import numpy as np

# Build app and layout
def build(self): #inherient function which we usually use in KIVY
    self.web_cam=Image(size_hint=(1,.8)) #these three are core UX components Image,Button,label
    self.button=Button(text="Verify",on_press=self.verify,size_hint=(1,.1))
    self.verification_label=Label(text="Verification Uninitiated",size_hint=(1,.1))
    
    # Add items to the Layout
    layout=BoxLayout(orientation='vertical') #here the sequence of 
    # the image, label,button are set sequentilly ina vertical manner ,top to bottom
    layout.add_widget(self.web_cam)
    layout.add_widget(self.button)
    layout.add_widget(self.verification_label)

    # Set up the capture using OpenCV
    for i in range(10):  # Test indices from 0 to 9
        self.capture = cv2.VideoCapture(i)
        if self.capture.isOpened():
            print(f"Camera found at index {i}")
            break
        else:
            print("No camera found")
            return layout
        
        # Check if the model file exists
        model_path = 'siamesemodel2.keras'
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            return layout

        # Load the TensorFlow/Keras model
        self.model = tf.keras.models.load_model(model_path, custom_objects={'L1Dist': L1Dist})

        Clock.schedule_interval(self.update, 1.0 / 33.0)  # Update at 30 FPS
        return layout

    # Run continuously to get webcam feed
    def update(self, *args):
        ret, frame = self.capture.read()
        if not ret:
            print("Failed to read frame. Exiting...")
            return
        
        # Resize the frame to 250x250 pixels
        resized_frame = cv2.resize(frame, (250, 250))

         # Convert it to texture
        buf = cv2.flip(resized_frame, 0).tostring()
        image_texture = Texture.create(size=(resized_frame.shape[1], resized_frame.shape[0]), colorfmt='bgr')
        image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam.texture = image_texture

    # Load image from the file and convert to 100*100 px
    def preprocess(self,file_path):











