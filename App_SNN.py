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


















