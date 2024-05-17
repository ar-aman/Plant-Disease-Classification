import streamlit as st
import tensorflow as tf
from tensorflow import keras
from keras import models, layers, Sequential
import matplotlib.pyplot as plt
import numpy as np

# introduction

st.set_page_config(layout='wide')
st.title('POTATO DISEASE DETECTION')

st.subheader('This classification model classifies the image of a potato leaf into three categories: ')

st.write('**1. Early Blight (Alternaria solani)** ')
st.write('**2. Late Blight (Phytophthora infestans)** ')
st.write('**3. Healthy Leaf** ')

# dependencies
IMAGE_SIZE = 256
BATCH_SIZE = 32
CHANNELS = 3
n_classes = 3
EPOCHS = 50
class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

def resize_rescale(image):
    resize_and_rescale = tf.keras.Sequential([
    layers.Resizing(IMAGE_SIZE, IMAGE_SIZE),
    layers.Rescaling(1.0/255)
])
    return resize_and_rescale 

def predict(model, img):
#   img_array = tf.keras.preprocessing.image.img_to_array(img[i].numpy())
#   img_array = tf.expand_dims(img_array, 0) #create a batch

#   predictions = model.predict(img_array)

#   predicted_class = class_names[np.argmax(predictions[0])]
    return #predicted_class

# user input
with st.expander('**DETECT DISEASE**'):
    img = st.file_uploader('upload image')

    if st.button('Detect'):
        # process image
        img = resize_rescale(img)
        img_batch = np.expand_dims(img,0)
        st.write('preprocesing complete')

        # load model
        # model = models.load_model('xyz.h5')

        # predict
        # result = predict(model, img_batch)
        # st.write('Classification: ', result)
        




