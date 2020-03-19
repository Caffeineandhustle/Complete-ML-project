#!/usr/bin/env python
# coding: utf-8

# # Image Recognition Classifier using CNN, Keras and Tensorflow backend

# In[3]:


#Importing Libraries and Splitting the Dataset
#After importing the libraries, we need to split our data into two parts- taining_set and test_set.
#In our case, the dataset is already split into two parts. 
#The training set has 4000 image each of dogs and cats while the test set has 1000 images of each.


from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import warnings
warnings.filterwarnings('ignore')


# In[18]:


#Initialize the CNN

classifier=Sequential()

#Convolution : to extract features from the input image. Convolution preserves the spatial 
#relationship between pixels by learning image features using small squares of input data.

classifier.add(Convolution2D(32,3,3,input_shape=(64,64,3),activation="relu"))
classifier.add(Convolution2D(20,kernel_size=(3,3),activation="relu",strides=2))


# In[19]:


#Pooling : Pooling (also called subsampling or downsampling) reduces 
#the dimensionality of each feature map but retains the most important information.

classifier.add(MaxPooling2D(pool_size=(2,2)))


# In[20]:


#Flattening : the matrix is converted into a linear array so that to input it into the nodes of our neural network.

classifier.add(Flatten())


# In[21]:


# Full Connection : Full connection is connecting our convolutional network to a 
#neural network and then compiling our network.
classifier.add(Dense(output_dim=128,activation="relu"))
classifier.add(Dense(output_dim=1,activation="sigmoid"))

#Compile : Here we have made 2 layer neural network with a sigmoid function as an activation function for the 
#last layer as we need to find the probability of the object being a cat or a dog.

classifier.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy'])


# In[22]:


# Fitting the CNN to the images

from keras_preprocessing.image import ImageDataGenerator

train_datagen=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
test_datagen= ImageDataGenerator(rescale=1./255)


# In[23]:


training_set=train_datagen.flow_from_directory('/Users/priyeshkucchu/Desktop/dataset/training_set/',                                               target_size=(64,64),batch_size=32,class_mode='binary')

test_set=test_datagen.flow_from_directory('/Users/priyeshkucchu/Desktop/dataset/test_set/',                                               target_size=(64,64),batch_size=32,class_mode='binary')


# In[24]:


#Training our network

from IPython.display import display
from PIL import Image

classifier.fit_generator(training_set,steps_per_epoch=8000,epochs=2,validation_data=test_set,validation_steps=800)


# In[28]:


# Test with a random image. Result should be dog.

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('/Users/priyeshkucchu/Desktop/RImage.jpeg',target_size=(64,64))
test_image=image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis=0)
result = classifier.predict(test_image)
training_set.class_indices

if result[0][0]>0.5:
    prediction="dog"
else:
    prediction="cat"
print(prediction)


# In[26]:


# Test with a random image. Result should be cat. 

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('/Users/priyeshkucchu/Desktop/ka.jpg',target_size=(64,64))
test_image=image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis=0)
result = classifier.predict(test_image)
training_set.class_indices

if result[0][0]>0.5:
    prediction="dog"
else:
    prediction="cat"
print(prediction)


# In[ ]:




