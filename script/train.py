# -*- coding: utf-8 -*-
"""RMFID Challange.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1g761V2kal48qsQZwkizKwzooErzUcuEx
"""

#from google.colab import drive
#drive.mount('/content/drive')

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/drive/MyDrive/RFMID Challange/

# Commented out IPython magic to ensure Python compatibility.
# %ls



import os

import pandas as pd
#for GPU Allocation
os.environ["CUDA_VISIBLE_DEVICES"]="1"
#from tensorflow.keras.applications import EfficientNetB7 as eff


df=pd.read_csv('RFMiD_Training_Labels.csv')
multi_labels =  df.columns[2:]
print(multi_labels)


#print(len(filename))

for j in range(1,1921):
 df['ID'] = df['ID'].replace({j: str(j) +'.png'}) # Replace  filename 1 >> 1.png
  
df.to_csv("RFMiD_Training_Labels.csv", index=False) 
  
print(df.head) 
#filename = df["ID"]
#filename[0]


import pandas as pd
from tqdm import tqdm 
import numpy as np
import os
path =  os.getcwd()


# example of converting an image with the Keras API
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
from keras import datasets, layers, models, Model
from keras.layers import Conv2D,MaxPooling2D,Dropout, Dense, Flatten,GlobalAveragePooling2D
import matplotlib.pyplot as plt


df=pd.read_csv('RFMiD_Training_Labels.csv')
df.columns
df.shape
filename= df.columns[0]

'''
train_image = []
for i in tqdm(range(df.shape[0])):
    img = load_img(path + '/Training/'+ str(df['ID'][i])+'.png',target_size=(224,224,3))
    img = img_to_array(img)
    img = img/255
    train_image.append(img)
X = np.array(train_image)

X.shape


plt.imshow(X[2])

y = np.array(df.drop(['ID', 'Disease_Risk'],axis=1))
print(y.shape)
print(y[11])

#from keras.utils import np_utils
#Y = np_utils.to_categorical(labels_train, num_classes)

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=42, test_size=0.1)

#!pip install efficientnet

'''

print(filename)

#___________________________________________________________________________
import pandas as pd

from tensorflow.keras.preprocessing.image import ImageDataGenerator
batch_size=64

train_datagen = ImageDataGenerator(rescale=1./255,  
                                   #validation_split=0.2,
                                    rotation_range=360,
                                    width_shift_range=0.5,
                                    height_shift_range=0.4,
                                    #width_shift_range=[-150,150]
                                    shear_range=0.2,
                                    zoom_range=[0.5,1.0],
                                    #zoom_range=0.2,
                                    horizontal_flip=True,
                                    vertical_flip=True,
                                    brightness_range=[0.1,1.0],
                                    #featurewise_center=True,
                                    #featurewise_std_normalization=True,
                                    #zca_whitening=True,
                                    fill_mode="nearest"
                                   )



val_datagen = ImageDataGenerator(rescale=1./255)

train_generator=train_datagen.flow_from_dataframe(
dataframe=df[:1500],
directory="./Training/",
x_col= 'ID',
y_col=multi_labels,
batch_size=32,
seed=42,
shuffle=True,
class_mode="raw",
target_size=(224,224))


valid_generator=val_datagen.flow_from_dataframe(
dataframe=df[1501:len(filename)],
directory="./Training/",
x_col="ID",
y_col=multi_labels,
batch_size=32,
seed=42,
shuffle=True,
class_mode="raw",
target_size=(224,224))

#___________________________________________________________________________

import keras.backend as K


def swish_activation(x):
    return x * K.sigmoid(x)


import tensorflow as tf

#from tensorflow.keras.applications import EfficientNetB7
#import efficientnet.keras as efn 
#import efficientnet.tfkeras as efn
#from tensorflow.keras.applications.efficientnet import EfficientNetB4

#from keras_efficientnets import EfficientNetB4, EfficientNetB1
#import keras
#keras.applications.EfficientNetB0
#import efficientnet as efn 

import efficientnet.keras as efn 

img_dim = (224,224,3)
base_model = efn.EfficientNetB4(input_shape=img_dim, weights='imagenet', include_top=False)  # or weights='noisy-student'
 
# from keras_efficientnets import EfficientNetB7 
# base_model = EfficientNetB7(input_shape=img_dim, weights='imagenet', include_top=False) 

x = base_model.output  

x = GlobalAveragePooling2D()(x)
x = Dense(512, activation=swish_activation, name='fc1')(x)
x = Dropout(0.3)(x)
x = Dense(256, activation=swish_activation, name='fc2')(x)
x = Dropout(0.3)(x)
x = Dense(128, activation=swish_activation, name='fc3')(x)
x = Dropout(0.3)(x)

predictions = Dense(28, activation='sigmoid', name='predictions')(x)
# add the top layer block to your base model
model = Model(base_model.input, predictions)

model.summary()

def create_model():
    conv_base = efn.EfficientNetB4(include_top = False, weights = 'imagenet',
                               input_shape = (224, 224, 3))
    model = conv_base.output
    model = layers.GlobalAveragePooling2D()(model)
    model = layers.Dense(5, activation = "softmax")(model)
    model = models.Model(conv_base.input, model)

    model.compile(optimizer = Adam(lr = 0.001),
                  loss = 'binary_crossentropy',
                  metrics = ["accuracy"])
    return model


#model = create_model()
#model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), batch_size=64)

# to train model with Augmentation generator use below code
'''
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,   #Scale the image between 0 and 1
                                   #validation_split=0.2,
                                    rotation_range=360,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    vertical_flip=True,
                                    brightness_range=[0.1,1.0],
                                  #featurewise_center=True, 
                                   #featurewise_std_normalization=True,
                                   #zca_whitening=True,
                                    fill_mode="nearest"
                                   )

val_datagen = ImageDataGenerator(rescale=1./255)  #We do not augment validation data. we only perform rescale
batch_size=16

train_datagen.fit(X_train)
train_datagen.fit(X_val)

#Create the image generators
#train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
#val_generator = train_datagen.flow(X_val, y_val, batch_size=batch_size)
#test_generator = val_datagen.flow(X_test, y_test, batch_size=batch_size) 


filepath = "DW_M_MobileNet_gen1.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
                             save_best_only=True, mode='min')

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5,
                                   verbose=1, mode='min', min_lr=0.00000000001)
earlystop = EarlyStopping(patience=8, verbose=1)
csv_logger = CSVLogger("model_history_log_DW_M.csv", append=True)

#callbacks_list = [checkpoint, reduce_lr]

callbacks_list = [checkpoint, reduce_lr, csv_logger]
model.load_weights('DW_M_MobileNet_gen1.h5') 

history = model.fit_generator(train_datagen.flow(X_train, y_train, batch_size=32), steps_per_epoch=len(X_train) / 16,
                    validation_data=train_datagen.flow(X_val, y_val, batch_size=32),
                    validation_steps=len(X_val) / 16,
                    epochs=200, verbose=1,
                   #callbacks=callbacks_list
                   )
'''
history =model.fit_generator(train_generator,
                             steps_per_epoch= train_generator.n//train_generator.batch_size,
                            epochs=1,verbose=1,
                              validation_data=val_generator,
                              validation_steps=valid_generator.n//valid_generator.batch_size,
                              callbacks=callbacks_list)
                

'''
import pandas as pd

from tensorflow.keras.preprocessing.image import ImageDataGenerator
batch_size=64

train_datagen = ImageDataGenerator(rescale=1./255,  
                                   #validation_split=0.2,
                                    rotation_range=360,
                                    width_shift_range=0.5,
                                    height_shift_range=0.4,
                                    #width_shift_range=[-150,150]
                                    shear_range=0.2,
                                    zoom_range=[0.5,1.0],
                                    #zoom_range=0.2,
                                    horizontal_flip=True,
                                    vertical_flip=True,
                                    brightness_range=[0.1,1.0],
                                    #featurewise_center=True,
                                    #featurewise_std_normalization=True,
                                    #zca_whitening=True,
                                    fill_mode="nearest"
                                   )



val_datagen = ImageDataGenerator(rescale=1./255)

train_generator=train_datagen.flow_from_dataframe(
dataframe=df[:1500],
#directory="./Training",
x_col= 'ID',
y_col=multi_labels,
batch_size=32,
seed=42,
shuffle=True,
class_mode="raw",
target_size=(224,224))


valid_generator=val_datagen.flow_from_dataframe(
dataframe=df[1501:len(filename)],
directory="./Training",
x_col="ID",
y_col=multi_labels,
batch_size=32,
seed=42,
shuffle=True,
class_mode="raw",
target_size=(224,224))
'''
