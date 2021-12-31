'''
Created on 12-Nov-2019

@author: Ravi
''' 
import os
import tensorflow as tf
import glob
from keras.optimizers import adam
from keras.backend.tensorflow_backend import set_session
import numpy as np
import pandas as pd
import cv2
# from losses import binary_focal_loss  
import random
import gc
from keras.callbacks import TensorBoard
import keras
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import sgd
from keras.applications import resnet50
from matplotlib import pyplot as plt

import keras.backend as K
K.set_image_data_format('channels_last')
 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
np.random.seed(1337)  # for reproducibility

#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.95
#set_session(tf.Session(config=config))
  
nrows = 380
ncolumns = 380  
channels = 3

train_dir = '/mnt/komal/Sandesh/RFMID/Evaluation_Set/'
     
path1 = sorted(glob.glob('/mnt/komal/Sandesh/RFMID/Evaluation_Set/' + '*' + '.png'))   
# path2 = sorted(glob.glob('/mnt/X/Ravi K/RIDD/Training_Set/processed/2_class/cropped/Augment/test_set/Normal/' + '*' + '.png'))   


# #  
test_images = path1 
print(len(test_images))

# random.shuffle(test_images)

out_path = '/mnt/komal/Sandesh/RFMID/'
# out_path = '/mnt/x/Ravi K/Other/ISBI_2019_challenge/iChallenge_AMD/results/prediction/'
# os.makedirs(path2write)

modelPath = '/mnt/komal/Sandesh/RFMID/Training_Set/RFMID1.h5'
# modelPath = '/mnt/X/Aman/Opthalmology/cropped and resized data/380/model_norm/effic_B4_noisy_new1.131-0.882331-0.840764.h5'

imheight = 380
imwidth = 380
imdepth = 3
data_shape = imheight * imwidth
classes = 28
data_format='.png'

dimen= 380
r=dimen//2

img_rows, img_cols = 512, 512
img_channels = 3 

#img_dim = (img_channels, img_rows, img_cols) if K.image_dim_ordering() == "th" else (img_rows, img_cols, img_channels)
img_dim = (imwidth,imheight,imdepth)
def swish_activation(x): 
    return x * K.sigmoid(x)
 
import efficientnet.keras as efn 

img_dim = (380,380,3)
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
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#opt = adam(lr=1e-5)
#model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])
#print(model.summary())
####################################################################
 
input_shape=(nrows, ncolumns,channels)

model.load_weights(modelPath)
model.summary()
dimen= 380
r = (dimen//2)

preds = []
fns = []
Y_pred = []
Y_prob = []
target_size = (380, 380)

X = [] # images
y = [] # labels
clas_prob=[]
cls_lbl = []

resized_coordinates = {'Id': [],
                       'AH': [],'AION': [],'ARMD': [],'BRVO': [],'CRS': [],
                       'CRVO': [],'CSR': [],'DN': [],'DR': [],'EDN': [],
                       'ERM': [],'LS': [],'MH': [],'MHL': [],'MS': [],
                       'MYA': [],'ODC': [],'ODE': [],'ODP': [],'OTHER': [],
                       'PT': [],'RP': [],'RPEC': [],'RS': [],'RT': [],
                       'ST': [],'TSLN': [],'TV': [],
                       }



num_classes = 28

preds = []
fns = []
Y_pred = []
Y_prob = []

X = [] # images
y = [] # labels
clas_prob=[]
cls_lbl = []

data_path = r'/mnt/komal/Sandesh/RFMID/Evaluation_Set/'
filenames = sorted(glob.glob(data_path+"/*.png"))  

      
for i in range(len(filenames)):
    MK = []   

    tileno = str(i+1)

    tdp = data_path+tileno+'.png'
    
    inp = cv2.imread(tdp, cv2.IMREAD_COLOR)
    inp = cv2.resize(inp, (380, 380),interpolation=cv2.INTER_NEAREST)
#    inp = (inp - inp.mean(axis=(0,1))) / (inp.std(axis=(0,1)))
    
#     tilename = image.split('/')
#     tileNum = tilename[-1]
    print(tileno+'.png')
    
    resized_coordinates['Id'].append(tileno)
    
    img1 = inp/255

    MK.append(((inp)))
    Test_image = np.array(MK)
    
    pred = model.predict(Test_image) 

#     clas_prob.append(pred[0][0])
#     resized_coordinates['Disease Risk'].append("{0:0.1f}".format(pred[0][0])
    
    clas_prob.append(pred[0][0])
#     clas_prob.append((1-pred[0][0]))
    resized_coordinates['AH'].append("{0:0.3f}".format(pred[0][0]))
    resized_coordinates['AION'].append("{0:0.3f}".format(pred[0][1]))
    resized_coordinates['ARMD'].append("{0:0.3f}".format(pred[0][2]))
    resized_coordinates['BRVO'].append("{0:0.3f}".format(pred[0][3]))
    resized_coordinates['CRS'].append("{0:0.3f}".format(pred[0][4]))
    resized_coordinates['CRVO'].append("{0:0.3f}".format(pred[0][5]))
    resized_coordinates['CSR'].append("{0:0.3f}".format(pred[0][6]))
    resized_coordinates['DN'].append("{0:0.3f}".format(pred[0][7]))
    resized_coordinates['DR'].append("{0:0.3f}".format(pred[0][8]))
    resized_coordinates['EDN'].append("{0:0.3f}".format(pred[0][9]))
    resized_coordinates['ERM'].append("{0:0.3f}".format(pred[0][10]))
    resized_coordinates['LS'].append("{0:0.3f}".format(pred[0][11]))
    resized_coordinates['MH'].append("{0:0.3f}".format(pred[0][12]))
    resized_coordinates['MHL'].append("{0:0.3f}".format(pred[0][13]))
    resized_coordinates['MS'].append("{0:0.3f}".format(pred[0][14]))
    resized_coordinates['MYA'].append("{0:0.3f}".format(pred[0][15]))
    resized_coordinates['ODC'].append("{0:0.3f}".format(pred[0][16]))
    resized_coordinates['ODE'].append("{0:0.3f}".format(pred[0][17]))
    resized_coordinates['ODP'].append("{0:0.3f}".format(pred[0][18]))
    resized_coordinates['OTHER'].append("{0:0.3f}".format(pred[0][19]))
    resized_coordinates['PT'].append("{0:0.3f}".format(pred[0][20]))
    resized_coordinates['RP'].append("{0:0.3f}".format(pred[0][21]))
    resized_coordinates['RPEC'].append("{0:0.3f}".format(pred[0][22]))
    resized_coordinates['RS'].append("{0:0.3f}".format(pred[0][23]))
    resized_coordinates['RT'].append("{0:0.3f}".format(pred[0][24]))
    resized_coordinates['ST'].append("{0:0.3f}".format(pred[0][25]))
    resized_coordinates['TSLN'].append("{0:0.3f}".format(pred[0][26]))
    resized_coordinates['TV'].append("{0:0.3f}".format(pred[0][27]))

 
coords_df = pd.DataFrame(data=resized_coordinates)
coords_df.to_csv('/mnt/komal/Sandesh/RFMID/Disease_classification_results_RIDD_multi_sandesh.csv')
# 



# from sklearn.metrics import classification_report 
# from sklearn.metrics import confusion_matrix 
# from sklearn.metrics import accuracy_score 
# # 
# actual = Y_prob
# predicted = Y_pred  
# results = confusion_matrix(actual, predicted) 
# print('Confusion Matrix :')
# print(results) 
# print('Accuracy Score :',accuracy_score(actual, predicted)) 
# print('Report : ')
# print(classification_report(actual, predicted))


