# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 09:22:35 2020

@author: MaryamHashemi
"""

import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
import cv2
import math
#import tensorflow as tf
#from tensorflow.python.client import device_lib 
#from keras.preprocessing import image
from keras.utils import np_utils
from skimage.transform import resize
import glob
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, InputLayer, Dropout, Activation
from keras.utils import plot_model
from sklearn.metrics import classification_report
from keras.layers.convolutional import Convolution2D, MaxPooling2D, MaxPooling1D, Convolution1D
from keras.applications.vgg16 import preprocess_input
from keras import losses
#oc curve and auc
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from keras.models import model_from_json
from sklearn.metrics import average_precision_score

from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.layers.recurrent import LSTM


videos_path1="E:/data/YawDD/YawDD dataset/Mirror-Table1/Female-Normal/"
videos_path2="E:/data/YawDD/YawDD dataset/Mirror-Table1/Female-Yawing/"
videos_path3="E:/data/YawDD/YawDD dataset/Mirror-Table1/Male-Normal/"
videos_path4="E:/data/YawDD/YawDD dataset/Mirror-Table1/Male-Yawing/"

video1=glob.glob(videos_path1+"*.bmp")
video2=glob.glob(videos_path2+"*.bmp")
video3=glob.glob(videos_path3+"*.bmp")
video4=glob.glob(videos_path4+"*.bmp")

videos=[video1,video2,video3,video4]




X_train =[]     # creating an empty array
X_valid=[]
y_train=[]
y_valid=[]
X_test=[]
y_test=[]
index=0
videocount=0
for i in videos:
    index+=1
    countframe=0
    if index%2==0:
        countlabel=1
    else:
        countlabel=0
    for j in i:
        videocount+=1
        countframe+=1
        frame=cv2.imread(j)
#        frame= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if 126<videocount<314 or 1562<videocount<1725 or 2884<videocount<3071 or 4163<videocount<4345:
            X_valid.append(frame)
            y_valid.append(countlabel)
            
        elif videocount<126 or 1397<videocount<1562 or 2679<videocount<2884 or 3981<videocount<4163:
            X_test.append(frame)
            y_test.append(countlabel)
            
        else:
            X_train.append(frame)
            y_train.append(countlabel)


X_valid = np.array(X_valid)    # converting list to array
X_train=np.array(X_train)
X_test=np.array(X_test)

dummy_y_train = np_utils.to_categorical(y_train)    # one hot encoding Classes
dummy_y_valid = np_utils.to_categorical(y_valid) 
dummy_y_test = np_utils.to_categorical(y_test)   # one hot encoding Classes

        
images_train = []
for i in range(0,X_train.shape[0]):
    a = resize(X_train[i], preserve_range=True, output_shape=(64,64,3)).astype(int)      # reshaping to 224*224*3
    images_train.append(a)
X_train = np.array(images_train)

images_valid = []
for i in range(0,X_valid.shape[0]):
    a = resize(X_valid[i], preserve_range=True, output_shape=(64,64,3)).astype(int)      # reshaping to 224*224*3
    images_valid.append(a)
X_valid = np.array(images_valid)       

images_test = []
for i in range(0,X_test.shape[0]):
    a = resize(X_test[i], preserve_range=True, output_shape=(64,64,3)).astype(int)      # reshaping to 224*224*3
    images_test.append(a)
X_test= np.array(images_test) 


X_train = preprocess_input(X_train, mode='tf')      # preprocessing the input data
X_valid = preprocess_input(X_valid, mode='tf')
X_test = preprocess_input(X_test, mode='tf')      # preprocessing the input data


del a,countlabel,images_train,images_valid,index,j,video1,video2,video3,video4,videos_path1,videos_path2,videos_path3,videos_path4,videocount,y_train,y_valid,images_test,y_test



#X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],3,1)
#X_valid=X_valid.reshape(X_valid.shape[0],X_valid.shape[1],X_valid.shape[2],3,1)
#X_test=X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],3,1)
from keras.layers import Reshape

modellstm = Sequential()
modellstm.add(Convolution2D(32, (3, 3), padding='same',
                        input_shape=(64,64,3)))
modellstm.add(Activation('relu'))
modellstm.add(Convolution2D(64, (3, 3), data_format='channels_first'),)
modellstm.add(Activation('relu'))
modellstm.add(Convolution2D(128, (3, 3), data_format='channels_first'),)
modellstm.add(Activation('relu'))
modellstm.add(MaxPooling2D(pool_size=(4, 4)))
modellstm.add(Dropout(0.25))
modellstm.add(MaxPooling2D(pool_size=(4, 4)))
#modellstm.add(Flatten())
#modellstm.add(Dense(128))
#modellstm.add(Activation('relu'))
#modellstm.add(Dropout(0.5))
modellstm.summary()
#modellstm.add(LSTM(32, return_sequences=True, input_shape=(4096,1))) 
modellstm.add(Reshape((8*3*28,1)))
modellstm.add(LSTM(32, return_sequences=True, input_shape=(8*3*28,1))) # returns a sequence of vectors of dimension 32
modellstm.add(LSTM(32,return_sequences=True))  # returns a sequence of vectors of dimension 32
modellstm.add(Flatten())
modellstm.add(Dense(512))
modellstm.add(Activation('relu'))
modellstm.add(Dropout(0.5))
modellstm.add(Dense(2))
modellstm.add(Activation('sigmoid'))
modellstm.summary()




sgd = SGD(lr=0.01,decay=1e-6, momentum=0.9, nesterov=True)

modellstm.compile(loss='categorical_crossentropy', optimizer=sgd , metrics=['accuracy'])



history2=modellstm.fit(X_train, dummy_y_train, batch_size=20,epochs=20, validation_data=(X_valid, dummy_y_valid))

scorevalid2 = modellstm.evaluate(X_valid, dummy_y_valid, batch_size=32)
scoretest2 = modellstm.evaluate(X_test, dummy_y_test, batch_size=32)


print('Valid score:', scorevalid2[0])
print('Valid accuracy:', scorevalid2[1])

print('test score:', scoretest2[0])
print('test accuracy:', scoretest2[1])
# Get training and test loss histories
training_loss = history2.history['loss']
test_loss = history2.history['val_loss']

# Create count of the number of epochs
epoch_count = range(1, len(training_loss) + 1)

# Visualize loss history
plt.plot(epoch_count, training_loss, 'r--')
plt.plot(epoch_count, test_loss, 'b-')
plt.legend(['Training Loss', 'Test Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show();

plt.figure()
plt.plot(history2.history['acc'])
plt.plot(history2.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Save the weights
model.save_weights('model_weights_CNN-LMST_yawing.h5')
# Save the model architecture
with open('model_architecture_CNN-LMST_yawing.json', 'w') as f:
    f.write(model.to_json())
#
modellstm.save_weights('model_weights_CNN-LMST_yawing2.h5')
# Save the model architecture
with open('model_architecture_CNN-LMST_yawing2.json', 'w') as m:
    m.write(model.to_json())
#end=datetime.datetime.now()
#elapsed=end-start
plot_model(model, to_file='CNN-LMST_yawing.png')
plot_model(modellstm, to_file='CNN-LMST_yawing2.png')
#print('training time',str(elapsed))

probs1 = modellstm.predict(X_valid)
# keep probabilities for the positive outcome only
#probs = probs[:, 1]
# calculate AUC
auc1 = roc_auc_score(dummy_y_valid, probs1)
print('AUC1 (validation): %.3f' % auc1)
# calculate roc curve
#score2 = model.evaluate(X_test, dummy_y_test, batch_size=32)
#print("Test accuracy:",score2[1])
#
probs2 = modellstm.predict(X_test)
auc2 = roc_auc_score(dummy_y_test, probs2)
print('AUC2 (Test): %.3f' % auc2)
# keep probabilities for the positive outcome only
#probs = probs[:, 1]
# calculate AUC


def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i,0]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i,0]!=y_hat[i]:
           FP += 1
        if y_actual[i,0]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i,0]!=y_hat[i]:
           FN += 1
    Recal=(TP)/(TP+FN) 
    Presicion=(TP)/(TP+FP)      
    return(TP, FP, TN, FN,"recall=",Recal,"presicion=",Presicion)
    
    
probs=[]    
for p in range (0,716):
    if probs1[p,0]>0.5:
        probs.append(1)
    else:
        probs.append(0)
            
    
perf_measure(dummy_y_valid,probs )


probstest=[]    
for p in range (0,674):
    if probs2[p,0]>0.5:
        probstest.append(1)
    else:
        probstest.append(0)
            
    
perf_measure(dummy_y_test,probstest )
