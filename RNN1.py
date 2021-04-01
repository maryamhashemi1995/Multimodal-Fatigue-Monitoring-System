# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 09:22:35 2020

@author: MaryamHashemi
"""

import cv2
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.applications.vgg16 import preprocess_input
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D, MaxPooling2D


images_path1="C:/Users/MaryamHashemi/Desktop/Review/Codes/sequence images-close/"
images_path2="C:/Users/MaryamHashemi/Desktop/Review/Codes/sequence images-open/"
images1=glob.glob(images_path1+"*.jpg")
images2=glob.glob(images_path2+"*.jpg")
images=[images1,images2]


labelname=[]
labelclass=[]
index=0
for i in images:
    index+=1
    countimg=0
    if index%2==0:
        countlabel=1
    else:
        countlabel=0
    for j in i:
        countimg+=1
        labelclass.append(countlabel)
        img=cv2.imread(j)
        labelname.append(img)
        
labelname=np.array(labelname)

y_train, y_test,X_train, X_test  = train_test_split(labelclass, labelname, test_size=0.4, shuffle=True)
X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.6, shuffle=True)

X_valid = np.array(X_valid)    # converting list to array
X_train=np.array(X_train)
X_test=np.array(X_test)


dummy_y_train = np_utils.to_categorical(y_train)    # one hot encoding Classes
dummy_y_valid_images = np_utils.to_categorical(y_valid)    # one hot encoding Classes
dummy_y_test = np_utils.to_categorical(y_test)

del images_path1, images_path2, images1, images2, images, labelclass, labelname



X_train = preprocess_input(X_train, mode='tf')      # preprocessing the input data
X_valid = preprocess_input(X_valid, mode='tf')      # preprocessing the input data
X_test = preprocess_input(X_test, mode='tf') 



model = Sequential()

model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Activation('relu'))
model.add(Convolution2D(24, (3, 3), data_format='channels_first'),)
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))


#model.add(Convolution2D(64, (3, 3), padding='same', data_format='channels_first'))
#model.add(Activation('relu'))
#model.add(Convolution2D(64, (3, 3)))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))
#model.add(Activation('relu'))
#model.add(Convolution2D(64, (3, 3)))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))



model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(nb_classes))
model.add(Activation('sigmoid'))

# let's train the model using SGD + momentum (how original).
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd , metrics=['accuracy'])


import datetime
start=datetime.datetime.now()

history=model.fit(X_train, dummy_y_train, batch_size=batch_size, epochs=epochs, verbose=2, validation_data=(X_valid, dummy_y_valid_images))

score = model.evaluate(X_test, dummy_y_test,  verbose=1)

print('Test score:', score[0])
print('Test accuracy:', score[1])


# Get training and test loss histories
training_loss = history.history['loss']
test_loss = history.history['val_loss']

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
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

## Save the weights
#model.save_weights('model_weights_FD-DNN_defect.h5')
## Save the model architecture
#with open('model_architecture_vgg19_zju.json', 'w') as f:
#    f.write(model.to_json())
#
#
#end=datetime.datetime.now()
#elapsed=end-start
#plot_model(model, to_file='modelfd-dnn-defect.png')
#print('training time',str(elapsed))

probs1 = model.predict(X_valid)
# keep probabilities for the positive outcome only
#probs = probs[:, 1]
# calculate AUC
auc1 = roc_auc_score(dummy_y_valid_images, probs1)
print('AUC1 (validation): %.3f' % auc1)
# calculate roc curve
score2 = model.evaluate(X_test, dummy_y_test, batch_size=32)
print("Test accuracy:",score2[1])

probs2 = model.predict(X_test)
# keep probabilities for the positive outcome only
#probs = probs[:, 1]
# calculate AUC
auc2 = roc_auc_score(dummy_y_test, probs2)
print('AUC2 (Test): %.3f' % auc2)
