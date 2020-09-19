#Importing the required libraries


from keras import layers
from keras.layers import Input,Dense,Activation,ZeroPadding2D,BatchNormalization,Flatten,Conv2D
from keras.layers import AveragePooling2D,MaxPooling2D,Dropout,GlobalAveragePooling2D,GlobalMaxPooling2D
from keras.utils import np_utils
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
import keras.backend as K

'''<------------------------------------CNN model-------------------------------------------->'''

def Keras_Model():
    num_of_classes = 10
    model = Sequential()
    model.add(Conv2D(filters=10,kernel_size=(5,5),input_shape=(28,28,1),activation='sigmoid'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same'))
    model.add(Conv2D(56,(5,5),activation='sigmoid'))
    model.add(MaxPooling2D(pool_size=(5,5),strides=(5,5),padding='same'))
    model.add(Flatten())
    model.add(Dense(num_of_classes,activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics = ['accuracy'])

    filepath = 'utpal.h5'
    checkpoint1 = ModelCheckpoint(filepath, monitor='val_acc', verbose = 1 , save_best_only = True , mode='max')
    callbacks_list = [checkpoint1]
    return model,callbacks_list

#Function to extract X_train,y_train,X_test and Y_test images

import gzip
import numpy as np


def Training_images():
    with gzip.open('train-images-idx3-ubyte.gz', 'r') as f:
        # first 4 bytes is a magic number
        magic_number = int.from_bytes(f.read(4), 'big')
        # second 4 bytes is the number of images
        image_count = int.from_bytes(f.read(4), 'big')
        # third 4 bytes is the row count
        row_count = int.from_bytes(f.read(4), 'big')
        # fourth 4 bytes is the column count
        column_count = int.from_bytes(f.read(4), 'big')
        # rest is the image pixel data, each pixel is stored as an unsigned byte
        # pixel values are 0 to 255
        image_data = f.read()
        train_images = np.frombuffer(image_data, dtype=np.uint8)\
            .reshape((image_count, row_count, column_count,1))
        return train_images


def Training_labels():
    with gzip.open('train-labels-idx1-ubyte.gz', 'r') as f:
        # first 4 bytes is a magic number
        magic_number = int.from_bytes(f.read(4), 'big')
        # second 4 bytes is the number of labels
        label_count = int.from_bytes(f.read(4), 'big')
        # rest is the label data, each label is stored as unsigned byte
        # label values are 0 to 9
        label_data = f.read()
        train_labels = np.frombuffer(label_data, dtype=np.uint8)
        return train_labels

def Testing_images():
    with gzip.open('t10k-images-idx3-ubyte.gz', 'r') as f:
        # first 4 bytes is a magic number
        magic_number = int.from_bytes(f.read(4), 'big')
        # second 4 bytes is the number of images
        image_count = int.from_bytes(f.read(4), 'big')
        # third 4 bytes is the row count
        row_count = int.from_bytes(f.read(4), 'big')
        # fourth 4 bytes is the column count
        column_count = int.from_bytes(f.read(4), 'big')
        # rest is the image pixel data, each pixel is stored as an unsigned byte
        # pixel values are 0 to 255
        image_data = f.read()
        test_images = np.frombuffer(image_data, dtype=np.uint8)\
            .reshape((image_count, row_count, column_count,1))
        return test_images

def Testing_labels():
    with gzip.open('t10k-labels-idx1-ubyte.gz', 'r') as f:
        # first 4 bytes is a magic number
        magic_number = int.from_bytes(f.read(4), 'big')
        # second 4 bytes is the number of labels
        label_count = int.from_bytes(f.read(4), 'big')
        # rest is the label data, each label is stored as unsigned byte
        # label values are 0 to 9
        label_data = f.read()
        test_labels = np.frombuffer(label_data, dtype=np.uint8)
        return test_labels

'''<------------------------------------Data Cleaning----------------------------------->'''
train_images = Training_images() 
train_labels  = Training_labels()
test_images = Testing_images()
test_labels = Testing_labels()

print('train images shape')
print(train_images.shape)

print('train labels shape')
print(train_labels.shape)


t = np.vstack(train_labels)#Temporary variable


train_labels = np_utils.to_categorical(t)


print('train label shape')
train_labels.shape

test_images.shape
t = np.vstack(test_labels)
t.shape

test_labels = np_utils.to_categorical(t)
print('train label shape')
print(test_labels.shape)

#Show some sample of train images

import matplotlib.pyplot as plt


for i in range(0,10):
    image = np.asarray(train_images[i]).squeeze()
    image = image/255 # to reduce calculation
    plt.imshow(image)
    plt.show()


for i in range(0,10):
    image = np.asarray(test_images[i]).squeeze()
    image = image/255 # to reduce calculation
    plt.imshow(image)
    plt.show()

model , callbacks_list = Keras_Model()
print(model.summary())
model.fit(train_images,train_labels,validation_data=(test_images,test_labels),epochs = 5,batch_size = 56, callbacks = callbacks_list )
scores = model.evaluate(test_images,test_labels,verbose = 0)
print("CNN error :%2f%% "%(100-scores[1]*100))
model.save('Utpal_reload.h5')

#Application


from keras.models import load_model
import numpy as np
from collections import deque
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

model1 = load_model('Utpal_reload.h5')
print(model1)

'''<------------------------Evaluation of the model------------------------>'''


pred = model1.predict([test_images])

print(pred)

print(np.argmax(pred[1]))


im = np.asarray(test_images[1]).squeeze()
plt.imshow(im)
plt.show()
