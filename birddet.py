# DCASE 2018 - Bird Audio Detection challenge (Task 3)

# This code is a basic implementation of bird audio detector (based on baseline code's architecture)
# This code performs three-fold crossvalidation checks performance of bird detector on a single dataset BirdVox20k
# AUC score calculation not added yet.

import h5py
import csv
import numpy as np
import random
import PIL.Image

import keras
from keras.layers import Conv2D, Dropout, MaxPooling2D, Dense, GlobalAveragePooling2D, Flatten, BatchNormalization
from keras.models import Sequential
from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator
from keras.losses import binary_crossentropy, mean_squared_error, mean_absolute_error

SPECTPATH = '/audio/audio/workingfiles/spect/'
#SPECTPATH = '/home/sidrah/DL/bulbul2018/workingfiles/spect/'
# path to spectrogram files stored in separate directories for each dataset
# -spect/
#       BirdVox-DCASE-20k
#       ff1010bird
#       warblrb10k

LABELPATH = '/audio/audio/labels/'
#LABELPATH = '/home/sidrah/DL/bulbul2018/labels/'
# path to label files stored in a single directory named accordingly for each dataset
# -labels/
#       BirdVox-DCASE-20k.csv, ff1010bird.csv, warblrb10k.csv

FILELIST = '/audio/audio/workingfiles/filelists/'
#FILELIST = '/home/sidrah/DL/bulbul2018/workingfiles/filelists/'
# create this directory in main project directory

#DATASET = 'BirdVox-DCASE-20k.csv'

BATCH_SIZE = 32
EPOCH_SIZE = 50
AUGMENT_SIZE = 16
shape = (700, 80)
spect = np.zeros(shape)
label = np.zeros(1)

def data_generator(filelistpath, batch_size=32, shuffle=False):
    batch_index = 0
    image_index = -1
    filelist = open(filelistpath[0], 'r')
    filenames = filelist.readlines()
    filelist.close()

    # shuffling filelist
    if shuffle==True:
        random.shuffle(filenames)

    dataset = ['BirdVox-DCASE-20k.csv', 'ff1010bird.csv', 'warblrb10k.csv']

    labels_list = []
    labels_dict = {}
    for n in range(len(dataset)):
        labels_list = csv.reader(open(LABELPATH + dataset[n], 'r'))
        next(labels_list)
        for k, r, v in labels_list:
            labels_dict[r + '/' + k + '.wav'] = v

    while True:
        image_index = (image_index + 1) % len(filenames)

        # if shuffle and image_index = 0
        # write code for shuffling filelist
        file_id = filenames[image_index].rstrip()

        if batch_index == 0:
            # re-initialize spectrogram and label batch
            spect_batch = np.zeros([1, spect.shape[0], spect.shape[1], 1])
            label_batch = np.zeros([1, 1])
            aug_spect_batch = np.zeros([batch_size, spect.shape[0], spect.shape[1], 1])
            aug_label_batch = np.zeros([batch_size, 1])

        hf = h5py.File(SPECTPATH + file_id + '.h5', 'r')
        imagedata = hf.get('features')
        imagedata = np.array(imagedata)
        hf.close()
        # normalizing intensity values of spectrogram from [-15.0966 to 2.25745] to [0 to 1] range
        imagedata = (imagedata + 15.0966)/(15.0966 + 2.25745)

        imagedata = np.reshape(imagedata, (1, imagedata.shape[0], imagedata.shape[1], 1))

        spect_batch[0, :, :, :] = imagedata
        label_batch[0, :] = labels_dict[file_id]

        gen_img = datagen.flow(imagedata, label_batch[0, :], batch_size=1, shuffle=False, save_to_dir='augimg/')
        aug_spect_batch[batch_index, :, :, :] = imagedata
        aug_label_batch[batch_index, :] = label_batch[0, :]
        batch_index += 1

        for n in range(AUGMENT_SIZE-1):
            aug_spect_batch[batch_index, :, :, :], aug_label_batch[batch_index, :] = gen_img.next()
            batch_index += 1
            if batch_index >= batch_size:
                batch_index = 0
                inputs = [aug_spect_batch]
                outputs = [aug_label_batch]
                yield inputs, outputs


def dataval_generator(filelistpath, batch_size=32, shuffle=False):
    batch_index = 0
    image_index = -1

    filelist = open(filelistpath[0], 'r')
    filenames = filelist.readlines()
    filelist.close()

    # shuffling filelist
    if shuffle == True:
        random.shuffle(filenames)

    dataset = ['BirdVox-DCASE-20k.csv', 'ff1010bird.csv', 'warblrb10k.csv']

    labels_list = []
    labels_dict = {}
    for n in range(len(dataset)):
        labels_list = csv.reader(open(LABELPATH + dataset[n], 'r'))
        next(labels_list)
        for k, r, v in labels_list:
            labels_dict[r + '/' + k + '.wav'] = v

    while True:
        image_index = (image_index + 1) % len(filenames)
        # if shuffle and image_index = 0
        # write code for shuffling filelist
        file_id = filenames[image_index].rstrip()

        if batch_index == 0:
            # re-initialize spectrogram and label batch
            spect_batch = np.zeros([batch_size, spect.shape[0], spect.shape[1], 1])
            label_batch = np.zeros([batch_size, 1])

        hf = h5py.File(SPECTPATH + file_id + '.h5', 'r')
        imagedata = hf.get('features')
        imagedata = np.array(imagedata)
        hf.close()
        # normalizing intensity values of spectrogram from [-15.0966 to 2.25745] to [0 to 1] range
        imagedata = (imagedata + 15.0966) / (15.0966 + 2.25745)

        imagedata = np.reshape(imagedata, (1, imagedata.shape[0], imagedata.shape[1], 1))

        spect_batch[batch_index, :, :, :] = imagedata
        label_batch[batch_index, :] = labels_dict[file_id]

        batch_index += 1


        if batch_index >= batch_size:
            batch_index = 0
            inputs = [spect_batch]
            outputs = [label_batch]
            yield inputs, outputs

train_filelist=[FILELIST+'train_B']
val_filelist=[FILELIST+'val_B']
test_filelist=[FILELIST+'test_B']
#train_filelist=['/audio/audio/workingfiles/filelists/train_B']
#val_filelist=['/audio/audio/workingfiles/filelists/val_B']

train_generator = data_generator(train_filelist, BATCH_SIZE, True)
validation_generator = dataval_generator(val_filelist, BATCH_SIZE, True)
test_generator = dataval_generator(test_filelist, BATCH_SIZE, False)

datagen = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=0,
    width_shift_range=0.05,
    height_shift_range=0.9,
    horizontal_flip=False,
    fill_mode="wrap")

model = Sequential()
# augmentation generator
# code from baseline : "augment:Rotation|augment:Shift(low=-1,high=1,axis=3)"
# keras augmentation:
#preprocessing_function

# convolution layers
model.add(Conv2D(16, (3, 3), padding='valid', input_shape=(700, 80, 1)))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=.001))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Conv2D(16, (3, 3), padding='valid'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=.001))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Conv2D(16, (3, 1), padding='valid'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=.001))
model.add(MaxPooling2D(pool_size=(3,1)))
model.add(Conv2D(16, (3, 1), padding='valid'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=.001))
model.add(MaxPooling2D(pool_size=(3,1)))

# dense layers
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(256))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=.001))
model.add(Dropout(0.5))
model.add(Dense(32))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=.001))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))

adam=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['acc'])

model.summary()

my_steps = np.round(16000.0*AUGMENT_SIZE / BATCH_SIZE)
my_val_steps = np.round(1000.0 / BATCH_SIZE)
my_test_steps = np.round(3000.0 / BATCH_SIZE)

history = model.fit_generator(
    train_generator,
    steps_per_epoch=my_steps,
    epochs=EPOCH_SIZE,
    validation_data=validation_generator,
    validation_steps=my_val_steps)

model.evaluate_generator(
    test_generator,
    steps=my_test_steps)


