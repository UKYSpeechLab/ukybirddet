# DCASE 2018 - Bird Audio Detection challenge (Task 3)

# This code is a basic implementation of bird audio detector (based on baseline code's architecture)
# This code performs three-fold crossvalidation checks performance of bird detector on a single dataset BirdVox20k
# AUC score calculation not added yet.

import h5py
import csv
import numpy as np

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, Dropout, MaxPooling2D, Dense, GlobalAveragePooling2D, Flatten
from keras.models import Sequential
from keras.layers.advanced_activations import LeakyReLU
from keras.losses import binary_crossentropy, mean_squared_error, mean_absolute_error

SPECTPATH = '/audio/audio/workingfiles/spect/'
# path to spectrogram files stored in separate directories for each dataset
# -spect/
#       BirdVox-DCASE-20k
#       ff1010bird
#       warblrb10k

LABELPATH = '/audio/audio/labels/'
# path to label files stored in a single directory named accordingly for each dataset
# -labels/
#       BirdVox-DCASE-20k.csv, ff1010bird.csv, warblrb10k.csv

FILELIST = '/audio/audio/workingfiles/filelist/'
# create this directory in main project directory

#DATASET = 'BirdVox-DCASE-20k.csv'

BATCH_SIZE = 32
EPOCH_SIZE = 50
shape = (700, 80)
spect = np.zeros(shape)
label = np.zeros(1)

# filelist-can also be path
def data_generator(filelistpth, batch_size=32, shuffle=False):
    batch_index = 0
    image_index = -1
    filelist = open(filelistpth, 'r')
    filenames = filelist.readlines()
    filelist.close()

    dataset = ['BirdVox-DCASE-20k.csv', 'ff1010bird.csv', 'warblrb10k.csv']

    labels_list = []
    labels_dict = {}
    for n in range(len(dataset)):
        labels_list = csv.reader(open(LABELPATH + dataset[n], 'r'))
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

        # normalizing intensity values of spectrogram from [-15.0966 to 2.25745] to [0 to 1] range
        imagedata = (imagedata + 15.0966)/(15.0966 + 2.25745)

        imagedata = np.reshape(imagedata, (imagedata.shape[0], imagedata.shape[1], 1))

        hf.close()

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

train_generator = data_generator(train_filelist, 32, False)
validation_generator = data_generator(val_filelist, 32, False)

model = Sequential()
# augmentation layer
# code from baseline : "augment:Rotation|augment:Shift(low=-1,high=1,axis=3)"
# keras augmentation layer : TO BE DONE

# normalization layer
# code from baseline : "normpart="normalize:BatchNorm(axes=0x1x2,alpha=0.1,beta=None,gamma=None)"  # normalize over all except frequency axis"
# keras normalization layer:
# keras.layers.normalization.BatchNormalization(epsilon=1e-06, mode=0, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one')

# convolution layers
model.add(Conv2D(16, (3, 3), padding='valid', input_shape=(700, 80, 1)))
model.add(LeakyReLU(alpha=.001))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Conv2D(16, (3, 3), padding='valid'))
model.add(LeakyReLU(alpha=.001))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Conv2D(16, (3, 1), padding='valid'))
model.add(LeakyReLU(alpha=.001))
model.add(MaxPooling2D(pool_size=(3,1)))
model.add(Conv2D(16, (3, 1), padding='valid'))
model.add(LeakyReLU(alpha=.001))
model.add(MaxPooling2D(pool_size=(3,1)))

# dense layers
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(256))
model.add(LeakyReLU(alpha=.001))
model.add(Dropout(0.5))
model.add(Dense(32))
model.add(LeakyReLU(alpha=.001))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

adam=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['acc'])

model.summary()

my_steps = np.round(14000.0 / BATCH_SIZE)
my_val_steps = np.round(6000.0 / BATCH_SIZE)

history = model.fit_generator(
    train_generator,
    steps_per_epoch=my_steps,
    epochs=EPOCH_SIZE,
    validation_data=validation_generator,
    validation_steps=my_val_steps)
