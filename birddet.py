# Author: Sidrah Liaqat
# This code is a basic implementation of bird detector based on bulbul architecture
# This code checks performance of bird detector on a single dataset BirdVox20k

import h5py
import csv
import numpy as np
from keras.layers import Conv2D, Dropout, MaxPooling2D, Dense, GlobalAveragePooling2D, Flatten
from keras import Sequential
from keras.losses import binary_crossentropy, mean_squared_error, mean_absolute_error

SPECTPATH = 'C:/Sidrah/DCASE2018/dataset/spect/'
LABELPATH = 'labels/'
FILELIST = 'filelist/'
DATASET = 'BirdVox-DCASE-20k.csv'
BATCH_SIZE = 32
shape=(700,80)
spect = np.zeros(shape)
label = np.zeros(1)

#filelist-can also be path
def data_generator(filelistpth, batch_size = 32, shuffle=False):
    batch_index = 0
    image_index = -1
    filelist = open(filelistpth, 'r')
    filenames = filelist.readlines()
    filelist.close()

    labels_list = csv.reader(open(LABELPATH+DATASET, 'r'))
    labels_dict = {}
    for k, r, v in labels_list:
        labels_dict[r+'/'+k+'.wav'] = v

    while True:
        image_index = (image_index + 1) % len(filenames)
        # if shuffle and image_index = 0
            #write code for shuffling filelist
        file_id = filenames[image_index].rstrip()

        if batch_index==0:
            # re-initialize spectrogram and label batch
            spect_batch = np.zeros([batch_size, spect.shape[0], spect.shape[1], 1 ])
            label_batch = np.zeros([batch_size, 1])

        hf = h5py.File(SPECTPATH + file_id + '.h5', 'r')
        imagedata = hf.get('features')
        imagedata = np.array(imagedata)

        imagedata = np.reshape(imagedata, (imagedata.shape[0], imagedata.shape[1], 1 ) )

        hf.close()

        spect_batch[batch_index, :, :, :] = imagedata
        label_batch[batch_index, :] = labels_dict[file_id]

        batch_index+=1

        if batch_index>=batch_size:
            batch_index = 0
            inputs = [spect_batch]
            outputs = [label_batch]
            yield inputs, outputs

train_generator = data_generator(FILELIST+'train', 32, False)
validation_generator = data_generator(FILELIST+'test', 32, False)

model = Sequential()
model.add(Conv2D(16, (3,3), padding='same', activation='relu', input_shape=(700,80, 1)))
model.add(Conv2D(8, (3,3), padding='same', activation='relu'))
model.add(Conv2D(4, (3,3), padding='same', activation='relu'))
#model1.add(GlobalAveragePooling2D())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#model1.compile(optimizer=optimizers.adam(), loss=losses.categorical_crossentropy())

model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['acc'])
#model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['acc'])

model.summary()

my_steps = np.round(14000.0 / BATCH_SIZE)
my_val_steps = np.round(6000.0 / BATCH_SIZE)

history = model.fit_generator(
        train_generator,
        steps_per_epoch=my_steps,
        epochs=20,
        validation_data=validation_generator,
        validation_steps=800)

print('Debugging!')