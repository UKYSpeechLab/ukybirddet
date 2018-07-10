# DCASE 2018 - Bird Audio Detection challenge (Task 3)

# This code is a basic implementation of bird audio detector (based on baseline code's architecture)

import h5py
import csv
import numpy as np
import random
import PIL.Image
from HTK import HTKFile

from sklearn.metrics import roc_auc_score

import keras
from keras.layers import Conv2D, Dropout, MaxPooling2D, Dense, GlobalAveragePooling2D, Flatten, BatchNormalization
from keras.models import Sequential, load_model
from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator
from keras.losses import binary_crossentropy, mean_squared_error, mean_absolute_error

import my_callbacks
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping

################################################
#
#   Global parameters
#
################################################

#checking mfc features
SPECTPATH = '/audio/audio/mfcfeatures/'
#SPECTPATH = '/audio/audio/workingfiles/spect/'
#SPECTPATH = '/home/sidrah/DL/bulbul2018/workingfiles/spect/'
#SPECTPATH = 'C:\Sidrah\DCASE2018\dataset\spect\'
# path to spectrogram files stored in separate directories for each dataset
# -spect/
#       BirdVox-DCASE-20k
#       ff1010bird
#       warblrb10k

LABELPATH = '/audio/audio/labels/'
#LABELPATH = '/home/sidrah/DL/bulbul2018/labels/'
#LABELPATH = 'C:\Sidrah\DCASE2018\dataset\labels\'
# path to label files stored in a single directory named accordingly for each dataset
# -labels/
#       BirdVox-DCASE-20k.csv, ff1010bird.csv, warblrb10k.csv

FILELIST = '/audio/audio/workingfiles/filelists/'
#FILELIST = '/home/sidrah/DL/bulbul2018/workingfiles/filelists/'
#FILELIST = 'C:\Sidrah\DCASE2018\dataset\filelists'
# create this directory in main project directory

dataset = ['BirdVox-DCASE-20k.csv', 'ff1010bird.csv', 'warblrb10k.csv']
#features =['h5','mfc']
logfile_name = 'backup/mfc_model_3epochonff/furtheronFF/FforF_mfc_cfg4LR_noaug.log'
checkpoint_model_name = 'backup/mfc_model_3epochonff/furtheronFF/FforF_mfc_cfg4LR_noaug_ckpt.h5'
final_model_name = 'backup/mfc_model_3epochonff/furtheronFF/FforF_mfc_cfg4LR_noaug_flmdl.h5'

BATCH_SIZE = 32
EPOCH_SIZE = 20
AUGMENT_SIZE = 8
with_augmentation = False
features='mfc'
model_operation = 'load'
# model_operations : 'new', 'load', 'test'
#shape = (700, 80)
shape = (1669, 160)
spect = np.zeros(shape)
label = np.zeros(1)

# Callbacks for logging during epochs
reduceLR = ReduceLROnPlateau(factor=0.2, patience=2, min_lr=0.00001)
checkPoint = ModelCheckpoint(filepath = checkpoint_model_name, save_best_only=True)
csvLogger = CSVLogger(logfile_name, separator=',', append=False)
#earlyStopping = EarlyStopping(patience=5)

################################################
#
#   Data set selection
#
################################################

# Parameters in this section can be adjusted to select different data sets to train, test, and validate on.

# Keys by which we will access properties of a data set. The values assigned here are ultimately meaningless.
# The 'k' prefix on these declarations signify that they will be used as keys in a dictionary.
k_VAL_FILE = 'validation_file_path'
k_TEST_FILE = 'test_file_path'
k_TRAIN_FILE = 'train_file_path'
k_VAL_SIZE = 'validate_size'
k_TEST_SIZE = 'test_size'
k_TRAIN_SIZE = 'train_size'

# Declare the dictionaries to represent the data sets
d_birdVox = {k_VAL_FILE: 'val_B', k_TEST_FILE: 'test_B', k_TRAIN_FILE: 'train_B', k_VAL_SIZE: 1000.0, k_TEST_SIZE: 3000.0, k_TRAIN_SIZE: 16000.0}
d_warblr = {k_VAL_FILE: 'val_W', k_TEST_FILE: 'test_W', k_TRAIN_FILE: 'train_W', k_VAL_SIZE: 400.0, k_TEST_SIZE: 1200.0, k_TRAIN_SIZE: 6400.0}
d_freefield = {k_VAL_FILE: 'val_F', k_TEST_FILE: 'test_F', k_TRAIN_FILE: 'train_F', k_VAL_SIZE: 385.0, k_TEST_SIZE: 1153.0, k_TRAIN_SIZE: 6152.0}

# Declare the training, validation, and testing sets here using the dictionaries defined above.
# Set these variables to change the data set.
training_set = d_freefield
validation_set = d_freefield
test_set = d_freefield

# Grab the file lists and sizes from the corresponding data sets.
train_filelist = FILELIST + training_set[k_TRAIN_FILE]
TRAIN_SIZE = training_set[k_TRAIN_SIZE]

val_filelist = FILELIST + validation_set[k_VAL_FILE]
VAL_SIZE = validation_set[k_VAL_SIZE]

test_filelist = FILELIST + test_set[k_TEST_FILE]
TEST_SIZE = test_set[k_TEST_SIZE]

################################################
#
#   Generator with Augmentation
#
################################################

# use this generator when augmentation is needed
def data_generator(filelistpath, batch_size=32, shuffle=False):
    batch_index = 0
    image_index = -1
    filelist = open(filelistpath, 'r')
    filenames = filelist.readlines()
    filelist.close()

    labels_dict = {}
    for n in range(len(dataset)):
        labels_list = csv.reader(open(LABELPATH + dataset[n], 'r'))
        next(labels_list)
        for k, r, v in labels_list:
            labels_dict[r + '/' + k + '.wav'] = v

    while True:
        image_index = (image_index + 1) % len(filenames)

        # if shuffle and image_index = 0
        # shuffling filelist
        if shuffle == True and image_index == 0:
            random.shuffle(filenames)

        file_id = filenames[image_index].rstrip()

        if batch_index == 0:
            # re-initialize spectrogram and label batch
            spect_batch = np.zeros([1, spect.shape[0], spect.shape[1], 1])
            label_batch = np.zeros([1, 1])
            aug_spect_batch = np.zeros([batch_size, spect.shape[0], spect.shape[1], 1])
            aug_label_batch = np.zeros([batch_size, 1])

        if features=='h5':
            hf = h5py.File(SPECTPATH + file_id + '.h5', 'r')
            imagedata = hf.get('features')
            imagedata = np.array(imagedata)
            hf.close()
        elif features == 'mfc':
            htk_reader = HTKFile()
            htk_reader.load(SPECTPATH + file_id.rstrip('.wav') + '.mfc')
            imagedata = np.array(htk_reader.data)

        # normalizing intensity values of spectrogram from [-15.0966 to 2.25745] to [0 to 1] range
        #imagedata = (imagedata + 15.0966)/(15.0966 + 2.25745)

        imagedata = np.reshape(imagedata, (1, imagedata.shape[0], imagedata.shape[1], 1))

        spect_batch[0, :, :, :] = imagedata
        label_batch[0, :] = labels_dict[file_id]

        gen_img = datagen.flow(imagedata, label_batch[0, :], batch_size=1, shuffle=False, save_to_dir=None)
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


################################################
#
#   Generator without Augmentation
#
################################################

def dataval_generator(filelistpath, batch_size=32, shuffle=False):
    batch_index = 0
    image_index = -1

    filelist = open(filelistpath, 'r')
    filenames = filelist.readlines()
    filelist.close()

    labels_dict = {}
    for n in range(len(dataset)):
        labels_list = csv.reader(open(LABELPATH + dataset[n], 'r'))
        next(labels_list)
        for k, r, v in labels_list:
            labels_dict[r + '/' + k + '.wav'] = v

    while True:
        image_index = (image_index + 1) % len(filenames)

        # if shuffle and image_index = 0
        # shuffling filelist
        if shuffle == True and image_index == 0:
            random.shuffle(filenames)

        file_id = filenames[image_index].rstrip()

        if batch_index == 0:
            # re-initialize spectrogram and label batch
            spect_batch = np.zeros([batch_size, spect.shape[0], spect.shape[1], 1])
            label_batch = np.zeros([batch_size, 1])

        if features == 'h5':
            hf = h5py.File(SPECTPATH + file_id + '.h5', 'r')
            imagedata = hf.get('features')
            imagedata = np.array(imagedata)
            hf.close()

        elif features == 'mfc':
            htk_reader = HTKFile()
            htk_reader.load(SPECTPATH + file_id.rstrip('.wav') + '.mfc')
            imagedata = np.array(htk_reader.data)

        # normalizing intensity values of spectrogram from [-15.0966 to 2.25745] to [0 to 1] range
        #imagedata = (imagedata + 15.0966) / (15.0966 + 2.25745)

        imagedata = np.reshape(imagedata, (1, imagedata.shape[0], imagedata.shape[1], 1))

        spect_batch[batch_index, :, :, :] = imagedata
        label_batch[batch_index, :] = labels_dict[file_id]

        batch_index += 1

        if batch_index >= batch_size:
            batch_index = 0
            inputs = [spect_batch]
            outputs = [label_batch]
            yield inputs, outputs


################################################
#
#   ROC Label Generation
#
################################################

def testdata(filelistpath, test_size):
    image_index = -1

    filelist = open(filelistpath, 'r')
    filenames = filelist.readlines()
    filelist.close()

    labels_dict = {}
    for n in range(len(dataset)):
        labels_list = csv.reader(open(LABELPATH + dataset[n], 'r'))
        next(labels_list)
        for k, r, v in labels_list:
            labels_dict[r + '/' + k + '.wav'] = v

    label_batch = np.zeros([int(test_size), 1])

    for m in range(len(filenames)):
        image_index = (image_index + 1) % len(filenames)

        file_id = filenames[image_index].rstrip()

        label_batch[image_index, :] = labels_dict[file_id]

        outputs = [label_batch]

    return outputs


################################################
#
#   Model Creation
#
################################################

if(with_augmentation == True):
    train_generator = data_generator(train_filelist, BATCH_SIZE, False)
else:
    train_generator = dataval_generator(train_filelist, BATCH_SIZE, False)

validation_generator = dataval_generator(val_filelist, BATCH_SIZE, False)
test_generator = dataval_generator(test_filelist, BATCH_SIZE, False)

datagen = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=0,
    width_shift_range=0.05,
    height_shift_range=0.9,
    horizontal_flip=False,
    fill_mode="wrap")

if model_operation == 'new':
    model = Sequential()
    # augmentation generator
    # code from baseline : "augment:Rotation|augment:Shift(low=-1,high=1,axis=3)"
    # keras augmentation:
    #preprocessing_function

    # convolution layers
    model.add(Conv2D(16, (3, 3), padding='valid', input_shape=(1669, 160, 1)))
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

elif model_operation == 'load' or model_operation == 'test':
    model = load_model('backup/mfc_model_3epochonff/BforB_mfc_cfg4LR_noaug_ckpt.h5')

if model_operation == 'new' or model_operation == 'load':
    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['acc'])

    # prepare callback
    histories = my_callbacks.Histories()

model.summary()

my_steps = np.floor(TRAIN_SIZE*AUGMENT_SIZE / BATCH_SIZE)
my_val_steps = np.floor(VAL_SIZE / BATCH_SIZE)
my_test_steps = np.floor(TEST_SIZE / BATCH_SIZE)

if model_operation == 'new' or model_operation == 'load':
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=my_steps,
        epochs=EPOCH_SIZE,
        validation_data=validation_generator,
        validation_steps=my_val_steps,
        callbacks= [checkPoint, reduceLR, csvLogger],
        verbose=True)

    model.save(final_model_name)

# generating prediction values for computing ROC_AUC score
# whether model_operation is 'new', 'load' or 'test'
pred_generator = dataval_generator(test_filelist, BATCH_SIZE, False)
y_test = testdata(test_filelist, int(TEST_SIZE))
y_test = np.reshape(y_test, (int(TEST_SIZE),1))
y_pred = model.predict_generator(
    pred_generator,
    steps=my_test_steps)
# Calculate total roc auc score
score = roc_auc_score(y_test[0:int(my_test_steps*BATCH_SIZE)], y_pred[0:int(my_test_steps*BATCH_SIZE)])
print("Total roc auc score = {0:0.4f}".format(score))

