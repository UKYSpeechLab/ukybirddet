import h5py
import csv
import numpy as np
import random
import PIL.Image
import matplotlib.pyplot as plt
from HTK import HTKFile

from sklearn.metrics import roc_auc_score, roc_curve, auc

import keras
from keras import Model
from keras.layers import Conv2D, Dropout, MaxPooling2D, Dense, GlobalAveragePooling2D, Flatten, BatchNormalization, AveragePooling2D
from keras.models import Sequential, load_model
from keras.layers import Input, Concatenate, Activation, concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator
from keras.losses import binary_crossentropy, mean_squared_error, mean_absolute_error
from keras.regularizers import l2

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

#SPECTPATH1 = '/home/sidrah/DL/ukybirddet/workingfiles/win_32ms/'
#SPECTPATH2 = '/home/sidrah/DL/ukybirddet/workingfiles/spect/'
#SPECTPATH3 = '/home/sidrah/DL/ukybirddet/workingfiles/80fbanks/win_12ms/'

SPECTPATH1 = 'workingfiles/features_high_frequency/'
SPECTPATH2 = 'workingfiles/features_baseline/'
SPECTPATH3 = 'workingfiles/features_high_temporal/'

LABELPATH = 'labels/'

FILELIST = 'workingfiles/filelists/'
RESULTPATH = 'trained_model/multimodel/'

dataset = ['BirdVox-DCASE-20k.csv', 'ff1010bird.csv', 'warblrb10k.csv']

BATCH_SIZE = 16
EPOCH_SIZE = 15
AUGMENT_SIZE = 1
with_augmentation = False
features1='mfc'
features2='h5'
features3='mfc'
model_operation = 'new'
# model_operations : 'new', 'load', 'test'
shape1 = (624, 160)
expected_shape1 = (624, 160)
shape2 = (700, 80)
expected_shape2 = (700, 80)
shape3 = (1669, 80)
expected_shape3 = (1669, 80)

spect1 = np.zeros(shape1)
spect2 = np.zeros(shape2)
spect3 = np.zeros(shape3)
label = np.zeros(1)

logfile_name = RESULTPATH + 'logfile.log'
checkpoint_model_name = RESULTPATH + 'ckpt.h5'
final_model_name = RESULTPATH + 'flmdl.h5'
# Callbacks for logging during epochs
reduceLR = ReduceLROnPlateau(factor=0.2, patience=5, min_lr=0.00001)
checkPoint = ModelCheckpoint(filepath = checkpoint_model_name, save_best_only=True)   # monitor = 'val_acc', mode = 'max'
csvLogger = CSVLogger(logfile_name, separator=',', append=False)
#earlyStopping = EarlyStopping(patience=5)

k_VAL_FILE = 'validation_file_path'
k_TEST_FILE = 'test_file_path'
k_TRAIN_FILE = 'train_file_path'
k_VAL_SIZE = 'validate_size'
k_TEST_SIZE = 'test_size'
k_TRAIN_SIZE = 'train_size'
k_CLASS_WEIGHT = 'class_weight'

# Declare the dictionaries to represent the data sets
d_birdVox = {k_VAL_FILE: 'val_B', k_TEST_FILE: 'test_B', k_TRAIN_FILE: 'train_B',
             k_VAL_SIZE: 1000.0, k_TEST_SIZE: 3000.0, k_TRAIN_SIZE: 16000.0,
             k_CLASS_WEIGHT: {0: 0.50,1: 0.50}}
d_warblr = {k_VAL_FILE: 'val_W', k_TEST_FILE: 'test_W', k_TRAIN_FILE: 'train_W',
            k_VAL_SIZE: 400.0, k_TEST_SIZE: 1200.0, k_TRAIN_SIZE: 6400.0,
            k_CLASS_WEIGHT: {0: 0.75, 1: 0.25}}
d_freefield = {k_VAL_FILE: 'val_F', k_TEST_FILE: 'test_F', k_TRAIN_FILE: 'train_F',
               k_VAL_SIZE: 385.0, k_TEST_SIZE: 1153.0, k_TRAIN_SIZE: 6152.0,
               k_CLASS_WEIGHT: {0: 0.25, 1: 0.75}}
d_fold1 = {k_VAL_FILE: 'test_BF', k_TEST_FILE: 'val_1', k_TRAIN_FILE: 'train_BF',
           k_VAL_SIZE: 4153.0, k_TEST_SIZE: 8000.0, k_TRAIN_SIZE: 22152.0,
           k_CLASS_WEIGHT: {0: 0.43, 1: 0.57}}
d_fold2 = {k_VAL_FILE: 'test_WF', k_TEST_FILE: 'val_2', k_TRAIN_FILE: 'train_WF',
           k_VAL_SIZE: 2353.0, k_TEST_SIZE: 20000.0, k_TRAIN_SIZE: 12552.0,
           k_CLASS_WEIGHT: {0: 0.50, 1: 0.50}}
d_fold3 = {k_VAL_FILE: 'test_BW', k_TEST_FILE: 'val_3', k_TRAIN_FILE: 'train_BW',
           k_VAL_SIZE: 4200.0, k_TEST_SIZE: 7690.0, k_TRAIN_SIZE: 22400.0,
           k_CLASS_WEIGHT: {0: 0.57, 1: 0.43}}
d_all3 = {k_VAL_FILE: 'val_BWF', k_TEST_FILE: 'test', k_TRAIN_FILE: 'train_BWF',
           k_VAL_SIZE: 1785.0, k_TEST_SIZE: 12620.0, k_TRAIN_SIZE: 35690.0,
k_CLASS_WEIGHT: {0: 0.50, 1: 0.50}}

# Declare the training, validation, and testing sets here using the dictionaries defined above.
# Set these variables to change the data set.
training_set = d_all3
validation_set = d_all3
test_set = d_all3

# Grab the file lists and sizes from the corresponding data sets.
train_filelist = FILELIST + training_set[k_TRAIN_FILE]
TRAIN_SIZE = training_set[k_TRAIN_SIZE]

val_filelist = FILELIST + validation_set[k_VAL_FILE]
VAL_SIZE = validation_set[k_VAL_SIZE]

test_filelist = FILELIST + test_set[k_TEST_FILE]
TEST_SIZE = test_set[k_TEST_SIZE]

def correct_dimensions(imagedata, expected_shape):
    # processing files with shapes other than expected shape in warblr dataset

    if imagedata.shape[0] != expected_shape[0]:
        old_imagedata = imagedata
        imagedata = np.zeros(expected_shape)

        if old_imagedata.shape[0] < expected_shape[0]:

            diff_in_frames = expected_shape[0] - old_imagedata.shape[0]
            if diff_in_frames < expected_shape[0] / 2:
                imagedata = np.vstack((old_imagedata, old_imagedata[
                    range(old_imagedata.shape[0] - diff_in_frames, old_imagedata.shape[0])]))

            elif diff_in_frames > expected_shape[0] / 2:
                count = np.floor(expected_shape[0] / old_imagedata.shape[0])
                remaining_diff = (expected_shape[0] - old_imagedata.shape[0] * int(count))
                imagedata = np.vstack(([old_imagedata] * int(count)))
                imagedata = np.vstack(
                    (imagedata, old_imagedata[range(old_imagedata.shape[0] - remaining_diff, old_imagedata.shape[0])]))

        elif old_imagedata.shape[0] > expected_shape[0]:
            diff_in_frames = old_imagedata.shape[0] - expected_shape[0]

            if diff_in_frames < expected_shape[0] / 2:
                imagedata[range(0, diff_in_frames + 1), :] = np.mean(np.array(
                    [old_imagedata[range(0, diff_in_frames + 1), :],
                     old_imagedata[range(old_imagedata.shape[0] - diff_in_frames - 1, old_imagedata.shape[0]), :]]),
                                                                     axis=0)
                imagedata[range(diff_in_frames + 1, expected_shape[0]), :] = old_imagedata[
                    range(diff_in_frames + 1, expected_shape[0])]

            elif diff_in_frames > expected_shape[0] / 2:
                count = int(np.floor(old_imagedata.shape[0] / expected_shape[0]))
                remaining_diff = (old_imagedata.shape[0] - expected_shape[0] * count)
                for index in range(0, count):
                    imagedata[range(0, expected_shape[0]), :] = np.sum(
                        [imagedata, old_imagedata[range(index * expected_shape[0], (index + 1) * expected_shape[0])]],
                        axis=0) / count
                    imagedata[range(0, remaining_diff), :] = np.mean(np.array(
                        [old_imagedata[range(old_imagedata.shape[0] - remaining_diff, old_imagedata.shape[0]), :],
                         imagedata[range(0, remaining_diff), :]]), axis=0)

    return imagedata

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
            spect_batch1 = np.zeros([batch_size, spect1.shape[0], spect1.shape[1], 1])
            spect_batch2 = np.zeros([batch_size, spect2.shape[0], spect2.shape[1], 1])
            spect_batch3 = np.zeros([batch_size, spect3.shape[0], spect3.shape[1], 1])
            label_batch = np.zeros([batch_size, 1])

        ####### feature matrix for network 1 ######################3
        if features1 == 'h5':
            hf = h5py.File(SPECTPATH1 + file_id + '.h5', 'r')
            imagedata1 = hf.get('features')
            imagedata1= np.array(imagedata1)
            hf.close()
            imagedata1 = (imagedata1 + 15.0966)/(15.0966 + 2.25745)

        elif features1 == 'mfc':
            htk_reader = HTKFile()
            htk_reader.load(SPECTPATH1 + file_id[:-4] + '.mfc')
            imagedata1 = np.array(htk_reader.data)
            imagedata1 = imagedata1/18.0

        imagedata1 = correct_dimensions(imagedata1, expected_shape1)
        imagedata1 = np.reshape(imagedata1, (1, imagedata1.shape[0], imagedata1.shape[1], 1))
        spect_batch1[batch_index, :, :, :] = imagedata1

        ####### feature matrix for network 2 ######################
        if features2 == 'h5':
            hf = h5py.File(SPECTPATH2 + file_id + '.h5', 'r')
            imagedata2 = hf.get('features')
            imagedata2 = np.array(imagedata2)
            hf.close()
            imagedata2 = (imagedata2 + 15.0966) / (15.0966 + 2.25745)

        elif features2 == 'mfc':
            htk_reader = HTKFile()
            htk_reader.load(SPECTPATH2 + file_id[:-4] + '.mfc')
            imagedata2 = np.array(htk_reader.data)
            imagedata2 = imagedata2 / 18.0

        imagedata2 = correct_dimensions(imagedata2, expected_shape2)
        imagedata2 = np.reshape(imagedata2, (1, imagedata2.shape[0], imagedata2.shape[1], 1))
        spect_batch2[batch_index, :, :, :] = imagedata2

        ####### feature matrix for network 3 ######################
        if features3 == 'h5':
            hf = h5py.File(SPECTPATH3 + file_id + '.h5', 'r')
            imagedata3 = hf.get('features')
            imagedata3 = np.array(imagedata3)
            hf.close()
            imagedata3 = (imagedata3 + 15.0966) / (15.0966 + 2.25745)

        elif features3 == 'mfc':
            htk_reader = HTKFile()
            htk_reader.load(SPECTPATH3 + file_id[:-4] + '.mfc')
            imagedata3 = np.array(htk_reader.data)
            imagedata3 = imagedata3 / 18.0

        imagedata3 = correct_dimensions(imagedata3, expected_shape3)
        imagedata3 = np.reshape(imagedata3, (1, imagedata3.shape[0], imagedata3.shape[1], 1))
        spect_batch3[batch_index, :, :, :] = imagedata3


        ########-----------------------------------###################

        label_batch[batch_index, :] = labels_dict[file_id]

        batch_index += 1

        if batch_index >= batch_size:
            batch_index = 0
            inputs1 = spect_batch1
            inputs2 = spect_batch2
            inputs3 = spect_batch3
            inp=[inputs1, inputs2, inputs3]
            outputs = [label_batch]
            yield inp, outputs


def testdata(filelistpath, test_size):
    image_index = -1

    filelist = open(filelistpath, 'r')
    filenames = filelist.readlines()
    filelist.close()

    dataset = (['Chernobyl.csv', 'PolandNFC.csv', 'warblrb10k-eval.csv'])
 
    labels_dict = {}
    for n in range(len(dataset)):
        labels_list = csv.reader(open(LABELPATH + dataset[n], 'r'))
        next(labels_list)
        for k, r, v in labels_list:
            labels_dict[r + '/' + k] = v

    label_batch = np.zeros([int(test_size), 1])

    for m in range(len(filenames)):
        image_index = (image_index + 1) % len(filenames)

        file_id = filenames[image_index].rstrip()

        label_batch[image_index, :] = labels_dict[file_id]

        outputs = [label_batch]

    return outputs

train_generator = dataval_generator(train_filelist, BATCH_SIZE, True)
validation_generator = dataval_generator(val_filelist, BATCH_SIZE, False)
test_generator = dataval_generator(test_filelist, BATCH_SIZE, False)


################################################
#
#   Model Creation
#
################################################

model1 = load_model('trained_model/high_frequency/flmdl.h5')
model2 = load_model('trained_model/baseline/flmdl.h5')
model3 = load_model('trained_model/high_temporal/flmdl.h5')

input1 = Input(shape = (624, 160, 1), name='inp1')

x = Conv2D(16, (3, 3), padding='valid') (input1)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=.001)(x)
x = MaxPooling2D(pool_size=(3, 3))(x)
x = Conv2D(16, (3, 3), padding='valid')(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=.001)(x)
x = MaxPooling2D(pool_size=(3, 3))(x)
x = Conv2D(16, (3, 3), padding='valid')(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=.001)(x)
x = MaxPooling2D(pool_size=(3, 1))(x)
x = Conv2D(16, (3, 3), padding='valid', kernel_regularizer=l2(0.01))(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=.001)(x)
x = MaxPooling2D(pool_size=(3, 1))(x)
x = Flatten()(x)
x = Dropout(0.5)(x)
x = Dense(256)(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=.001)(x)
x = Dropout(0.5)(x)
y1 = Dense(32)(x)
x = BatchNormalization()(y1)
x = LeakyReLU(alpha=.001)(x)
x = Dropout(0.5)(x)
x = Dense(1, activation='sigmoid')(x)

model = Model(inputs = input1, outputs = x)

for layer in model.layers:
    layer.name = layer.name + str("_1_1")

ctr = 0  # starting from 1
for layer in model1.layers:
    l_t = model1.layers[ctr]
    layer_model = model.layers[ctr+1]
    layer_model.set_weights(l_t.get_weights())
    layer_model.trainable = False
    #print(ctr)
    #print(l_t.name)
    #print('Debug')
    ctr += 1


################################
# MODEL 2
input2 = Input(shape = (700, 80, 1), name = 'inp2')
xx = Conv2D(16, (3, 3), padding='valid') (input2)
xx = BatchNormalization()(xx)
xx = LeakyReLU(alpha=.001)(xx)
xx = MaxPooling2D(pool_size=(3, 3))(xx)
xx = Conv2D(16, (3, 3), padding='valid')(xx)
xx = BatchNormalization()(xx)
xx = LeakyReLU(alpha=.001)(xx)
xx = MaxPooling2D(pool_size=(3, 3))(xx)
xx = Conv2D(16, (3, 3), padding='valid')(xx)
xx = BatchNormalization()(xx)
xx = LeakyReLU(alpha=.001)(xx)
xx = MaxPooling2D(pool_size=(3, 1))(xx)
xx = Conv2D(16, (3, 3), padding='valid', kernel_regularizer=l2(0.01))(xx)
xx = BatchNormalization()(xx)
xx = LeakyReLU(alpha=.001)(xx)
xx = MaxPooling2D(pool_size=(3, 1))(xx)
xx = Flatten()(xx)
xx = Dropout(0.5)(xx)
xx = Dense(256)(xx)
xx = BatchNormalization()(xx)
xx = LeakyReLU(alpha=.001)(xx)
xx = Dropout(0.5)(xx)
y2 = Dense(32)(xx)
xx = BatchNormalization()(y2)
xx = LeakyReLU(alpha=.001)(xx)
xx = Dropout(0.5)(xx)
xx = Dense(1, activation='sigmoid')(xx)

model2_new = Model(inputs = input2, outputs = xx)

for layer in model2_new.layers:
    layer.name = layer.name + str("_2_2")


ctr = 0  # starting from 1
for layer in model2.layers:
    l_t = model2.layers[ctr]
    layer_model = model2_new.layers[ctr+1]
    layer_model.set_weights(l_t.get_weights())
    layer_model.trainable = False
    #print(ctr)
    #print(l_t.name)
    #print('Debug')
    ctr += 1



################################
# MODEL 3
input3 = Input(shape = (1669, 80, 1))
xxx = Conv2D(16, (3, 3), padding='valid') (input3)
xxx = BatchNormalization()(xxx)
xxx = LeakyReLU(alpha=.001)(xxx)
xxx = MaxPooling2D(pool_size=(3, 3))(xxx)
xxx = Conv2D(16, (3, 3), padding='valid')(xxx)
xxx = BatchNormalization()(xxx)
xxx = LeakyReLU(alpha=.001)(xxx)
xxx = MaxPooling2D(pool_size=(3, 3))(xxx)
xxx = Conv2D(16, (3, 3), padding='valid')(xxx)
xxx = BatchNormalization()(xxx)
xxx = LeakyReLU(alpha=.001)(xxx)
xxx = MaxPooling2D(pool_size=(3, 1))(xxx)
xxx = Conv2D(16, (3, 3), padding='valid', kernel_regularizer=l2(0.01))(xxx)
xxx = BatchNormalization()(xxx)
xxx = LeakyReLU(alpha=.001)(xxx)
xxx = MaxPooling2D(pool_size=(3, 1))(xxx)
xxx = Flatten()(xxx)
xxx = Dropout(0.5)(xxx)
xxx = Dense(256)(xxx)
xxx = BatchNormalization()(xxx)
xxx = LeakyReLU(alpha=.001)(xxx)
xxx = Dropout(0.5)(xxx)
y3 = Dense(32)(xxx)
xxx = BatchNormalization()(y3)
xxx = LeakyReLU(alpha=.001)(xxx)
xxx = Dropout(0.5)(xxx)
xxx = Dense(1, activation='sigmoid')(xxx)

model3_new = Model(inputs = input3, outputs = xxx)

for layer in model3_new.layers:
    layer.name = layer.name + str("_3_3")


ctr = 0  # starting from 1
for layer in model3.layers:
    l_t = model3.layers[ctr]
    layer_model = model3_new.layers[ctr+1]
    layer_model.set_weights(l_t.get_weights())
    layer_model.trainable = False
    #print(ctr)
    #print(l_t.name)
    #print('Debug')
    ctr += 1

combined_feat = Concatenate()([y1, y2, y3])
xf = Dense(32)(combined_feat)
xf = BatchNormalization()(xf)
xf = LeakyReLU(0.1)(xf)

xf = Dense(16)(xf)
xf = BatchNormalization()(xf)
xf = LeakyReLU(0.1)(xf)

yf = Dense(1, activation='sigmoid', name = 'final_output')(xf)

#final_model = Model(inputs=[input2, input3], outputs=y)
final_model = Model(inputs=[input1, input2, input3], outputs=yf)

adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
final_model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['acc'])

histories = my_callbacks.Histories()

final_model.summary()

print('Debug')

my_steps = np.floor(TRAIN_SIZE*AUGMENT_SIZE / BATCH_SIZE)
my_val_steps = np.floor(VAL_SIZE / BATCH_SIZE)
my_test_steps = np.floor(TEST_SIZE / BATCH_SIZE)

history = final_model.fit_generator(
        train_generator,
        steps_per_epoch=my_steps,
        epochs=EPOCH_SIZE,
        validation_data=validation_generator,
        validation_steps=my_val_steps,
        callbacks= [checkPoint, reduceLR, csvLogger],
        class_weight= training_set[k_CLASS_WEIGHT],
        verbose=True)

final_model.save(final_model_name)
