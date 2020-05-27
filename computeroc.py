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

################################################
#
#   Global parameters
#
################################################

SPECTPATH1 = '/ukybirddet/workingfiles/win_32ms/'
SPECTPATH2 = '/ukybirddet/workingfiles/spect/'
SPECTPATH3 = '/ukybirddet/workingfiles/80fbanks/win_12ms/'

LABELPATH = '/ukybirddet/labels/'

FILELIST = '/ukybirddet/workingfiles/filelists/'

#dataset = ['BirdVox-DCASE-20k.csv', 'ff1010bird.csv', 'warblrb10k.csv']
dataset = (['Chernobyl.csv', 'PolandNFC.csv', 'warblrb10k-eval.csv'])

BATCH_SIZE = 16
EPOCH_SIZE = 30
AUGMENT_SIZE = 1
with_augmentation = False
features1='mfc'
features2='h5'
features3='mfc'
model_operation = 'new'
TEST_SIZE = 12620.0
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

    filelist = open(filelistpath[0], 'r')
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
            hf = h5py.File(SPECTPATH1 + file_id[:-4] + '.h5', 'r')
            imagedata1 = hf.get('features')
            imagedata1= np.array(imagedata1)
            hf.close()
            imagedata1 = (imagedata1 + 15.0966)/(15.0966 + 2.25745)

        elif features1 == 'mfc':
            htk_reader = HTKFile()
            htk_reader.load(SPECTPATH1 + file_id[:-8] + '.mfc')
            imagedata1 = np.array(htk_reader.data)
            imagedata1 = imagedata1/18.0

        imagedata1 = correct_dimensions(imagedata1, expected_shape1)
        imagedata1 = np.reshape(imagedata1, (1, imagedata1.shape[0], imagedata1.shape[1], 1))
        spect_batch1[batch_index, :, :, :] = imagedata1

        ####### feature matrix for network 2 ######################
        if features2 == 'h5':
            hf = h5py.File(SPECTPATH2 + file_id[:-4] + '.h5', 'r')
            imagedata2 = hf.get('features')
            imagedata2 = np.array(imagedata2)
            hf.close()
            imagedata2 = (imagedata2 + 15.0966) / (15.0966 + 2.25745)

        elif features2 == 'mfc':
            htk_reader = HTKFile()
            htk_reader.load(SPECTPATH2 + file_id[:-8] + '.mfc')
            imagedata2 = np.array(htk_reader.data)
            imagedata2 = imagedata2 / 18.0

        imagedata2 = correct_dimensions(imagedata2, expected_shape2)
        imagedata2 = np.reshape(imagedata2, (1, imagedata2.shape[0], imagedata2.shape[1], 1))
        spect_batch2[batch_index, :, :, :] = imagedata2

        ####### feature matrix for network 3 ######################
        if features3 == 'h5':
            hf = h5py.File(SPECTPATH3 + file_id[:-4] + '.h5', 'r')
            imagedata3 = hf.get('features')
            imagedata3 = np.array(imagedata3)
            hf.close()
            imagedata3 = (imagedata3 + 15.0966) / (15.0966 + 2.25745)

        elif features3 == 'mfc':
            htk_reader = HTKFile()
            htk_reader.load(SPECTPATH3 + file_id[:-8] + '.mfc')
            imagedata3 = np.array(htk_reader.data)
            imagedata3 = imagedata3 / 18.0

        imagedata3 = correct_dimensions(imagedata3, expected_shape3)
        imagedata3 = np.reshape(imagedata3, (1, imagedata3.shape[0], imagedata3.shape[1], 1))
        spect_batch3[batch_index, :, :, :] = imagedata3


        ########-----------------------------------###################

        

        batch_index += 1

        if batch_index >= batch_size:
            batch_index = 0
            inputs1 = spect_batch1
            inputs2 = spect_batch2
            inputs3 = spect_batch3
            inp=[inputs1, inputs2, inputs3]
            
            yield inp


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

test_filelist=[FILELIST+'test']
MODELPATH='/multimodel/backup/fulltrain/ckpt.h5'
model = load_model(MODELPATH)
print('Model: '+ MODELPATH + ' --- Test file: test')

model.summary()

my_test_steps = np.ceil(TEST_SIZE / BATCH_SIZE)

pred_generator = dataval_generator(test_filelist, BATCH_SIZE, False)

y_pred = model.predict_generator(
    pred_generator,
    steps=my_test_steps)

print(y_pred)

testfile = open(test_filelist[0],'r')
testfilenames = testfile.readlines()
testfile.close()

fidwr = open('DCASE_parallel_submission','wt')
try:
    writer = csv.writer(fidwr)
    for i in range(len(testfilenames)):
        strf = testfilenames[i]
        writer.writerow((strf[strf.find('/')+1:-9], str(float(y_pred[i]))))
finally:
    fidwr.close()
