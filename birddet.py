# DCASE 2018 - Bird Audio Detection challenge (Task 3)

# This code is a basic implementation of bird audio detector (based on baseline code's architecture)

import h5py
import csv
import numpy as np
import random
import PIL.Image
import matplotlib.pyplot as plt
from HTK import HTKFile

from sklearn.metrics import roc_auc_score, roc_curve, auc

import keras
from keras.layers import Conv2D, Dropout, MaxPooling2D, Dense, GlobalAveragePooling2D, Flatten, BatchNormalization
from keras.models import Sequential, load_model, Input
from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator
from keras.losses import binary_crossentropy, mean_squared_error, mean_absolute_error
from keras.regularizers import l2
from keras.applications import ResNet50
from keras import Model

from keras import layers
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import GlobalMaxPooling2D
from keras.layers import ZeroPadding2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import BatchNormalization
from keras.models import Model
from keras.preprocessing import image
import keras.backend as K
from keras.utils import layer_utils

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
#SPECTPATH = '/audio/audio/mfcfeatures/'
SPECTPATH = '/audio/audio/workingfiles/spect/'
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
logfile_name = 'test.log'
checkpoint_model_name = 'test_ckpt.h5'
final_model_name = 'test_flmdl.h5'

BATCH_SIZE = 16
EPOCH_SIZE = 20
AUGMENT_SIZE = 8
with_augmentation = False
features='h5'
model_operation = 'new'
# model_operations : 'new', 'load', 'test'
shape = (700, 80)
#shape = (1669, 160)
expected_shape = (700, 80)
spect = np.zeros(shape)
label = np.zeros(1)

# Callbacks for logging during epochs
reduceLR = ReduceLROnPlateau(factor=0.2, patience=5, min_lr=0.00001)
checkPoint = ModelCheckpoint(filepath = checkpoint_model_name, save_best_only=True)   # criterion = 'val_acc', mode = 'max'
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
training_set = d_warblr
validation_set = d_warblr
test_set = d_warblr

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

    # shuffling filelist
    if shuffle==True:
        random.shuffle(filenames)

    dataset = ['BirdVox-DCASE-20k.csv', 'ff1010bird.csv', 'warblrb10k.csv']

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
            # normalizing intensity values of spectrogram from [-15.0966 to 2.25745] to [0 to 1] range
            imagedata = (imagedata + 15.0966)/(15.0966 + 2.25745)
        elif features == 'mfc':
            htk_reader = HTKFile()
            htk_reader.load(SPECTPATH + file_id.rstrip('.wav') + '.mfc')
            imagedata = np.array(htk_reader.data)

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
            spect_batch = np.zeros([batch_size, spect.shape[0], spect.shape[1], 3])
            label_batch = np.zeros([batch_size, 1])

        if features == 'h5':
            hf = h5py.File(SPECTPATH + file_id + '.h5', 'r')
            imagedata = hf.get('features')
            imagedata = np.array(imagedata)
            hf.close()

            # normalizing intensity values of spectrogram from [-15.0966 to 2.25745] to [0 to 1] range
            imagedata = (imagedata + 15.0966)/(15.0966 + 2.25745)

        elif features == 'mfc':
            htk_reader = HTKFile()
            htk_reader.load(SPECTPATH + file_id.rstrip('.wav') + '.mfc')
            imagedata = np.array(htk_reader.data)

        # processing files with shapes other than expected shape in warblr dataset
        old_imagedata = imagedata
        imagedata = np.zeros(expected_shape)

        if old_imagedata.shape[0] < expected_shape[0]:

            diff_in_frames = expected_shape[0] - old_imagedata.shape[0]
            if diff_in_frames < expected_shape[0] / 2:
                imagedata = np.vstack((old_imagedata, old_imagedata[
                    range(old_imagedata.shape[0] - diff_in_frames, old_imagedata.shape[0])]))

                # print(['PADDING on file: ' + str(k) + 'Prev duration: ' + str(imagedata.shape[0]) + '  New duration: ' + str(new_imagedata.shape[0])])
            elif diff_in_frames > expected_shape[0] / 2:
                count = np.floor(expected_shape[0] / old_imagedata.shape[0])
                remaining_diff = (expected_shape[0] - old_imagedata.shape[0] * int(count))
                imagedata = np.vstack(([old_imagedata] * int(count)))
                imagedata = np.vstack(
                    (imagedata, old_imagedata[range(old_imagedata.shape[0] - remaining_diff, old_imagedata.shape[0])]))

                print(['PADDING on file: ' + str(k) + 'Prev duration: ' + str(imagedata.shape[0]) + '  New duration: ' + str(imagedata.shape[0])])

            else:
                print(['Unchanged file: ' + str(k) + 'Prev duration: ' + str(old_imagedata.shape[0])])

        elif old_imagedata.shape[0] > expected_shape[0]:
            diff_in_frames = old_imagedata.shape[0] - expected_shape[0]

            if diff_in_frames < expected_shape[0] / 2:
                imagedata[range(0, diff_in_frames + 1), :] = np.mean(np.array([old_imagedata[range(0, diff_in_frames + 1), :],old_imagedata[range(old_imagedata.shape[0] - diff_in_frames - 1, old_imagedata.shape[0]), :]]),axis=0)
                imagedata[range(diff_in_frames + 1, expected_shape[0]), :] = old_imagedata[range(diff_in_frames + 1, expected_shape[0])]

            elif diff_in_frames > expected_shape[0] / 2:
                count = int(np.floor(old_imagedata.shape[0] / expected_shape[0]))
                remaining_diff = (old_imagedata.shape[0] - expected_shape[0] * count)
                for index in range(0, count):
                    imagedata[range(0, expected_shape[0]), :] = np.sum([imagedata, old_imagedata[range(index * expected_shape[0], (index + 1) * expected_shape[0])]],axis=0) / count
                    imagedata[range(0, remaining_diff), :] = np.mean(np.array([old_imagedata[range(old_imagedata.shape[0] - remaining_diff, old_imagedata.shape[0]), :],imagedata[range(0, remaining_diff), :]]), axis=0)

                print(['PADDING on file: ' + str(k) + 'Prev duration: ' + str(imagedata.shape[0]) + '  New duration: ' + str(imagedata.shape[0])])
            else:
                print(['Unchanged file: ' + str(k) + 'Prev duration: ' + str(old_imagedata.shape[0])])

        imagedata = np.reshape(imagedata, (1, imagedata.shape[0], imagedata.shape[1], 1))

        spect_batch[batch_index, :, :, 0] = imagedata
        spect_batch[batch_index, :, :, 1] = imagedata
        spect_batch[batch_index, :, :, 2] = imagedata
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

def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x

if model_operation == 'new':
    my_input = Input(shape=(700, 80, 3))

    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = ZeroPadding2D((3, 3))(my_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    # Till activation_40: layer number=141

    x = Conv2D(32, (1,1), activation='relu', padding='same')(x)

    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(400, activation='relu')(x)
    x = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=my_input, outputs=x)

    model.summary()

    base_model = ResNet50(include_top=False, input_shape=(700,700,3))
    #base_model.summary()

    #my_model = Model(inputs=base_model.input, outputs=)

    ctr = 0
    for layer in model.layers:
        if ctr < 142:
            layer.trainable = False
            temp_wt = base_model.layers[ctr].get_weights()
            layer.set_weights(temp_wt)
            print('layer number ' + str(ctr))
            print(layer.name)
        ctr += 1

    print('Debug')

elif model_operation == 'load' or model_operation == 'test':
    model = load_model('test_ckpt.h5')

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
fpr, tpr, threshold = roc_curve(y_test[0:int(my_test_steps*BATCH_SIZE)], y_pred[0:int(my_test_steps*BATCH_SIZE)])
roc_auc = auc(fpr, tpr)
print("Total roc auc score = {0:0.4f}".format(roc_auc))
#print(history.history)

# method I: plt
plt.figure(0)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('loss.png')

plt.figure(1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Accuracy')
plt.ylabel('Accuracy %')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('acc.png')

plt.figure(2)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig('ROC_curve.png')

#plt.figure(1)



