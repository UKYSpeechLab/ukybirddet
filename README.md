Bird Audio Detection Challenge 2018 - DCASE Task 3

This is the submission for DCASE 2018 Task 3 by UKYSpeechLab

**Required Software:**
- Python version 3.5
- Tensorflow-gpu (tested on v1.8)
- Keras

**Other packages and dependencies:**
- numpy
- csv
- h5py
- scikit-learn

**Selecting Datasets for Processing:**

There are three datasets: birdvox, ff, and warblr, represented by the dictionaries `d_birdVox`, `d_ff`, and `d_warblr` respectively. If you want to train, test, or validate the model with different combinations of these sets, look to lines 94-96, specifically: 

`training_set = d_ff`

`validation_set = d_ff`

`test_set = d_birdVox`

**Domain Adaptation:**

We are following the approach outlined in http://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/download/12443/11842

To see it in action, check out `adapt.m`. It is currently set up to compute the covariance matrices for the Warlbr, FarField, and BirdVox datasets. These matrices are output as `.h5` files.
