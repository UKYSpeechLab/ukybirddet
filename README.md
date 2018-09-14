**Bird Audio Detection Challenge 2018 - DCASE Task 3**

This is the submission for DCASE 2018 Task 3 by UKYSpeechLab

**Required Software:**

- Python version 3.5
- Tensorflow-gpu (tested on v1.8)
- Keras
- other packages and dependencies listed in [requirements.txt](https://github.com/UKYSpeechLab/ukybirddet/blob/master/requirements.txt)

**Project directory structure**

For running this project on the DCASE challenge 2018 data, follow this directory structure:

- <project directory>/workingfiles/features_baseline [download link](https://drive.google.com/drive/folders/1Zf8LQxZF9KISByGmmxx-dbtHLc5dk9Ib?usp=sharing)
- <project directory>/workingfiles/features_high_frequency [download link](https://drive.google.com/open?id=14DJczxbwCv0Z7uFBfDDUVv6a3cZ3SAu4)
- <project directory>/workingfiles/features_high_temporal [download link](https://drive.google.com/open?id=1rPwJL8Y0EPPbh6V-HhTlvcyqnFJOQmfI)
- <project directory>/workingfiles/features_ht_enhanced [download link]

In each of these four directories, place the feature files from the provided download links such that each directory should have following six sub-directories:

- "BirdVox-DCASE-20k" (20,000 files)
- "Chernobyl" (6620 files)
- "ff1010bird" (7690 files)
- "PolandNFC" (4000 files)
- "warblrb10k" (8000 files)
- "warblrb10k-eval" (2000 files)

In order to reproduce the results of this submission, place the python files in the main project directory and run in the following order:
- birddet_baseline.py
- birddet_high_temporal.py
- birddet_high_frequency.py
- birddet_adaptation.py
- birddet_enhancement.py
- birddet_multimodel.py
- computeroc.py

As a result of running the above codes, six prediction files will be generated in the "prediction" directory. These are csv format files which have the same format that is the required format of DCASE challenge submission.

Average of above six prediction files:

Our top scoring submission to the DCASE challenge is a simple unweighted average of the above six predictions. 
Run the python code "compute_ave.py" from the main project directory after all of the following six prediction files have been generated.
- DCASE_submission_baseline.csv
- DCASE_submission_high_frequency.csv
- DCASE_submission_high_temporal.csv
- DCASE_submission_adaptation.csv
- DCASE_submission_enhancement.csv
- DCASE_submission_multimodel.csv
- Compute_ave.py will generate "DCASE_submission_final.csv" which is the final prediction file.
