import csv
import numpy as np

PREDICTIONPATH = 'prediction/'

test_filelist = 'workingfiles/filelists/test'

pred_baseline = csv.reader(open(PREDICTIONPATH+'DCASE_submission_baseline.csv', 'r'))
arr_baseline = {}
for k, r in pred_baseline:
    arr_baseline[k]=r

pred_high_frequency = csv.reader(open(PREDICTIONPATH+'DCASE_submission_high_frequency.csv', 'r'))
arr__high_frequency = {}
for k, r in pred_high_frequency:
    arr__high_frequency[k]=r

pred_high_temporal = csv.reader(open(PREDICTIONPATH+'DCASE_submission_high_temporal.csv', 'r'))
arr_high_temporal = {}
for k, r in pred_high_temporal:
    arr_high_temporal[k]=r

pred_adaptation = csv.reader(open(PREDICTIONPATH+'DCASE_submission_adaptation.csv', 'r'))
arr_adaptation = {}
for k, r in pred_adaptation:
    arr_adaptation[k]=r

pred_enhancement = csv.reader(open(PREDICTIONPATH + 'DCASE_submission_enhancement.csv', 'r'))
arr_enhancement = {}
for k, r in pred_enhancement:
    arr_enhancement[k] = r

pred_multimodel = csv.reader(open(PREDICTIONPATH + 'DCASE_submission_multimodel.csv', 'r'))
arr_multimodel = {}
for k, r in pred_multimodel:
    arr_multimodel[k] = r

testfile = open(test_filelist, 'r')
testfilenames = testfile.readlines()
testfile.close()

fidwr = open(PREDICTIONPATH+'DCASE_submission_final.csv', 'wt')
try:
    writer = csv.writer(fidwr)
    for i in range(len(testfilenames)):
        strf = testfilenames[i]
        strf = strf[strf.find('/')+1:-9]
        average_score = np.average([float(arr_baseline[strf]), float(arr__high_frequency[strf]),
                                    float(arr_high_temporal[strf]), float(arr_adaptation[strf]),
                                    float(arr_enhancement[strf]), float(arr_multimodel[strf])])
        writer.writerow((strf, str(float(average_score))))
finally:
    fidwr.close()