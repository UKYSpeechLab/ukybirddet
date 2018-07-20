%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This script will produce the covariance matrices required by the
% covariance normalization technique. In order to do this, it will select a
% dataset to consider, then select 32 spectrograms drawn from the testing
% set for the dataset. The selection process is random and reseeded on each
% iteration. It will output the final covariance matrix for each dataset
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%
%   GLOBAL PARAMETERS
%

% Setting `DEBUG = true` will do determine how convergent the covariance
% matrices for a given dataset are.
DEBUG = false;

% BATCH_SIZE sets the number of spectrograms that will be "merged" into
% one, of which we will then compute the covariance matrix.
BATCH_SIZE = 32;

% Set the target dataset. This set will be used to compute the covaraince
% matrix for the target.
target_dataset_name = "warblrb10k";

% path prefixes should be the paths to the specified object
spex_path_prefix = '~/Documents/DCASE2018/tensorflow/ukybirddet/spect/';
flist_path_prefix = '~/Documents/DCASE2018/tensorflow/ukybirddet/labels/';

% The file name for the target covariance matrix. If it doesn't exist, set
% this parameter and one will be created with the name you specify. If the
% covariance matrix has already been created, specify the path to that
% file.
target_cov_name = 'covariance_target_warblrb10k.h5';

source_cov_name_base = "covariance_source_";

% Declare dataset dictionaries
d_birdvox.dataset_name = "BirdVox-DCASE-20k";
d_birdvox.output_matrix_name = char(source_cov_name_base + d_birdvox.dataset_name + ".h5");

d_ff.dataset_name = "ff1010bird";
d_ff.output_matrix_name = char(source_cov_name_base + d_ff.dataset_name + ".h5");

source_datasets = [d_birdvox, d_ff];

%
%   COMPUTE COVARIANCE OF TARGET FILES
%

% We only want to compute the covariance matrix of the target set once,
% since it will be reused for the output all the other datasets. First, we
% will check if the target covariance file exists. If it does, great: we
% move on. If it doesn't, then we'll compute it.

if ~exist(target_cov_name, 'file')
    
    file = importdata(flist_path_prefix + target_dataset_name + '.csv');
    filelist = file.textdata(:, 1);
    filelist(1) = [];
    filenames=cell2mat(filelist);
    
    for index = 1 : BATCH_SIZE
        fn = spex_path_prefix + target_dataset_name + '/' + (filenames(index,:)) + '.wav.h5';
        readdata = hdf5read(char(fn), '/features');
        N = size(readdata,2);
        matdata(:,(index-1) * N+1 : index * N) = readdata;
    end
    
    cov_t = cov(matdata');
    c_t_unscaled = cov_t + eye(size(cov_t(2)));
    c_t = c_t_unscaled ^ (1/2);
    hdf5write(target_cov_name, '/cov', c_t);
    
end

c_t = hdf5read(target_cov_name, '/cov');

%
%   COMPUTE COVARIANCES OF SOURCES
%

for index = 1:length(source_datasets)
    
    source_dataset_name = source_datasets(index).dataset_name;
    
    %   PARSE DATA SET FILE NAMES

    file = importdata(flist_path_prefix + source_dataset_name + '.csv');
    filelist = file.textdata(:, 1);
    filelist(1) = [];

    if(source_dataset_name == 'BirdVox-DCASE-20k' || source_dataset_name == 'warblrb10k')
        filenames=cell2mat(filelist);
    elseif(source_dataset_name=='ff1010bird')
        filenames=cellfun(@str2num, filelist); 
    end

    %   COLLECT SPECTROGRAMS INTO ONE -> `matdata`

    for j = 1 : BATCH_SIZE
        fn = spex_path_prefix + source_dataset_name + '/' + (filenames(j,:)) + '.wav.h5';
        readdata = hdf5read(char(fn), '/features');
        N = size(readdata,2);
        matdata(:,(j-1) * N+1 : j * N) = readdata;
    end

    %   COMPUTE COVARIANCE MATRIX AND WRITE

    % Compute covariance of source distribution and fetch covariance of 
    % target distribution from file
    cov_s = cov(matdata');
    c_s_unscaled = cov_s + eye(size(cov_s, 2));
    c_s = c_s_unscaled ^ (-1/2);
    A = c_s * c_t;

    hdf5write(source_datasets(index).output_matrix_name, '/cov', A);
    
end
