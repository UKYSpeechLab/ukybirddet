%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This script will loop through the datasets specified in the
% source_datasets variable and compute the transform matrix that is
% required by the covariance adaptation method. The transform matrix is
% output for each dataset in an appropriately named .h5 file in the
% directory that this script is placed. 
%
% A number of parameters need to be changed to fit your development
% environment. They are all listed in the GLOBAL PARAMETERS section.
%
% To declare a new dataset that this script should process, simply make a
% struct with the keys 'dataset_name' and 'output_matrix_name'
% corresponding to the directory's name that contains the dataset and the
% name you want the output transform matrix to be, respectively. Then, add 
% that struct to the array 'source_datasets'.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%
%   GLOBAL PARAMETERS
%

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
target_cov_name = 'transform_target_warblrb10k.h5';

% The base name for the output transform matrix
source_cov_name_base = "transform_source_";

% Declare dataset dictionaries
d_birdvox.dataset_name = "BirdVox-DCASE-20k";
d_birdvox.output_matrix_name = char(source_cov_name_base + d_birdvox.dataset_name + ".h5");

d_ff.dataset_name = "ff1010bird";
d_ff.output_matrix_name = char(source_cov_name_base + d_ff.dataset_name + ".h5");

source_datasets = [d_birdvox, d_ff];

%
%   COMPUTE COVARIANCE OF TARGET SET
%

% We only want to compute the transform matrix of the target set once,
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
%   COMPUTE TRANSFORMS OF SOURCES
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

    %   COLLECT SPECTROGRAMS INTO ONE MATRIX

    for j = 1 : BATCH_SIZE
        fn = spex_path_prefix + source_dataset_name + '/' + (filenames(j,:)) + '.wav.h5';
        readdata = hdf5read(char(fn), '/features');
        N = size(readdata,2);
        matdata(:,(j-1) * N+1 : j * N) = readdata;
    end

    %   COMPUTE TRANSFORM MATRIX AND WRITE

    % Compute transform of source distribution and fetch covariance of 
    % target distribution from file
    cov_s = cov(matdata');
    c_s_unscaled = cov_s + eye(size(cov_s, 2));
    c_s = c_s_unscaled ^ (-1/2);
    A = c_s * c_t; % This is the transform matrix.

    hdf5write(source_datasets(index).output_matrix_name, '/cov', A);
    
end
