%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This script will loop through the datasets specified in the
% source_datasets variable and compute the transform matrix that is
% required by the covariance adaptation method. The transform matrix is
% output for each dataset in an appropriately named .h5 file in the
% directory that this script is placed. The script will compute the final
% transform for a given dataset <var: ITERATIONS> number of times. You can
% set the parameter DEBUG to true so that the differences between
% successive transform matrices are plotted.
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

close all;
clear;

%
%   GLOBAL PARAMETERS
%

% If DEBUG is set to true, the script will output images corresponding to
% the differences between the output matrices of sequential passes (the 
% number of which is defined in ITERATIONS). 
DEBUG = true;

% BATCH_SIZE sets the number of spectrograms that will be "merged" into
% one, of which we will then compute the covariance matrix.
BATCH_SIZE = 128;

% Set the target dataset. This set will be used to compute the covaraince
% matrix for the target.
target_dataset_name = "warblrb10k";

% path prefixes should be the paths to the specified object
%spex_path_prefix = '~/Documents/DCASE2018/tensorflow/ukybirddet/spect/';
%flist_path_prefix = '~/Documents/DCASE2018/tensorflow/ukybirddet/labels/';
spex_path_prefix = '/home/sidrah/DL/bulbul2018/workingfiles/spect/';
flist_path_prefix = '/home/sidrah/DL/bulbul2018/labels/';

features = 'h5';
% can be h5 or mfc

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

d_chern.dataset_name = "Chernobyl";
d_chern.output_matrix_name = char(source_cov_name_base + d_chern.dataset_name + ".h5");

d_poland.dataset_name = "PolandNFC";
d_poland.output_matrix_name = char(source_cov_name_base + d_poland.dataset_name + ".h5");

source_datasets = [d_birdvox, d_ff, d_chern, d_poland];

ITERATIONS = 8;

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
        if(strcmp(features,'h5'))
            fn = spex_path_prefix + target_dataset_name + '/' + (filenames(index,:)) + '.wav.h5';        
            readdata = hdf5read(char(fn), '/features');
            readdata = readdata';
        elseif(strcmp(features, 'mfc'))
            fn = spex_path_prefix + target_dataset_name + '/' + (filenames(index,:)) + '.mfc';    
            readdata = readhtk('fn');
        end
        normalized = normalize_features(readdata);
        N = size(normalized,1);
        matdata((index-1) * N+1 : index * N,:) = normalized;
    end
    
    cov_t = cov(matdata);
    c_t_unscaled = cov_t + eye(size(cov_t, 2));
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

    %if(source_dataset_name == 'BirdVox-DCASE-20k' || source_dataset_name == 'warblrb10k' || source_dataset_name == 'Chernobyl')
    %    filenames=cell2mat(filelist);
    %elseif(source_dataset_name=='ff1010bird')
    %    filenames=cellfun(@str2num, filelist); 
    %elseif(source_dataset_name=='PolandNFC')
    %    filenames=char(filelist); 
    filenames=char(filelist);
    filenames=strtrim(filenames);
    %end
 
    for iter = 1:ITERATIONS
        
        % Shuffle filenames by randomly selecting BATCH_SIZE files
        rand_selected_files = datasample(filenames, BATCH_SIZE);
        %disp(rand_selected_files(1:3,:));

        %   COLLECT SPECTROGRAMS INTO ONE MATRIX

        for j = 1 : size(rand_selected_files, 1)
            if(source_dataset_name == 'BirdVox-DCASE-20k' || source_dataset_name == 'ff1010bird')
                fn = spex_path_prefix + source_dataset_name + '/' + strtrim(rand_selected_files(j,:)) + '.wav.h5';
            elseif(source_dataset_name == 'Chernobyl' || source_dataset_name == 'PolandNFC')
                fn = spex_path_prefix + source_dataset_name + '/' + strtrim(rand_selected_files(j,:)) + '.h5';
            end
            
            if(strcmp(features,'h5'))
                fn = spex_path_prefix + target_dataset_name + '/' + (filenames(index,:)) + '.wav.h5';        
                readdata = hdf5read(char(fn), '/features');
                readdata = readdata';
            elseif(strcmp(features, 'mfc'))
                fn = spex_path_prefix + target_dataset_name + '/' + (filenames(index,:)) + '.mfc';    
                readdata = readhtk('fn');
            end
            
            normalized = normalize_features(readdata);
            N = size(normalized,1);
            matdata((j-1) * N+1 : j * N,:) = normalized;
        end

        %   COMPUTE TRANSFORM MATRIX AND WRITE

        % Compute transform of source distribution and fetch covariance of 
        % target distribution from file
        cov_s = cov(matdata);
        c_s_unscaled = cov_s + eye(size(cov_s, 2));
        c_s = c_s_unscaled ^ (-1/2);
        A = c_s * c_t; % This is the transform matrix.

        % Compare A with the previous matrix
        if iter == 1
            current_mat = A;
        else
            difference = current_mat - A;
            %fprintf('%s: Difference from ITER %d and ITER %d: %f\n', source_dataset_name, iter - 1, iter, mean(mean(difference)));
            current_mat = A;
        end
        
        hdf5write(source_datasets(index).output_matrix_name, '/cov', A);
        
        h = figure(index);
        %title('Differences of Covariance Transformations for ' + source_dataset_name);
        
        % Debug differences with subplots
        if DEBUG
            if iter ~= 1
                subplot(ceil(ITERATIONS / 2), 2, iter - 1);
                imagesc(difference);
                title({source_dataset_name, 'Pass' num2str(iter)});
                colorbar;
            end
        end
    
    end
    
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This function will return a matrix. It is expecting `mat` to be a matrix
% of features where each feature is indexed as a column (i.e., 700 X 80)
% would refer to 700 examples and 80 features.
%
% We will normalize all of the features so that they sum to 1 by taking
% summing over the features and creating a scale-vector, `lambda`. We will 
% then repeat the scale vector to form a matrix of the same size as the
% input parameter matrix. Next, we scale the input parameter matrix by
% dividing it by the scaling matrix. We then take the z-score of the scaled
% input and return the result.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function A = normalize_features(mat)
    
    % Sum the features.
    lambda = sum(mat, 2);
    
    % Tile the lambda vector (this is mainly for easy division).
    lambda_extended = repmat(lambda, 1, size(mat, 2));
    
    % Scale the input matrix
    scaled = mat ./ lambda_extended; 

    % Compute z-scale
    A = zscore(scaled, 1);
    
end
