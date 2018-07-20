function [matdata, labels] = readfromh5(dataset_name)
dataset = ['BirdVox-DCASE-20k','ff1010bird','warblrb10k'];

BATCH_SIZE = 16;

fid = fopen(dataset_name+'.csv'); %Opens the file 
filelist = textscan(fid, '%s %s %d', 'delimiter',','); 
fclose(fid); %Closes the file

filelabels = filelist{3};

if(dataset_name=='BirdVox-DCASE-20k'||dataset_name=='warblrb10k')
    filename=cell2mat(filelist{1});
    pathprefix = '/home/sidrah/DL/bulbul2018/workingfiles/spect/';
elseif(dataset_name=='ff1010bird')
    filename=cellfun(@str2num,filelist{1}); 
    pathprefix = '/home/sidrah/DL/bulbul2018/workingfiles/spect/';
end
%for(index=1:length(filename))
for index=1:BATCH_SIZE
    fn = pathprefix+dataset_name+'/'+(filename(index,:))+'.wav.h5';
    readdata = hdf5read(char(fn),'features');
    N = size(readdata,2);
    matdata(:,(index-1)*N+1:index*N)= readdata;
    labels((index-1)*N+1:index*N,1)=repmat(filelabels(index),[N,1]);
end

end
