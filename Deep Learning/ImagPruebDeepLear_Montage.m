%% Imágenes Prueba Deep Learning

% We're going to do a test of Deep Learning with 
% baseball and soccer photos
clc; clear;

% Folders to use

codeFolder = 'C:\Users\ADMIN\Dropbox\Doctorado\MATLAB Codigos\Práctica';
cd(codeFolder);
ImagesFolder = 'C:\Users\ADMIN\Documents\Universidad\Doctorado\Pruebas Deep Learning\Imágenes Deporte Para Matlab';

%% Load the images that we're going to use

SportsPhotos = imageDatastore(ImagesFolder,...
    'IncludeSubfolders',true, 'FileExtensions', {'.jpg'},...
    'LabelSource','foldernames');

WatchPhoto = readimage(SportsPhotos,145);
imshow (WatchPhoto)


%% Display Class Names and Counts

InfoFoldersPhotos = countEachLabel(SportsPhotos)
categories = InfoFoldersPhotos.Label;

%% Display Sampling of Image Data

ImagesSamples = splitEachLabel(SportsPhotos,16);
montage(ImagesSamples.Files(1:16));
title(char(tbl.Label(1)));

%% Show sampling of all data
for ii = 1:4
    sf = (ii-1)*16 +1;
    ax(ii) = subplot(2,2,ii);
    montage(ImagesSamples.Files(sf:sf+3));
    title(char(tbl.Label(ii)));
end
% expandAxes(ax); % this is an optional feature, 
% you can download this from the fileexchange as well!
%% Pre-process Training Data: *Feature Extraction using Bag Of Words*
% Bag of features, also known as bag of visual words is one way to extract 
% features from images. To represent an image using this approach, an image 
% can be treated as a document and occurance of visual "words" in images
% are used to generate a histogram that represents an image.
%% Partition 700 images for training and 200 for testing
[training_set, test_set] = prepareInputFiles(imds);

