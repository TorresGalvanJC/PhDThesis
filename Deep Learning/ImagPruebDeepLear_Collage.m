%% Imágenes Prueba Deep Learning

% We're going to do a test of Deep Learning with 
% baseball and soccer photos
clc; clear;

% LAPTOP Folders to use

% codeFolder = 'C:\Users\ADMIN\Dropbox\Doctorado\MATLAB Codigos\Práctica';
% cd(codeFolder);
% ImagesFolder = 'C:\Users\ADMIN\Documents\Universidad\Doctorado\Pruebas Deep Learning\Imágenes Deporte Para Matlab';

% BIG PC Folders to use

codeFolder = 'C:\Users\Ramón\Desktop\Juan Carlos Torres\Práctica\Prueba Deep Learning';
cd(codeFolder);
ImagesFolder = 'C:\Users\Ramón\Desktop\Juan Carlos Torres\Reducidas';

SportsPhotos2 = imageSet(ImagesFolder, 'recursive')
{SportsPhotos2.Description}
[SportsPhotos2.Count]

% For read one image, (folder), photo
B = read(SportsPhotos2(3), 45);
imshow (B)

% For a collage of all the images of folders
helperDisplayImageMontage(SportsPhotos2(1).ImageLocation(1:2:end))

% Assigning a name to the folders and obtaining their properties

AtleticoSanLuisSet = SportsPhotos2(1)
PumasSet = SportsPhotos2(2)
Sultanes = SportsPhotos2(3)
TorosTijuanaSet = SportsPhotos2(4)
Yaquis = SportsPhotos2(5)

% Partition in 2 folders
[ASL1, ASL2] = partition(AtleticoSanLuisSet, 0.7, 'randomize')

% Prepare Training Image Sets
minSetCount = min ([SportsPhotos2.Count]); % Determine the smallest amount of images in a 
SportsPhotos2_Sets = partition (SportsPhotos2, minSetCount, 'randomize');
[SportsPhotos2_Sets.Count]


%% Display Sampling of Image Data

ImagesSamples = splitEachLabel(SportsPhotos,16);
montage(ImagesSamples.Files(1:16));
title(char(tbl.Label(1)));

%% Show sampling of all data
for ii = 1:4
    sf = (ii-1)*16 +1;
    ax(ii) = subplot(2,2,ii);
    helperDisplayImageMontage(SportsPhotos2.Files(sf:sf+3));
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

