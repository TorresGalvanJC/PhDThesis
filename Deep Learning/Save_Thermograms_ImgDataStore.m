%% Image Data Store for Thermograms

% Save Thermograms in imageDataStore object from my laptop
% for save memmory in the big PC

clear
clc

codeFolder = 'C:\Users\ADMIN\Dropbox\Doctorado\MATLAB Codigos\Pr�ctica\Prueba Deep Learning';
cd(codeFolder);
ImagesFolder = 'C:\Users\ADMIN\Documents\Universidad\Doctorado\Termogramas\Termogramas Transfer Learning';
resultsFolder = 'C:\Users\ADMIN\Documents\Universidad\Doctorado\Termogramas\Matr�z Datos Concatenada';


%% Load in input images

ImagesThermo = imageDatastore(fullfile(ImagesFolder), ...
    'IncludeSubfolders',true,...
    'LabelSource', 'foldernames',...
    'FileExtensions',{'.jpg','.txt'})
tbl = countEachLabel(ImagesThermo)
save(fullfile(resultsFolder, ...
    'Thermography_imgDataStore.mat'),'ImagesThermo', 'tbl');
fprintf('Results saved to: %s\n', resultsFolder);