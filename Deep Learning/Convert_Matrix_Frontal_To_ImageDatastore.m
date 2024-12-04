%% Load data
clear; close all; clc
load('C:\Users\ADMIN\Downloads\thermography_data_base.mat')
% Define folders
CtrlFolder = 'C:\Users\ADMIN\Documents\Universidad\Doctorado\Termogramas\Termogramas Transfer Learning\Sanos';
AbnormalFolder = 'C:\Users\ADMIN\Documents\Universidad\Doctorado\Termogramas\Termogramas Transfer Learning\Enfermos';
 
%% Write images in corresponding folders
nImages = size(imageMatrix,3);
ioi_text_waitbar(0, 'Please wait...');
for idxImages = 1:nImages
    if strcmp(orientation(idxImages),'Fr')
        % Read
        testImg = squeeze(imageMatrix(:,:,idxImages));
        % Normalize between 0 and 1
        testImg = testImg-min(testImg(:));
        testImg = testImg/max(testImg(:));
        % Resize to 227x227 (not necessary if ReadFcn is added)
%         testImg = imresize(testImg, [227, 227]);
        % Expand to RGB
        testImg = repmat(testImg,[1 1 3]);
        % Select folder and filename
        if strcmp(labels(idxImages),'En')
            folderName = AbnormalFolder;
            fileName = sprintf('Abnormal_%03d.tif',idxImages);
        else
            folderName = CtrlFolder;
            fileName = sprintf('Normal_%03d.tif',idxImages);
        end
        % Write TIF
        imwrite(testImg,fullfile(folderName,fileName));
        clear testImg
        % Update progress bar
        ioi_text_waitbar(idxImages/nImages, sprintf('Writing image %d from %d', idxImages, nImages));
    end
end
ioi_text_waitbar('Clear');
 
% EOF