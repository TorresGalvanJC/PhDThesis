%% Brasil Data Base
clear; close all; clc
load('C:\Users\ADMIN\Documents\Universidad\Doctorado\Termogramas\Termogramas Base Datos Brasil\Matríz de Termogramas 3 Vistas\thermography_Brasil_data_base.mat')

% Define folders
CtrlFolder = 'C:\Users\ADMIN\Documents\Universidad\Doctorado\Termogramas\Termogramas Transfer Learning\Normal';
AbnormalFolder = 'C:\Users\ADMIN\Documents\Universidad\Doctorado\Termogramas\Termogramas Transfer Learning\Abnormal';
 
%% Write images in corresponding folders
nImages = size(BraImageMatrix,3);
ioi_text_waitbar(0, 'Please wait...');
for idxImages = 1:nImages
    if strcmp(BraOrientation(idxImages),'Fr')
        % Read
        testImg = squeeze(BraImageMatrix(:,:,idxImages));
        % Normalize between 0 and 1
        testImg = testImg-min(testImg(:));
        testImg = testImg/max(testImg(:));
        % Resize to 227x227 (not necessary if ReadFcn is added)
%         testImg = imresize(testImg, [227, 227]);
        % Expand to RGB
        testImg = repmat(testImg,[1 1 3]);
        % Select folder and filename
        if strcmp(BraLabels(idxImages),'En')
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
 
%% IJC Data Base

clear; close all; clc
load('C:\Users\ADMIN\Documents\Universidad\Doctorado\Termogramas\Termogramas IJC\Matrices Temperatura IJC\thermography_KnownPatients_data_base_IJC.mat')

% Define folders
CtrlFolder = 'C:\Users\ADMIN\Documents\Universidad\Doctorado\Termogramas\Termogramas Transfer Learning\Normal';
AbnormalFolder = 'C:\Users\ADMIN\Documents\Universidad\Doctorado\Termogramas\Termogramas Transfer Learning\Abnormal';
 
%% Write images in corresponding folders
nImages = size(IJC_ImageMatrix,3);
ioi_text_waitbar(0, 'Please wait...');
for idxImages = 1:nImages
    if strcmp(IJC_Orientation(idxImages),'Fr')
        % Read
        testImg = squeeze(IJC_ImageMatrix(:,:,idxImages));
        % Normalize between 0 and 1
        testImg = testImg-min(testImg(:));
        testImg = testImg/max(testImg(:));
        % Resize to 227x227 (not necessary if ReadFcn is added)
%         testImg = imresize(testImg, [227, 227]);
        % Expand to RGB
        testImg = repmat(testImg,[1 1 3]);
        % Select folder and filename
        if strcmp(IJC_Labels(idxImages),'En')
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