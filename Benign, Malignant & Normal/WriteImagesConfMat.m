%% Save Images of Confusion Matrix

clc; clear;
codeFolder = ('E:\Archivos Juan Carlos Torres\MATLAB\MATLAB Codigos\Benign, Malignant & Normal');
addpath('E:\Archivos Juan Carlos Torres\MATLAB\MATLAB Codigos');

%% AlexNet 500 Times
FilesFolder = ('E:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\Termogramas\Tumors\AlexNet\500 Times');
cd(FilesFolder);

dirStruct = dir(fullfile(FilesFolder, '*.mat'));
iFiles = numel(dirStruct);

ioi_text_waitbar(0, 'Please wait...')


% Starting the loop

for iiFiles = 1:iFiles
    currentFile = dirStruct(iiFiles).name;
    load([FilesFolder '\' currentFile], 'Confusion_Matrix1');
    saveas(Confusion_Matrix1, [currentFile '.jpg']);
        
    % Update progress bar
    ioi_text_waitbar(iiFiles/iFiles, sprintf('Reading AlexNet 500 times file %d from %d', iiFiles, iFiles));
        
end

ioi_text_waitbar('Clear');

clc; clear; close all

%% AlexNet Balanced Class

FilesFolder = ('E:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\Termogramas\Tumors\AlexNet\Balanced Class');
cd(FilesFolder);

dirStruct = dir(fullfile(FilesFolder, '*.mat'));
iFiles = numel(dirStruct);

ioi_text_waitbar(0, 'Please wait...')


% Starting the loop

for iiFiles = 1:iFiles
    currentFile = dirStruct(iiFiles).name;
    load([FilesFolder '\' currentFile], 'Confusion_Matrix1');
    saveas(Confusion_Matrix1, [currentFile '.jpg']);
        
    % Update progress bar
    ioi_text_waitbar(iiFiles/iFiles, sprintf('Reading AlexNet Balanced Class file %d from %d', iiFiles, iFiles));
        
end

ioi_text_waitbar('Clear');

clc; clear; close all

%% GoogleNet  500 Times
FilesFolder = ('E:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\Termogramas\Tumors\GoogleNet\500 Times');
cd(FilesFolder);

dirStruct = dir(fullfile(FilesFolder, '*.mat'));
iFiles = numel(dirStruct);

ioi_text_waitbar(0, 'Please wait...')


% Starting the loop

for iiFiles = 1:iFiles
    currentFile = dirStruct(iiFiles).name;
    load([FilesFolder '\' currentFile], 'Confusion_Matrix1');
    saveas(Confusion_Matrix1, [currentFile '.jpg']);
        
    % Update progress bar
    ioi_text_waitbar(iiFiles/iFiles, sprintf('Reading GoogleNet 500 times file %d from %d', iiFiles, iFiles));
        
end

ioi_text_waitbar('Clear');

clc; clear; close all

%% GoogleNet Balanced Class

FilesFolder = ('E:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\Termogramas\Tumors\GoogleNet\Balanced Class');
cd(FilesFolder);

dirStruct = dir(fullfile(FilesFolder, '*.mat'));
iFiles = numel(dirStruct);

ioi_text_waitbar(0, 'Please wait...')


% Starting the loop

for iiFiles = 1:iFiles
    currentFile = dirStruct(iiFiles).name;
    load([FilesFolder '\' currentFile], 'Confusion_Matrix1');
    saveas(Confusion_Matrix1, [currentFile '.jpg']);
        
    % Update progress bar
    ioi_text_waitbar(iiFiles/iFiles, sprintf('Reading GoogleNet Balanced Class file %d from %d', iiFiles, iFiles));
        
end

ioi_text_waitbar('Clear');

clc; clear; close all


%% Inception V3  500 Times
FilesFolder = ('E:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\Termogramas\Tumors\Inception V3\500 Times');
cd(FilesFolder);

dirStruct = dir(fullfile(FilesFolder, '*.mat'));
iFiles = numel(dirStruct);

ioi_text_waitbar(0, 'Please wait...')


% Starting the loop

for iiFiles = 1:iFiles
    currentFile = dirStruct(iiFiles).name;
    load([FilesFolder '\' currentFile], 'Confusion_Matrix1');
    saveas(Confusion_Matrix1, [currentFile '.jpg']);
        
    % Update progress bar
    ioi_text_waitbar(iiFiles/iFiles, sprintf('Reading InceptionV3 500 times file %d from %d', iiFiles, iFiles));
        
end

ioi_text_waitbar('Clear');

clc; clear; close all

%% Inception V3 Balanced Class

FilesFolder = ('E:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\Termogramas\Tumors\Inception V3\Balanced Class');
cd(FilesFolder);

dirStruct = dir(fullfile(FilesFolder, '*.mat'));
iFiles = numel(dirStruct);

ioi_text_waitbar(0, 'Please wait...')


% Starting the loop

for iiFiles = 1:iFiles
    currentFile = dirStruct(iiFiles).name;
    load([FilesFolder '\' currentFile], 'Confusion_Matrix1');
    saveas(Confusion_Matrix1, [currentFile '.jpg']);
        
    % Update progress bar
    ioi_text_waitbar(iiFiles/iFiles, sprintf('Reading InceptionV3 Balanced Class file %d from %d', iiFiles, iFiles));
        
end

ioi_text_waitbar('Clear');

clc; clear; close all
%%
%% ResNet50  500 Times
FilesFolder = ('E:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\Termogramas\Tumors\ResNet 50\500 Times');
cd(FilesFolder);

dirStruct = dir(fullfile(FilesFolder, '*.mat'));
iFiles = numel(dirStruct);

ioi_text_waitbar(0, 'Please wait...')


% Starting the loop

for iiFiles = 1:iFiles
    currentFile = dirStruct(iiFiles).name;
    load([FilesFolder '\' currentFile], 'Confusion_Matrix1');
    saveas(Confusion_Matrix1, [currentFile '.jpg']);
        
    % Update progress bar
    ioi_text_waitbar(iiFiles/iFiles, sprintf('Reading ResNet50 500 times file %d from %d', iiFiles, iFiles));
        
end

ioi_text_waitbar('Clear');

clc; clear; close all

%% ResNet50 Balanced Class

FilesFolder = ('E:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\Termogramas\Tumors\ResNet 50\Balanced Class');
cd(FilesFolder);

dirStruct = dir(fullfile(FilesFolder, '*.mat'));
iFiles = numel(dirStruct);

ioi_text_waitbar(0, 'Please wait...')


% Starting the loop

for iiFiles = 1:iFiles
    currentFile = dirStruct(iiFiles).name;
    load([FilesFolder '\' currentFile], 'Confusion_Matrix1');
    saveas(Confusion_Matrix1, [currentFile '.jpg']);
        
    % Update progress bar
    ioi_text_waitbar(iiFiles/iFiles, sprintf('Reading ResNet50 Balanced Class file %d from %d', iiFiles, iFiles));
        
end

ioi_text_waitbar('Clear');

clc; clear; close all

%% ResNet101  500 Times
FilesFolder = ('E:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\Termogramas\Tumors\ResNet 101\500 Times');
cd(FilesFolder);

dirStruct = dir(fullfile(FilesFolder, '*.mat'));
iFiles = numel(dirStruct);

ioi_text_waitbar(0, 'Please wait...')


% Starting the loop

for iiFiles = 1:iFiles
    currentFile = dirStruct(iiFiles).name;
    load([FilesFolder '\' currentFile], 'Confusion_Matrix1');
    saveas(Confusion_Matrix1, [currentFile '.jpg']);
        
    % Update progress bar
    ioi_text_waitbar(iiFiles/iFiles, sprintf('Reading ResNet101 500 times file %d from %d', iiFiles, iFiles));
        
end

ioi_text_waitbar('Clear');

clc; clear; close all

%% ResNet101 Balanced Class

FilesFolder = ('E:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\Termogramas\Tumors\ResNet 101\Balanced Class');
cd(FilesFolder);

dirStruct = dir(fullfile(FilesFolder, '*.mat'));
iFiles = numel(dirStruct);

ioi_text_waitbar(0, 'Please wait...')


% Starting the loop

for iiFiles = 1:iFiles
    currentFile = dirStruct(iiFiles).name;
    load([FilesFolder '\' currentFile], 'Confusion_Matrix1');
    saveas(Confusion_Matrix1, [currentFile '.jpg']);
        
    % Update progress bar
    ioi_text_waitbar(iiFiles/iFiles, sprintf('Reading ResNet101 Balanced Class file %d from %d', iiFiles, iFiles));
        
end

ioi_text_waitbar('Clear');

clc; clear; close all

%% VGG 16  500 Times
FilesFolder = ('E:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\Termogramas\Tumors\VGG 16\500 Times');
cd(FilesFolder);

dirStruct = dir(fullfile(FilesFolder, '*.mat'));
iFiles = numel(dirStruct);

ioi_text_waitbar(0, 'Please wait...')


% Starting the loop

for iiFiles = 1:iFiles
    currentFile = dirStruct(iiFiles).name;
    load([FilesFolder '\' currentFile], 'Confusion_Matrix1');
    saveas(Confusion_Matrix1, [currentFile '.jpg']);
        
    % Update progress bar
    ioi_text_waitbar(iiFiles/iFiles, sprintf('Reading VGG 16 500 times file %d from %d', iiFiles, iFiles));
        
end

ioi_text_waitbar('Clear');

clc; clear; close all

%% VGG 16 Balanced Class

FilesFolder = ('E:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\Termogramas\Tumors\VGG 16\Balanced Class');
cd(FilesFolder);

dirStruct = dir(fullfile(FilesFolder, '*.mat'));
iFiles = numel(dirStruct);

ioi_text_waitbar(0, 'Please wait...')


% Starting the loop

for iiFiles = 1:iFiles
    currentFile = dirStruct(iiFiles).name;
    load([FilesFolder '\' currentFile], 'Confusion_Matrix1');
    saveas(Confusion_Matrix1, [currentFile '.jpg']);
        
    % Update progress bar
    ioi_text_waitbar(iiFiles/iFiles, sprintf('Reading VGG 16 Balanced Class file %d from %d', iiFiles, iFiles));
        
end

ioi_text_waitbar('Clear');

clc; clear; close all

%% VGG 19  500 Times
FilesFolder = ('E:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\Termogramas\Tumors\VGG 19\500 Times');
cd(FilesFolder);

dirStruct = dir(fullfile(FilesFolder, '*.mat'));
iFiles = numel(dirStruct);

ioi_text_waitbar(0, 'Please wait...')


% Starting the loop

for iiFiles = 1:iFiles
    currentFile = dirStruct(iiFiles).name;
    load([FilesFolder '\' currentFile], 'Confusion_Matrix1');
    saveas(Confusion_Matrix1, [currentFile '.jpg']);
        
    % Update progress bar
    ioi_text_waitbar(iiFiles/iFiles, sprintf('Reading VGG 19 500 times file %d from %d', iiFiles, iFiles));
        
end

ioi_text_waitbar('Clear');

clc; clear; close all

%% VGG 19 Balanced Class

FilesFolder = ('E:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\Termogramas\Tumors\VGG 19\Balanced Class');
cd(FilesFolder);

dirStruct = dir(fullfile(FilesFolder, '*.mat'));
iFiles = numel(dirStruct);

ioi_text_waitbar(0, 'Please wait...')


% Starting the loop

for iiFiles = 1:iFiles
    currentFile = dirStruct(iiFiles).name;
    load([FilesFolder '\' currentFile], 'Confusion_Matrix1');
    saveas(Confusion_Matrix1, [currentFile '.jpg']);
        
    % Update progress bar
    ioi_text_waitbar(iiFiles/iFiles, sprintf('Reading VGG 19 Balanced Class file %d from %d', iiFiles, iFiles));
        
end

ioi_text_waitbar('Clear');

clc; clear; close all


