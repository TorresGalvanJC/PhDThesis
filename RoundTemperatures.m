%% Round Values of Temperature Matrix

% The values of the matrix temperature of the .txt files are rounded for
% analyze if this has an impact in the neuronal net
clc; clear
rawDataFolder = 'C:\Users\ADMIN\Documents\Universidad\Doctorado\Termogramas\Termograma Redondeado';
folderName = 'C:\Users\ADMIN\Documents\Universidad\Doctorado\Termogramas\Termograma Redondeado\Redondeado 040';
A = .040;
dirStruct = dir(fullfile(rawDataFolder, '*.txt'));
nFiles = numel(dirStruct);

for iFiles = 1:nFiles
    
    currentFile = dirStruct(iFiles).name;
    idxExt = regexp(dirStruct(iFiles).name, '.txt') - 1;
    %PatientID(iFiles) = str2double((dirStruct(iFiles).name(2:4)));
    %PatientClass(iFiles) = str2double((dirStruct(iFiles).name(6:7)));
    % imageName = str2double((dirStruct(iFiles).name(1:7)));
    imageName = currentFile(1:7);
    
    tempMat = read_txt_thermogram(fullfile(rawDataFolder, currentFile));
    hFig = figure;
    imagesc(tempMat, [30 max(tempMat(:))])
    axis image
    colorbar
    colormap('cool')
    set(hFig, 'color', 'w')
    % Specify window units
    set(hFig, 'units', 'inches')
    % Change figure and paper size
    set(hFig, 'Position', [0.1 0.1 3 3])
    set(hFig, 'PaperPosition', [0.1 0.1 3 3])
    %% Round temperature matrix
    tempMatRound = round(tempMat/A)*A;
    
    %% Write Image of Matrix
    % Read
    testImg = squeeze(tempMatRound(:,:));
    % Normalize between 0 and 1
    testImg = testImg-min(testImg(:));
    testImg = testImg/max(testImg(:));
    
    % Expand to RGB
    testImg = repmat(testImg,[1 1 3]);
    ImageName = [imageName,'.tif'];
            
    % Write TIF
    imwrite(testImg, fullfile(folderName,ImageName));
end
close all

%%
clc; clear
rawDataFolder = 'C:\Users\ADMIN\Documents\Universidad\Doctorado\Termogramas\Termograma Redondeado';
folderName = 'C:\Users\ADMIN\Documents\Universidad\Doctorado\Termogramas\Termograma Redondeado\Redondeado 045';
A = .045;
dirStruct = dir(fullfile(rawDataFolder, '*.txt'));
nFiles = numel(dirStruct);

for iFiles = 1:nFiles
    
    currentFile = dirStruct(iFiles).name;
    idxExt = regexp(dirStruct(iFiles).name, '.txt') - 1;
    %PatientID(iFiles) = str2double((dirStruct(iFiles).name(2:4)));
    %PatientClass(iFiles) = str2double((dirStruct(iFiles).name(6:7)));
    % imageName = str2double((dirStruct(iFiles).name(1:7)));
    imageName = currentFile(1:7);
    
    tempMat = read_txt_thermogram(fullfile(rawDataFolder, currentFile));
    hFig = figure;
    imagesc(tempMat, [30 max(tempMat(:))])
    axis image
    colorbar
    colormap('cool')
    set(hFig, 'color', 'w')
    % Specify window units
    set(hFig, 'units', 'inches')
    % Change figure and paper size
    set(hFig, 'Position', [0.1 0.1 3 3])
    set(hFig, 'PaperPosition', [0.1 0.1 3 3])
    %% Round temperature matrix
    tempMatRound = round(tempMat/A)*A;
    
    %% Write Image of Matrix
    % Read
    testImg = squeeze(tempMatRound(:,:));
    % Normalize between 0 and 1
    testImg = testImg-min(testImg(:));
    testImg = testImg/max(testImg(:));
    
    % Expand to RGB
    testImg = repmat(testImg,[1 1 3]);
    ImageName = [imageName,'.tif'];
            
    % Write TIF
    imwrite(testImg, fullfile(folderName,ImageName));
end
close all

%%
clc; clear
rawDataFolder = 'C:\Users\ADMIN\Documents\Universidad\Doctorado\Termogramas\Termograma Redondeado';
folderName = 'C:\Users\ADMIN\Documents\Universidad\Doctorado\Termogramas\Termograma Redondeado\Redondeado 050';
A = .050;
dirStruct = dir(fullfile(rawDataFolder, '*.txt'));
nFiles = numel(dirStruct);

for iFiles = 1:nFiles
    
    currentFile = dirStruct(iFiles).name;
    idxExt = regexp(dirStruct(iFiles).name, '.txt') - 1;
    %PatientID(iFiles) = str2double((dirStruct(iFiles).name(2:4)));
    %PatientClass(iFiles) = str2double((dirStruct(iFiles).name(6:7)));
    % imageName = str2double((dirStruct(iFiles).name(1:7)));
    imageName = currentFile(1:7);
    
    tempMat = read_txt_thermogram(fullfile(rawDataFolder, currentFile));
    hFig = figure;
    imagesc(tempMat, [30 max(tempMat(:))])
    axis image
    colorbar
    colormap('cool')
    set(hFig, 'color', 'w')
    % Specify window units
    set(hFig, 'units', 'inches')
    % Change figure and paper size
    set(hFig, 'Position', [0.1 0.1 3 3])
    set(hFig, 'PaperPosition', [0.1 0.1 3 3])
    %% Round temperature matrix
    tempMatRound = round(tempMat/A)*A;
    
    %% Write Image of Matrix
    % Read
    testImg = squeeze(tempMatRound(:,:));
    % Normalize between 0 and 1
    testImg = testImg-min(testImg(:));
    testImg = testImg/max(testImg(:));
    
    % Expand to RGB
    testImg = repmat(testImg,[1 1 3]);
    ImageName = [imageName,'.tif'];
            
    % Write TIF
    imwrite(testImg, fullfile(folderName,ImageName));
end
close all

%%
clc; clear
rawDataFolder = 'C:\Users\ADMIN\Documents\Universidad\Doctorado\Termogramas\Termograma Redondeado';
folderName = 'C:\Users\ADMIN\Documents\Universidad\Doctorado\Termogramas\Termograma Redondeado\Redondeado 080';
A = .080;
dirStruct = dir(fullfile(rawDataFolder, '*.txt'));
nFiles = numel(dirStruct);

for iFiles = 1:nFiles
    
    currentFile = dirStruct(iFiles).name;
    idxExt = regexp(dirStruct(iFiles).name, '.txt') - 1;
    %PatientID(iFiles) = str2double((dirStruct(iFiles).name(2:4)));
    %PatientClass(iFiles) = str2double((dirStruct(iFiles).name(6:7)));
    % imageName = str2double((dirStruct(iFiles).name(1:7)));
    imageName = currentFile(1:7);
    
    tempMat = read_txt_thermogram(fullfile(rawDataFolder, currentFile));
    hFig = figure;
    imagesc(tempMat, [30 max(tempMat(:))])
    axis image
    colorbar
    colormap('cool')
    set(hFig, 'color', 'w')
    % Specify window units
    set(hFig, 'units', 'inches')
    % Change figure and paper size
    set(hFig, 'Position', [0.1 0.1 3 3])
    set(hFig, 'PaperPosition', [0.1 0.1 3 3])
    %% Round temperature matrix
    tempMatRound = round(tempMat/A)*A;
    
    %% Write Image of Matrix
    % Read
    testImg = squeeze(tempMatRound(:,:));
    % Normalize between 0 and 1
    testImg = testImg-min(testImg(:));
    testImg = testImg/max(testImg(:));
    
    % Expand to RGB
    testImg = repmat(testImg,[1 1 3]);
    ImageName = [imageName,'.tif'];
            
    % Write TIF
    imwrite(testImg, fullfile(folderName,ImageName));
end
close all
%%
clc; clear
rawDataFolder = 'C:\Users\ADMIN\Documents\Universidad\Doctorado\Termogramas\Termograma Redondeado';
folderName = 'C:\Users\ADMIN\Documents\Universidad\Doctorado\Termogramas\Termograma Redondeado\Redondeado 100';
A = .100;
dirStruct = dir(fullfile(rawDataFolder, '*.txt'));
nFiles = numel(dirStruct);

for iFiles = 1:nFiles
    
    currentFile = dirStruct(iFiles).name;
    idxExt = regexp(dirStruct(iFiles).name, '.txt') - 1;
    %PatientID(iFiles) = str2double((dirStruct(iFiles).name(2:4)));
    %PatientClass(iFiles) = str2double((dirStruct(iFiles).name(6:7)));
    % imageName = str2double((dirStruct(iFiles).name(1:7)));
    imageName = currentFile(1:7);
    
    tempMat = read_txt_thermogram(fullfile(rawDataFolder, currentFile));
    hFig = figure;
    imagesc(tempMat, [30 max(tempMat(:))])
    axis image
    colorbar
    colormap('cool')
    set(hFig, 'color', 'w')
    % Specify window units
    set(hFig, 'units', 'inches')
    % Change figure and paper size
    set(hFig, 'Position', [0.1 0.1 3 3])
    set(hFig, 'PaperPosition', [0.1 0.1 3 3])
    %% Round temperature matrix
    tempMatRound = round(tempMat/A)*A;
    
    %% Write Image of Matrix
    % Read
    testImg = squeeze(tempMatRound(:,:));
    % Normalize between 0 and 1
    testImg = testImg-min(testImg(:));
    testImg = testImg/max(testImg(:));
    
    % Expand to RGB
    testImg = repmat(testImg,[1 1 3]);
    ImageName = [imageName,'.tif'];
            
    % Write TIF
    imwrite(testImg, fullfile(folderName,ImageName));
end
close all



% clc; clear
% filename = 'C:\Users\ADMIN\Documents\Universidad\Doctorado\Termogramas\Termograma Redondeado\P286_En_Fr.txt';
% folderName = 'C:\Users\ADMIN\Documents\Universidad\Doctorado\Termogramas\Termograma Redondeado\Redondeado 005';
% tempMat = read_txt_thermogram(filename);
% hFig = figure;
% imagesc(tempMat, [30 max(tempMat(:))])
% axis image
% colorbar
% colormap('cool')
% % imwrite(tempMat, colormap('jet'), 'P005_Sa_Fr.png', 'png')
% set(hFig, 'color', 'w')
% % Specify window units
% set(hFig, 'units', 'inches')
% % Change figure and paper size
% set(hFig, 'Position', [0.1 0.1 3 3])
% set(hFig, 'PaperPosition', [0.1 0.1 3 3])
% 
% %% Answer to round a matrix
% 
% tempMatRound = round(tempMat/.005)*.005;
% 
% %% Write Image of Matrix
% % Read
% testImg = squeeze(tempMatRound(:,:));
% % Normalize between 0 and 1
% testImg = testImg-min(testImg(:));
% testImg = testImg/max(testImg(:));
% 
% % Expand to RGB
% testImg = repmat(testImg,[1 1 3]);
% ImageName = sprintf('Abnormal_Test_286.tif');
% % Write TIF
% imwrite(testImg, fullfile(folderName,ImageName));