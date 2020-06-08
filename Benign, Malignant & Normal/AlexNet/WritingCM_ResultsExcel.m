%% Write all Confusion Matrix Results in Excel

clear; clc; 

% Folders

codeFolder = 'K:\Archivos Juan Carlos Torres\MATLAB\MATLAB Codigos\Deep Learning\Data Augmented\Res Net 101';
ExcelFile = 'K:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\Practice.xlsx';
ResNet101Files = 'K:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\ResNet 101\200 Brasil + IJC Results\Primeras Aproximaciones';
addpath('K:\Archivos Juan Carlos Torres\MATLAB\MATLAB Codigos');

% Files
dirStruct = dir(fullfile(ResNet101Files , '*.mat'));
RN101Files = numel(dirStruct);

ioi_text_waitbar(0, 'Please wait...')

% Starting the loop

for iiFiles = 1:RN101Files
    load(['K:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\ResNet 101\200 Brasil + IJC Results\Primeras Aproximaciones\Calis_' num2str(iiFiles) '.mat']);
    close all
         
    ExcelData = {'200', 'Brazil + IJC', 'ResNet-101', cp.Sensitivity, cp.Specificity,...
        
        AUC(1,1), AUC(1,2), AUC(1,3),...
        cp.PPV, cp.NPV, cp.Fmeasure, cp.BA, cp.Gmean, cp.kappa};
    
    % Specify the place where it going to be write the data
    sheet = 1;
    my_cell = sprintf( 'B%s', num2str(iiFiles+2) );
    xlswrite(ExcelFile, ExcelData, sheet, my_cell);
    
    % Update progress bar
    ioi_text_waitbar(iiFiles/RN101Files, sprintf('Reading file %d from %d', iiFiles, RN101Files));
    
    cd(codeFolder)
end

ioi_text_waitbar('Clear');