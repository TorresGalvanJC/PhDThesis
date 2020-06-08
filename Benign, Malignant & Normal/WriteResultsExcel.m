%% Write all Confusion Matrix Results in Excel

%% AlexNet
% 
% clear; clc;
% 
% % Folders
% 
% codeFolder = 'K:\Archivos Juan Carlos Torres\MATLAB\MATLAB
% Codigos\Benign, Malignant & Normal'; ExcelFile = 'K:\Archivos Juan Carlos
% Torres\MATLAB\Deep Learning\Tumors_Comparison.xlsx';
% 
% FilesFolder = 'K:\Archivos Juan Carlos Torres\MATLAB\Deep
% Learning\Termogramas\Tumors\AlexNet\500 Times';
% 
% addpath('K:\Archivos Juan Carlos Torres\MATLAB\MATLAB Codigos');
% 
% % Files dirStruct = dir(fullfile(FilesFolder, '*.mat')); iFiles =
% numel(dirStruct);
% 
% ioi_text_waitbar(0, 'Please wait...')
% 
% % Starting the loop
% 
% for iiFiles = 1:iFiles
%     load([FilesFolder '\Calis_' num2str(iiFiles) '.mat'], 'cp'); close
%     all
%          
%     ExcelData = {'500 times', 'AlexNet',...
%         cp.Sensitivity, cp.Specificity,... cp.Precision,
%         cp.FalseDiscoveryRate, cp.FalsePositiveRate,...
%         cp.FalseNegativeRate, cp.NegativePredictiveValue, cp.Prevalence,
%         ... cp.NegativeLikelihoodRatio, cp.Accuracy, cp.Error,
%         cp.F1_score,... cp.G, cp.MatthewsCorrelationCoefficient,
%         cp.Kappa};
%     
%     % Specify the place where it going to be write the data sheet = 1;
%     my_cell = sprintf( 'B%s', num2str(iiFiles+2) ); xlswrite(ExcelFile,
%     ExcelData, sheet, my_cell);
%     
%     % Update progress bar
%     
%     ioi_text_waitbar(iiFiles/iFiles, sprintf('Reading Alex Net 500 times
%     file %d from %d', iiFiles, iFiles));
%     
%     cd(codeFolder)
% end
% 
% ioi_text_waitbar('Clear');
% 
% 
% clear; clc;
% 
% % Folders
% 
% codeFolder = 'K:\Archivos Juan Carlos Torres\MATLAB\MATLAB
% Codigos\Benign, Malignant & Normal'; ExcelFile = 'K:\Archivos Juan Carlos
% Torres\MATLAB\Deep Learning\Tumors_Comparison.xlsx';
% 
% FilesFolder = 'K:\Archivos Juan Carlos Torres\MATLAB\Deep
% Learning\Termogramas\Tumors\AlexNet\Balanced Class';
% 
% addpath('K:\Archivos Juan Carlos Torres\MATLAB\MATLAB Codigos');
% 
% % Files dirStruct = dir(fullfile(FilesFolder, '*.mat')); iFiles =
% numel(dirStruct);
% 
% ioi_text_waitbar(0, 'Please wait...')
% 
% % Starting the loop
% 
% for iiFiles = 1:iFiles
%     load([FilesFolder '\Calis_' num2str(iiFiles) '.mat'], 'cp'); close
%     all
%          
%     ExcelData = {'BalancedClass','AlexNet',...
%         cp.Sensitivity, cp.Specificity,... cp.Precision,
%         cp.FalseDiscoveryRate, cp.FalsePositiveRate,...
%         cp.FalseNegativeRate, cp.NegativePredictiveValue, cp.Prevalence,
%         ... cp.NegativeLikelihoodRatio, cp.Accuracy, cp.Error,
%         cp.F1_score,... cp.G, cp.MatthewsCorrelationCoefficient,
%         cp.Kappa};
%     
%     % Specify the place where it going to be write the data sheet = 2;
%     my_cell = sprintf( 'B%s', num2str(iiFiles+2) ); xlswrite(ExcelFile,
%     ExcelData, sheet, my_cell);
%     
%     % Update progress bar ioi_text_waitbar(iiFiles/iFiles,
%     sprintf('Reading Alex Net Balanced Class file %d from %d', iiFiles,
%     iFiles));
%     
%     cd(codeFolder)
% end
% 
% ioi_text_waitbar('Clear');
% 
% %% GoogleNet
% 
% clear; clc;
% 
% % Folders
% 
% codeFolder = 'K:\Archivos Juan Carlos Torres\MATLAB\MATLAB
% Codigos\Benign, Malignant & Normal'; ExcelFile = 'K:\Archivos Juan Carlos
% Torres\MATLAB\Deep Learning\Tumors_Comparison.xlsx';
% 
% FilesFolder = 'K:\Archivos Juan Carlos Torres\MATLAB\Deep
% Learning\Termogramas\Tumors\GoogleNet\500 Times';
% 
% addpath('K:\Archivos Juan Carlos Torres\MATLAB\MATLAB Codigos');
% 
% % Files dirStruct = dir(fullfile(FilesFolder, '*.mat')); iFiles =
% numel(dirStruct);
% 
% ioi_text_waitbar(0, 'Please wait...')
% 
% % Starting the loop
% 
% for iiFiles = 1:iFiles
%     
%     load([FilesFolder '\Calis_' num2str(iiFiles) '.mat'], 'cp'); close
%     all
%          
%     ExcelData = {'500 times', 'GoogleNetNet',...
%         cp.Sensitivity, cp.Specificity,... cp.Precision,
%         cp.FalseDiscoveryRate, cp.FalsePositiveRate,...
%         cp.FalseNegativeRate, cp.NegativePredictiveValue, cp.Prevalence,
%         ... cp.NegativeLikelihoodRatio, cp.Accuracy, cp.Error,
%         cp.F1_score,... cp.G, cp.MatthewsCorrelationCoefficient,
%         cp.Kappa};
%     
%     % Specify the place where it going to be write the data sheet = 1;
%     A=2+(32*1); my_cell = sprintf( 'B%s', num2str(iiFiles+A) );
%     xlswrite(ExcelFile, ExcelData, sheet, my_cell);
%     
%     % Update progress bar ioi_text_waitbar(iiFiles/iFiles,
%     sprintf('Reading GoogleNet 500 times file %d from %d', iiFiles,
%     iFiles));
%     
%     cd(codeFolder)
% end
% 
% ioi_text_waitbar('Clear');
% 
% 
% clear; clc;
% 
% % Folders
% 
% codeFolder = 'K:\Archivos Juan Carlos Torres\MATLAB\MATLAB
% Codigos\Benign, Malignant & Normal'; ExcelFile = 'K:\Archivos Juan Carlos
% Torres\MATLAB\Deep Learning\Tumors_Comparison.xlsx';
% 
% FilesFolder = 'K:\Archivos Juan Carlos Torres\MATLAB\Deep
% Learning\Termogramas\Tumors\GoogleNet\Balanced Class';
% 
% addpath('K:\Archivos Juan Carlos Torres\MATLAB\MATLAB Codigos');
% 
% % Files dirStruct = dir(fullfile(FilesFolder, '*.mat')); iFiles =
% numel(dirStruct);
% 
% ioi_text_waitbar(0, 'Please wait...')
% 
% % Starting the loop
% 
% for iiFiles = 1:iFiles
%     load([FilesFolder '\Calis_' num2str(iiFiles) '.mat'], 'cp'); close
%     all
%          
%     ExcelData = {'BalancedClass','GoogleNet',...
%         cp.Sensitivity, cp.Specificity,... cp.Precision,
%         cp.FalseDiscoveryRate, cp.FalsePositiveRate,...
%         cp.FalseNegativeRate, cp.NegativePredictiveValue, cp.Prevalence,
%         ... cp.NegativeLikelihoodRatio, cp.Accuracy, cp.Error,
%         cp.F1_score,... cp.G, cp.MatthewsCorrelationCoefficient,
%         cp.Kappa};
%     
%     % Specify the place where it going to be write the data sheet = 2; A
%     = 2+(30*1) my_cell = sprintf( 'B%s', num2str(iiFiles+A) );
%     xlswrite(ExcelFile, ExcelData, sheet, my_cell);
%     
%     % Update progress bar
%     
%     ioi_text_waitbar(iiFiles/iFiles, sprintf('Reading GoogleNet Balanced
%     Class file %d from %d', iiFiles, iFiles));
%     
%     cd(codeFolder)
% end
% 
% ioi_text_waitbar('Clear');
%% Inception V3
clear; clc; 

% Folders

codeFolder = 'K:\Archivos Juan Carlos Torres\MATLAB\MATLAB Codigos\Benign, Malignant & Normal';
ExcelFile = 'K:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\Tumors_Comparison.xlsx';

FilesFolder = 'K:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\Termogramas\Tumors\Inception V3\500 Times';

addpath('K:\Archivos Juan Carlos Torres\MATLAB\MATLAB Codigos');

% Files
dirStruct = dir(fullfile(FilesFolder, '*.mat'));
iFiles = numel(dirStruct);

ioi_text_waitbar(0, 'Please wait...')

% Starting the loop

for iiFiles = 1:iFiles
    load([FilesFolder '\Calis_' num2str(iiFiles) '.mat'], 'cp');
    close all
         
    ExcelData = {'500 times', 'Inception V3',...
        cp.Sensitivity, cp.Specificity,...
        cp.Precision, cp.FalseDiscoveryRate, cp.FalsePositiveRate,...
        cp.FalseNegativeRate, cp.NegativePredictiveValue, cp.Prevalence, ...
        cp.NegativeLikelihoodRatio, cp.Accuracy, cp.Error, cp.F1_score,...
        cp.G, cp.MatthewsCorrelationCoefficient, cp.Kappa};
    
    % Specify the place where it going to be write the data
    sheet = 1;
    A=2+(32*2);
    my_cell = sprintf( 'B%s', num2str(iiFiles+A) );
    xlswrite(ExcelFile, ExcelData, sheet, my_cell);
    
    % Update progress bar
    ioi_text_waitbar(iiFiles/iFiles, sprintf('Reading Inception V3 500 times file %d from %d', iiFiles, iFiles));
    
    cd(codeFolder)
end

ioi_text_waitbar('Clear');


clear; clc; 

% Folders

codeFolder = 'K:\Archivos Juan Carlos Torres\MATLAB\MATLAB Codigos\Benign, Malignant & Normal';
ExcelFile = 'K:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\Tumors_Comparison.xlsx';

FilesFolder = 'K:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\Termogramas\Tumors\Inception V3\Balanced Class';

addpath('K:\Archivos Juan Carlos Torres\MATLAB\MATLAB Codigos');

% Files
dirStruct = dir(fullfile(FilesFolder, '*.mat'));
iFiles = numel(dirStruct);

ioi_text_waitbar(0, 'Please wait...')

% Starting the loop

for iiFiles = 1:iFiles
    load([FilesFolder '\Calis_' num2str(iiFiles) '.mat'], 'cp');
    close all
         
    ExcelData = {'BalancedClass','Inception V3',...
        cp.Sensitivity, cp.Specificity,...
        cp.Precision, cp.FalseDiscoveryRate, cp.FalsePositiveRate,...
        cp.FalseNegativeRate, cp.NegativePredictiveValue, cp.Prevalence, ...
        cp.NegativeLikelihoodRatio, cp.Accuracy, cp.Error, cp.F1_score,...
        cp.G, cp.MatthewsCorrelationCoefficient, cp.Kappa};
    
    % Specify the place where it going to be write the data
    sheet = 2;
    A = 2+(30*2)
    my_cell = sprintf( 'B%s', num2str(iiFiles+A) );
    xlswrite(ExcelFile, ExcelData, sheet, my_cell);
    
    % Update progress bar
    ioi_text_waitbar(iiFiles/iFiles, sprintf('Reading Inception V3 Balanced Class file %d from %d', iiFiles, iFiles));
    
    cd(codeFolder)
end

ioi_text_waitbar('Clear');
%% ResNet 50

clear; clc; 

% Folders

codeFolder = 'K:\Archivos Juan Carlos Torres\MATLAB\MATLAB Codigos\Benign, Malignant & Normal';
ExcelFile = 'K:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\Tumors_Comparison.xlsx';

FilesFolder = 'K:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\Termogramas\Tumors\ResNet 50\500 Times';

addpath('K:\Archivos Juan Carlos Torres\MATLAB\MATLAB Codigos');

% Files
dirStruct = dir(fullfile(FilesFolder, '*.mat'));
iFiles = numel(dirStruct);

ioi_text_waitbar(0, 'Please wait...')

% Starting the loop

for iiFiles = 1:iFiles
    load([FilesFolder '\Calis_' num2str(iiFiles) '.mat'], 'cp');
    close all
         
    ExcelData = {'500 times', 'ResNet 50',...
        cp.Sensitivity, cp.Specificity,...
        cp.Precision, cp.FalseDiscoveryRate, cp.FalsePositiveRate,...
        cp.FalseNegativeRate, cp.NegativePredictiveValue, cp.Prevalence, ...
        cp.NegativeLikelihoodRatio, cp.Accuracy, cp.Error, cp.F1_score,...
        cp.G, cp.MatthewsCorrelationCoefficient, cp.Kappa};
    
    % Specify the place where it going to be write the data
    sheet = 1;
    A=2+(32*3);
    my_cell = sprintf( 'B%s', num2str(iiFiles+A) );
    xlswrite(ExcelFile, ExcelData, sheet, my_cell);
    
    % Update progress bar
    ioi_text_waitbar(iiFiles/iFiles, sprintf('Reading ResNet50 500 times file %d from %d', iiFiles, iFiles));
    
    cd(codeFolder)
end

ioi_text_waitbar('Clear');


clear; clc; 

% Folders

codeFolder = 'K:\Archivos Juan Carlos Torres\MATLAB\MATLAB Codigos\Benign, Malignant & Normal';
ExcelFile = 'K:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\Tumors_Comparison.xlsx';

FilesFolder = 'K:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\Termogramas\Tumors\ResNet 50\Balanced Class';

addpath('K:\Archivos Juan Carlos Torres\MATLAB\MATLAB Codigos');

% Files
dirStruct = dir(fullfile(FilesFolder, '*.mat'));
iFiles = numel(dirStruct);

ioi_text_waitbar(0, 'Please wait...')

% Starting the loop

for iiFiles = 1:iFiles
    load([FilesFolder '\Calis_' num2str(iiFiles) '.mat'], 'cp');
    close all
         
    ExcelData = {'BalancedClass','ResNet 50',...
        cp.Sensitivity, cp.Specificity,...
        cp.Precision, cp.FalseDiscoveryRate, cp.FalsePositiveRate,...
        cp.FalseNegativeRate, cp.NegativePredictiveValue, cp.Prevalence, ...
        cp.NegativeLikelihoodRatio, cp.Accuracy, cp.Error, cp.F1_score,...
        cp.G, cp.MatthewsCorrelationCoefficient, cp.Kappa};
    
    % Specify the place where it going to be write the data
    sheet = 2;
    A = 2+(30*3)
    my_cell = sprintf( 'B%s', num2str(iiFiles+A) );
    xlswrite(ExcelFile, ExcelData, sheet, my_cell);
    
    % Update progress bar
    ioi_text_waitbar(iiFiles/iFiles, sprintf('Reading ResNet50 Balanced Class file %d from %d', iiFiles, iFiles));
    
    cd(codeFolder)
end

ioi_text_waitbar('Clear');
%% ResNet 101

clear; clc; 

% Folders

codeFolder = 'K:\Archivos Juan Carlos Torres\MATLAB\MATLAB Codigos\Benign, Malignant & Normal';
ExcelFile = 'K:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\Tumors_Comparison.xlsx';

FilesFolder = 'K:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\Termogramas\Tumors\ResNet 101\500 Times';

addpath('K:\Archivos Juan Carlos Torres\MATLAB\MATLAB Codigos');

% Files
dirStruct = dir(fullfile(FilesFolder, '*.mat'));
iFiles = numel(dirStruct);

ioi_text_waitbar(0, 'Please wait...')

% Starting the loop

for iiFiles = 1:iFiles
    load([FilesFolder '\Calis_' num2str(iiFiles) '.mat'], 'cp');
    close all
         
    ExcelData = {'500 times', 'ResNet 101',...
        cp.Sensitivity, cp.Specificity,...
        cp.Precision, cp.FalseDiscoveryRate, cp.FalsePositiveRate,...
        cp.FalseNegativeRate, cp.NegativePredictiveValue, cp.Prevalence, ...
        cp.NegativeLikelihoodRatio, cp.Accuracy, cp.Error, cp.F1_score,...
        cp.G, cp.MatthewsCorrelationCoefficient, cp.Kappa};
    
    % Specify the place where it going to be write the data
    sheet = 1;
    A=2+(32*4);
    my_cell = sprintf( 'B%s', num2str(iiFiles+A) );
    xlswrite(ExcelFile, ExcelData, sheet, my_cell);
    
    % Update progress bar
    ioi_text_waitbar(iiFiles/iFiles, sprintf('Reading ResNet101 500 times file %d from %d', iiFiles, iFiles));
    
    cd(codeFolder)
end

ioi_text_waitbar('Clear');


clear; clc; 

% Folders

codeFolder = 'K:\Archivos Juan Carlos Torres\MATLAB\MATLAB Codigos\Benign, Malignant & Normal';
ExcelFile = 'K:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\Tumors_Comparison.xlsx';

FilesFolder = 'K:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\Termogramas\Tumors\ResNet 101\Balanced Class';

addpath('K:\Archivos Juan Carlos Torres\MATLAB\MATLAB Codigos');

% Files
dirStruct = dir(fullfile(FilesFolder, '*.mat'));
iFiles = numel(dirStruct);

ioi_text_waitbar(0, 'Please wait...')

% Starting the loop

for iiFiles = 1:iFiles
    load([FilesFolder '\Calis_' num2str(iiFiles) '.mat'], 'cp');
    close all
         
    ExcelData = {'BalancedClass','ResNet 101',...
        cp.Sensitivity, cp.Specificity,...
        cp.Precision, cp.FalseDiscoveryRate, cp.FalsePositiveRate,...
        cp.FalseNegativeRate, cp.NegativePredictiveValue, cp.Prevalence, ...
        cp.NegativeLikelihoodRatio, cp.Accuracy, cp.Error, cp.F1_score,...
        cp.G, cp.MatthewsCorrelationCoefficient, cp.Kappa};
    
    % Specify the place where it going to be write the data
    sheet = 2;
    A = 2+(30*4)
    my_cell = sprintf( 'B%s', num2str(iiFiles+A) );
    xlswrite(ExcelFile, ExcelData, sheet, my_cell);
    
    % Update progress bar
    ioi_text_waitbar(iiFiles/iFiles, sprintf('Reading ResNet101 Balanced Class file %d from %d', iiFiles, iFiles));
    
    cd(codeFolder)
end

ioi_text_waitbar('Clear');
% %% VGG 16
% 
% clear; clc; 
% 
% % Folders
% 
% codeFolder = 'K:\Archivos Juan Carlos Torres\MATLAB\MATLAB Codigos\Benign, Malignant & Normal';
% ExcelFile = 'K:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\Tumors_Comparison.xlsx';
% 
% FilesFolder = 'K:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\Termogramas\Tumors\VGG 16\500 Times';
% 
% addpath('K:\Archivos Juan Carlos Torres\MATLAB\MATLAB Codigos');
% 
% % Files
% dirStruct = dir(fullfile(FilesFolder, '*.mat'));
% iFiles = numel(dirStruct);
% 
% ioi_text_waitbar(0, 'Please wait...')
% 
% % Starting the loop
% 
% for iiFiles = 1:iFiles
%     load([FilesFolder '\Calis_' num2str(iiFiles) '.mat'], 'cp');
%     close all
%          
%     ExcelData = {'500 times', 'VGG 16',...
%         cp.Sensitivity, cp.Specificity,...
%         cp.Precision, cp.FalseDiscoveryRate, cp.FalsePositiveRate,...
%         cp.FalseNegativeRate, cp.NegativePredictiveValue, cp.Prevalence, ...
%         cp.NegativeLikelihoodRatio, cp.Accuracy, cp.Error, cp.F1_score,...
%         cp.G, cp.MatthewsCorrelationCoefficient, cp.Kappa};
%     
%     % Specify the place where it going to be write the data
%     sheet = 1;
%     A=2+(32*5);
%     my_cell = sprintf( 'B%s', num2str(iiFiles+A) );
%     xlswrite(ExcelFile, ExcelData, sheet, my_cell);
%     
%     % Update progress bar
%     ioi_text_waitbar(iiFiles/iFiles, sprintf('Reading VGG 16, 500 times file %d from %d', iiFiles, iFiles));
%     
%     cd(codeFolder)
% end
% 
% ioi_text_waitbar('Clear');
% 
% 
% clear; clc; 
% 
% % Folders
% 
% codeFolder = 'K:\Archivos Juan Carlos Torres\MATLAB\MATLAB Codigos\Benign, Malignant & Normal';
% ExcelFile = 'K:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\Tumors_Comparison.xlsx';
% 
% FilesFolder = 'K:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\Termogramas\Tumors\VGG 16\Balanced Class';
% 
% addpath('K:\Archivos Juan Carlos Torres\MATLAB\MATLAB Codigos');
% 
% % Files
% dirStruct = dir(fullfile(FilesFolder, '*.mat'));
% iFiles = numel(dirStruct);
% 
% ioi_text_waitbar(0, 'Please wait...')
% 
% % Starting the loop
% 
% for iiFiles = 1:iFiles
%     load([FilesFolder '\Calis_' num2str(iiFiles) '.mat'], 'cp');
%     close all
%          
%     ExcelData = {'BalancedClass','VGG 16',...
%         cp.Sensitivity, cp.Specificity,...
%         cp.Precision, cp.FalseDiscoveryRate, cp.FalsePositiveRate,...
%         cp.FalseNegativeRate, cp.NegativePredictiveValue, cp.Prevalence, ...
%         cp.NegativeLikelihoodRatio, cp.Accuracy, cp.Error, cp.F1_score,...
%         cp.G, cp.MatthewsCorrelationCoefficient, cp.Kappa};
%     
%     % Specify the place where it going to be write the data
%     sheet = 2;
%     A = 2+(30*5)
%     my_cell = sprintf( 'B%s', num2str(iiFiles+A) );
%     xlswrite(ExcelFile, ExcelData, sheet, my_cell);
%     
%     % Update progress bar
%     ioi_text_waitbar(iiFiles/iFiles, sprintf('Reading VGG 16, Balanced Class file %d from %d', iiFiles, iFiles));
%     
%     cd(codeFolder)
% end
% 
% ioi_text_waitbar('Clear');
%% VGG 19
% 
% clear; clc; 
% 
% % Folders
% 
% codeFolder = 'K:\Archivos Juan Carlos Torres\MATLAB\MATLAB Codigos\Benign, Malignant & Normal';
% ExcelFile = 'K:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\Tumors_Comparison.xlsx';
% 
% FilesFolder = 'K:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\Termogramas\Tumors\VGG 19\500 Times';
% 
% addpath('K:\Archivos Juan Carlos Torres\MATLAB\MATLAB Codigos');
% 
% % Files
% dirStruct = dir(fullfile(FilesFolder, '*.mat'));
% iFiles = numel(dirStruct);
% 
% ioi_text_waitbar(0, 'Please wait...')
% 
% % Starting the loop
% 
% for iiFiles = 1:iFiles
%     load([FilesFolder '\Calis_' num2str(iiFiles) '.mat'], 'cp');
%     close all
%          
%     ExcelData = {'500 times', 'VGG 19',...
%         cp.Sensitivity, cp.Specificity,...
%         cp.Precision, cp.FalseDiscoveryRate, cp.FalsePositiveRate,...
%         cp.FalseNegativeRate, cp.NegativePredictiveValue, cp.Prevalence, ...
%         cp.NegativeLikelihoodRatio, cp.Accuracy, cp.Error, cp.F1_score,...
%         cp.G, cp.MatthewsCorrelationCoefficient, cp.Kappa};
%     
%     % Specify the place where it going to be write the data
%     sheet = 1;
%     A=2+(32*6);
%     my_cell = sprintf( 'B%s', num2str(iiFiles+A) );
%     xlswrite(ExcelFile, ExcelData, sheet, my_cell);
%     
%     % Update progress bar
%     ioi_text_waitbar(iiFiles/iFiles, sprintf('Reading VGG 19, 500 times file %d from %d', iiFiles, iFiles));
%     
%     cd(codeFolder)
% end
% 
% ioi_text_waitbar('Clear');
% 

clear; clc; 

% Folders

codeFolder = 'K:\Archivos Juan Carlos Torres\MATLAB\MATLAB Codigos\Benign, Malignant & Normal';
ExcelFile = 'K:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\Tumors_Comparison.xlsx';

FilesFolder = 'K:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\Termogramas\Tumors\VGG 19\Balanced Class';

addpath('K:\Archivos Juan Carlos Torres\MATLAB\MATLAB Codigos');

% Files
dirStruct = dir(fullfile(FilesFolder, '*.mat'));
iFiles = numel(dirStruct);

ioi_text_waitbar(0, 'Please wait...')

% Starting the loop

for iiFiles = 1:iFiles
    load([FilesFolder '\Calis_' num2str(iiFiles) '.mat'], 'cp');
    close all
         
    ExcelData = {'BalancedClass','VGG 19',...
        cp.Sensitivity, cp.Specificity,...
        cp.Precision, cp.FalseDiscoveryRate, cp.FalsePositiveRate,...
        cp.FalseNegativeRate, cp.NegativePredictiveValue, cp.Prevalence, ...
        cp.NegativeLikelihoodRatio, cp.Accuracy, cp.Error, cp.F1_score,...
        cp.G, cp.MatthewsCorrelationCoefficient, cp.Kappa};
    
    % Specify the place where it going to be write the data
    sheet = 2;
    A = 2+(30*6)
    my_cell = sprintf( 'B%s', num2str(iiFiles+A) );
    xlswrite(ExcelFile, ExcelData, sheet, my_cell);
    
    % Update progress bar
    ioi_text_waitbar(iiFiles/iFiles, sprintf('Reading VGG 19, Balanced Class file %d from %d', iiFiles, iFiles));
    
    cd(codeFolder)
end

ioi_text_waitbar('Clear');