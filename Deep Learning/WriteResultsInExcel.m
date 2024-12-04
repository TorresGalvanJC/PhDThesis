%% AlexNet Brazil

% 1 AlexNet Brazil Select the MATLAB file
load('E:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\AlexNet\Brasil Frontal\1ra Corrida\DeepLearningFirstResults.mat')

cp = my_classperf(cm);
ExcelData = {cp.Se, cp.Sp, AUC(1,1), AUC(1,2), AUC(1,3), cp.PPV, cp.NPV, cp.Fmeasure, cp.BA, cp.Gmean, cp.kappa}; %
    %

% Write data in Excel file deep_learning_comparison

% Location and name of data
filename = 'C:\Users\ADMIN\Dropbox\Profesionales Compartidas\Doctorado Juan Carlos Torres\deep_learning_comparison.xlsx';

% Specify the place where it going to be write the data
sheet = 2;
xlRange = 'D2:N2';

% Write the data
xlswrite(filename,ExcelData,sheet,xlRange)

clear; clc; close all





% 2 AlexNet Brazil Select the MATLAB file
load('E:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\AlexNet\Brasil Frontal\2da Corrida\DeepLearningFirstResults.mat')

cp = my_classperf(cm);ExcelData = {cp.Se, cp.Sp, AUC(1,1), AUC(1,2), AUC(1,3), cp.PPV, cp.NPV, cp.Fmeasure, cp.BA, cp.Gmean, cp.kappa}; %
    %

% Write data in Excel file deep_learning_comparison

% Location and name of data
filename = 'C:\Users\ADMIN\Dropbox\Profesionales Compartidas\Doctorado Juan Carlos Torres\deep_learning_comparison.xlsx';

% Specify the place where it going to be write the data
sheet = 2;
xlRange = 'D3:N3';

% Write the data
xlswrite(filename,ExcelData,sheet,xlRange)

clear; clc; close all





% 3 AlexNet Brazil Select the MATLAB file
load('E:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\AlexNet\Brasil Frontal\3ra Corrida\DeepLearningFirstResults.mat')

cp = my_classperf(cm);ExcelData = {cp.Se, cp.Sp, AUC(1,1), AUC(1,2), AUC(1,3), cp.PPV, cp.NPV, cp.Fmeasure, cp.BA, cp.Gmean, cp.kappa}; %
    %

% Write data in Excel file deep_learning_comparison

% Location and name of data
filename = 'C:\Users\ADMIN\Dropbox\Profesionales Compartidas\Doctorado Juan Carlos Torres\deep_learning_comparison.xlsx';

% Specify the place where it going to be write the data
sheet = 2;
xlRange = 'D4:N4';

% Write the data
xlswrite(filename,ExcelData,sheet,xlRange)

clear; clc; close all





% 4 AlexNet Brazil Select the MATLAB file
load('E:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\AlexNet\Brasil Frontal\4ta Corrida\DeepLearningFirstResults.mat')

cp = my_classperf(cm);ExcelData = {cp.Se, cp.Sp, AUC(1,1), AUC(1,2), AUC(1,3), cp.PPV, cp.NPV, cp.Fmeasure, cp.BA, cp.Gmean, cp.kappa}; %
    %

% Write data in Excel file deep_learning_comparison

% Location and name of data
filename = 'C:\Users\ADMIN\Dropbox\Profesionales Compartidas\Doctorado Juan Carlos Torres\deep_learning_comparison.xlsx';

% Specify the place where it going to be write the data
sheet = 2;
xlRange = 'D5:N5';

% Write the data
xlswrite(filename,ExcelData,sheet,xlRange)

clear; clc; close all






% 5 AlexNet Brazil Select the MATLAB file
load('E:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\AlexNet\Brasil Frontal\5ta Corrida\DeepLearningFirstResults.mat')

cp = my_classperf(cm);ExcelData = {cp.Se, cp.Sp, AUC(1,1), AUC(1,2), AUC(1,3), cp.PPV, cp.NPV, cp.Fmeasure, cp.BA, cp.Gmean, cp.kappa}; %
    %

% Write data in Excel file deep_learning_comparison

% Location and name of data
filename = 'C:\Users\ADMIN\Dropbox\Profesionales Compartidas\Doctorado Juan Carlos Torres\deep_learning_comparison.xlsx';

% Specify the place where it going to be write the data
sheet = 2;
xlRange = 'D6:N6';

% Write the data
xlswrite(filename,ExcelData,sheet,xlRange)

clear; clc; close all

%% AlexNet Brazil+IJC

% 1 AlexNet Brazil + IJC 
% Select the MATLAB file
load('E:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\AlexNet\Brasil + IJC Frontal\1ra Corrida\DeepLearningFirstResults.mat')

cp = my_classperf(cm);ExcelData = {cp.Se, cp.Sp, AUC(1,1), AUC(1,2), AUC(1,3), cp.PPV, cp.NPV, cp.Fmeasure, cp.BA, cp.Gmean, cp.kappa}; %
    %

% Write data in Excel file deep_learning_comparison

% Location and name of data
filename = 'C:\Users\ADMIN\Dropbox\Profesionales Compartidas\Doctorado Juan Carlos Torres\deep_learning_comparison.xlsx';

% Specify the place where it going to be write the data
sheet = 2;
xlRange = 'D7:N7';

% Write the data
xlswrite(filename,ExcelData,sheet,xlRange)

clear; clc; close all




% 2 AlexNet Brazil + IJC 
% Select the MATLAB file
load('E:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\AlexNet\Brasil + IJC Frontal\2da Corrida\DeepLearningFirstResults.mat')

cp = my_classperf(cm);ExcelData = {cp.Se, cp.Sp, AUC(1,1), AUC(1,2), AUC(1,3), cp.PPV, cp.NPV, cp.Fmeasure, cp.BA, cp.Gmean, cp.kappa}; %
    %

% Write data in Excel file deep_learning_comparison

% Location and name of data
filename = 'C:\Users\ADMIN\Dropbox\Profesionales Compartidas\Doctorado Juan Carlos Torres\deep_learning_comparison.xlsx';

% Specify the place where it going to be write the data
sheet = 2;
xlRange = 'D8:N8';

% Write the data
xlswrite(filename,ExcelData,sheet,xlRange)

clear; clc; close all




% 3 AlexNet Brazil + IJC 
% Select the MATLAB file
load('E:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\AlexNet\Brasil + IJC Frontal\3ra Corrida\DeepLearningFirstResults.mat')

cp = my_classperf(cm);ExcelData = {cp.Se, cp.Sp, AUC(1,1), AUC(1,2), AUC(1,3), cp.PPV, cp.NPV, cp.Fmeasure, cp.BA, cp.Gmean, cp.kappa}; %
    %

% Write data in Excel file deep_learning_comparison

% Location and name of data
filename = 'C:\Users\ADMIN\Dropbox\Profesionales Compartidas\Doctorado Juan Carlos Torres\deep_learning_comparison.xlsx';

% Specify the place where it going to be write the data
sheet = 2;
xlRange = 'D9:N9';

% Write the data
xlswrite(filename,ExcelData,sheet,xlRange)

clear; clc; close all




% 4 AlexNet Brazil + IJC 
% Select the MATLAB file
load('E:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\AlexNet\Brasil + IJC Frontal\4ta Corrida\DeepLearningFirstResults.mat')

cp = my_classperf(cm);ExcelData = {cp.Se, cp.Sp, AUC(1,1), AUC(1,2), AUC(1,3), cp.PPV, cp.NPV, cp.Fmeasure, cp.BA, cp.Gmean, cp.kappa}; %
    %

% Write data in Excel file deep_learning_comparison

% Location and name of data
filename = 'C:\Users\ADMIN\Dropbox\Profesionales Compartidas\Doctorado Juan Carlos Torres\deep_learning_comparison.xlsx';

% Specify the place where it going to be write the data
sheet = 2;
xlRange = 'D10:N10';

% Write the data
xlswrite(filename,ExcelData,sheet,xlRange)

clear; clc; close all




% 5 AlexNet Brazil + IJC 
% Select the MATLAB file
load('E:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\AlexNet\Brasil + IJC Frontal\5ta Corrida\DeepLearningFirstResults.mat')

cp = my_classperf(cm);ExcelData = {cp.Se, cp.Sp, AUC(1,1), AUC(1,2), AUC(1,3), cp.PPV, cp.NPV, cp.Fmeasure, cp.BA, cp.Gmean, cp.kappa}; %
    %

% Write data in Excel file deep_learning_comparison

% Location and name of data
filename = 'C:\Users\ADMIN\Dropbox\Profesionales Compartidas\Doctorado Juan Carlos Torres\deep_learning_comparison.xlsx';

% Specify the place where it going to be write the data
sheet = 2;
xlRange = 'D11:N11';

% Write the data
xlswrite(filename,ExcelData,sheet,xlRange)

clear; clc; close all


%% GoogleNet

% 1 GoogleNet Brazil 
% Select the MATLAB file
load('E:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\GoogleNet\Brasil Frontal\1ra Corrida\DeepLearningFirstResults.mat')

cp = my_classperf(cm);ExcelData = {cp.Se, cp.Sp, AUC(1,1), AUC(1,2), AUC(1,3), cp.PPV, cp.NPV, cp.Fmeasure, cp.BA, cp.Gmean, cp.kappa}; %
    %

% Write data in Excel file deep_learning_comparison

% Location and name of data
filename = 'C:\Users\ADMIN\Dropbox\Profesionales Compartidas\Doctorado Juan Carlos Torres\deep_learning_comparison.xlsx';

% Specify the place where it going to be write the data
sheet = 2;
xlRange = 'D12:N12';

% Write the data
xlswrite(filename,ExcelData,sheet,xlRange)

clear; clc; close all




% 2 GoogleNet Brazil 
% Select the MATLAB file
load('E:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\GoogleNet\Brasil Frontal\2da Corrida\DeepLearningFirstResults.mat')

cp = my_classperf(cm);ExcelData = {cp.Se, cp.Sp, AUC(1,1), AUC(1,2), AUC(1,3), cp.PPV, cp.NPV, cp.Fmeasure, cp.BA, cp.Gmean, cp.kappa}; %
    %

% Write data in Excel file deep_learning_comparison

% Location and name of data
filename = 'C:\Users\ADMIN\Dropbox\Profesionales Compartidas\Doctorado Juan Carlos Torres\deep_learning_comparison.xlsx';

% Specify the place where it going to be write the data
sheet = 2;
xlRange = 'D13:N13';

% Write the data
xlswrite(filename,ExcelData,sheet,xlRange)

clear; clc; close all




% 3 GoogleNet Brazil 
% Select the MATLAB file
load('E:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\GoogleNet\Brasil Frontal\3ra Corrida\DeepLearningFirstResults.mat')

cp = my_classperf(cm);ExcelData = {cp.Se, cp.Sp, AUC(1,1), AUC(1,2), AUC(1,3), cp.PPV, cp.NPV, cp.Fmeasure, cp.BA, cp.Gmean, cp.kappa}; %
    %

% Write data in Excel file deep_learning_comparison

% Location and name of data
filename = 'C:\Users\ADMIN\Dropbox\Profesionales Compartidas\Doctorado Juan Carlos Torres\deep_learning_comparison.xlsx';

% Specify the place where it going to be write the data
sheet = 2;
xlRange = 'D14:N14';

% Write the data
xlswrite(filename,ExcelData,sheet,xlRange)

clear; clc; close all




% 
% 4 GoogleNet Brazil 
% Select the MATLAB file
load('E:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\GoogleNet\Brasil Frontal\4ta Corrida\DeepLearningFirstResults.mat')

cp = my_classperf(cm);ExcelData = {cp.Se, cp.Sp, AUC(1,1), AUC(1,2), AUC(1,3), cp.PPV, cp.NPV, cp.Fmeasure, cp.BA, cp.Gmean, cp.kappa}; %
    %

% Write data in Excel file deep_learning_comparison

% Location and name of data
filename = 'C:\Users\ADMIN\Dropbox\Profesionales Compartidas\Doctorado Juan Carlos Torres\deep_learning_comparison.xlsx';

% Specify the place where it going to be write the data
sheet = 2;
xlRange = 'D15:N15';

% Write the data
xlswrite(filename,ExcelData,sheet,xlRange)

clear; clc; close all



% 5 GoogleNet Brazil 
% Select the MATLAB file
load('E:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\GoogleNet\Brasil Frontal\5ta Corrida\DeepLearningFirstResults.mat')

cp = my_classperf(cm);ExcelData = {cp.Se, cp.Sp, AUC(1,1), AUC(1,2), AUC(1,3), cp.PPV, cp.NPV, cp.Fmeasure, cp.BA, cp.Gmean, cp.kappa}; %
    %

% Write data in Excel file deep_learning_comparison

% Location and name of data
filename = 'C:\Users\ADMIN\Dropbox\Profesionales Compartidas\Doctorado Juan Carlos Torres\deep_learning_comparison.xlsx';

% Specify the place where it going to be write the data
sheet = 2;
xlRange = 'D16:N16';

% Write the data
xlswrite(filename,ExcelData,sheet,xlRange)

clear; clc; close all





% 1 GoogleNet Brazil + IJC
% Select the MATLAB file
load('E:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\GoogleNet\Brasil + IJC Frontal\1ra Corrida\DeepLearningFirstResults.mat')

cp = my_classperf(cm);ExcelData = {cp.Se, cp.Sp, AUC(1,1), AUC(1,2), AUC(1,3), cp.PPV, cp.NPV, cp.Fmeasure, cp.BA, cp.Gmean, cp.kappa}; %
    %

% Write data in Excel file deep_learning_comparison

% Location and name of data
filename = 'C:\Users\ADMIN\Dropbox\Profesionales Compartidas\Doctorado Juan Carlos Torres\deep_learning_comparison.xlsx';

% Specify the place where it going to be write the data
sheet = 2;
xlRange = 'D17:N17';

% Write the data
xlswrite(filename,ExcelData,sheet,xlRange)

clear; clc; close all





% 2 18
% 2 GoogleNet Brazil + IJC
% Select the MATLAB file
load('E:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\GoogleNet\Brasil + IJC Frontal\2da Corrida\DeepLearningFirstResults.mat')

cp = my_classperf(cm);ExcelData = {cp.Se, cp.Sp, AUC(1,1), AUC(1,2), AUC(1,3), cp.PPV, cp.NPV, cp.Fmeasure, cp.BA, cp.Gmean, cp.kappa}; %
    %

% Write data in Excel file deep_learning_comparison

% Location and name of data
filename = 'C:\Users\ADMIN\Dropbox\Profesionales Compartidas\Doctorado Juan Carlos Torres\deep_learning_comparison.xlsx';

% Specify the place where it going to be write the data
sheet = 2;
xlRange = 'D18:N18';

% Write the data
xlswrite(filename,ExcelData,sheet,xlRange)

clear; clc; close all



% 3 19
% 3 GoogleNet Brazil + IJC
% Select the MATLAB file
load('E:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\GoogleNet\Brasil + IJC Frontal\3ra Corrida\DeepLearningFirstResults.mat')

cp = my_classperf(cm);ExcelData = {cp.Se, cp.Sp, AUC(1,1), AUC(1,2), AUC(1,3), cp.PPV, cp.NPV, cp.Fmeasure, cp.BA, cp.Gmean, cp.kappa}; %
    %

% Write data in Excel file deep_learning_comparison

% Location and name of data
filename = 'C:\Users\ADMIN\Dropbox\Profesionales Compartidas\Doctorado Juan Carlos Torres\deep_learning_comparison.xlsx';

% Specify the place where it going to be write the data
sheet = 2;
xlRange = 'D19:N19';

% Write the data
xlswrite(filename,ExcelData,sheet,xlRange)

clear; clc; close all


% 4 20
% 4 GoogleNet Brazil + IJC
% Select the MATLAB file
load('E:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\GoogleNet\Brasil + IJC Frontal\4ta Corrida\DeepLearningFirstResults.mat')

cp = my_classperf(cm);ExcelData = {cp.Se, cp.Sp, AUC(1,1), AUC(1,2), AUC(1,3), cp.PPV, cp.NPV, cp.Fmeasure, cp.BA, cp.Gmean, cp.kappa}; %
    %

% Write data in Excel file deep_learning_comparison

% Location and name of data
filename = 'C:\Users\ADMIN\Dropbox\Profesionales Compartidas\Doctorado Juan Carlos Torres\deep_learning_comparison.xlsx';

% Specify the place where it going to be write the data
sheet = 2;
xlRange = 'D20:N20';

% Write the data
xlswrite(filename,ExcelData,sheet,xlRange)

clear; clc; close all



% 5 21
% 5 GoogleNet Brazil + IJC
% Select the MATLAB file
load('E:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\GoogleNet\Brasil + IJC Frontal\5ta Corrida\DeepLearningFirstResults.mat')

cp = my_classperf(cm);ExcelData = {cp.Se, cp.Sp, AUC(1,1), AUC(1,2), AUC(1,3), cp.PPV, cp.NPV, cp.Fmeasure, cp.BA, cp.Gmean, cp.kappa}; %
    %

% Write data in Excel file deep_learning_comparison

% Location and name of data
filename = 'C:\Users\ADMIN\Dropbox\Profesionales Compartidas\Doctorado Juan Carlos Torres\deep_learning_comparison.xlsx';

% Specify the place where it going to be write the data
sheet = 2;
xlRange = 'D21:N21';

% Write the data
xlswrite(filename,ExcelData,sheet,xlRange)

clear; clc; close all





%% Inception-v3

% 1 Inception Brazil 
% Select the MATLAB file
load('E:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\Inception v3\Brasil Frontal\1ra Corrida\DeepLearningFirstResults.mat')

cp = my_classperf(cm);ExcelData = {cp.Se, cp.Sp, AUC(1,1), AUC(1,2), AUC(1,3), cp.PPV, cp.NPV, cp.Fmeasure, cp.BA, cp.Gmean, cp.kappa}; %
    %

% Write data in Excel file deep_learning_comparison

% Location and name of data
filename = 'C:\Users\ADMIN\Dropbox\Profesionales Compartidas\Doctorado Juan Carlos Torres\deep_learning_comparison.xlsx';

% Specify the place where it going to be write the data
sheet = 2;
xlRange = 'D22:N22';

% Write the data
xlswrite(filename,ExcelData,sheet,xlRange)

clear; clc; close all






% 2 23
% 2 Inception Brazil 
% Select the MATLAB file
load('E:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\Inception v3\Brasil Frontal\2da Corrida\DeepLearningFirstResults.mat')

cp = my_classperf(cm);ExcelData = {cp.Se, cp.Sp, AUC(1,1), AUC(1,2), AUC(1,3), cp.PPV, cp.NPV, cp.Fmeasure, cp.BA, cp.Gmean, cp.kappa}; %
    %

% Write data in Excel file deep_learning_comparison

% Location and name of data
filename = 'C:\Users\ADMIN\Dropbox\Profesionales Compartidas\Doctorado Juan Carlos Torres\deep_learning_comparison.xlsx';

% Specify the place where it going to be write the data
sheet = 2;
xlRange = 'D23:N23';

% Write the data
xlswrite(filename,ExcelData,sheet,xlRange)

clear; clc; close all


% 3 24
% 3 Inception Brazil 
% Select the MATLAB file
load('E:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\Inception v3\Brasil Frontal\3ra Corrida\DeepLearningFirstResults.mat')

cp = my_classperf(cm);ExcelData = {cp.Se, cp.Sp, AUC(1,1), AUC(1,2), AUC(1,3), cp.PPV, cp.NPV, cp.Fmeasure, cp.BA, cp.Gmean, cp.kappa}; %
    %

% Write data in Excel file deep_learning_comparison

% Location and name of data
filename = 'C:\Users\ADMIN\Dropbox\Profesionales Compartidas\Doctorado Juan Carlos Torres\deep_learning_comparison.xlsx';

% Specify the place where it going to be write the data
sheet = 2;
xlRange = 'D24:N24';

% Write the data
xlswrite(filename,ExcelData,sheet,xlRange)

clear; clc; close all



% 4 25
% 4 Inception Brazil 
% Select the MATLAB file
% load('E:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\Inception v3\Brasil Frontal\4ta Corrida\DeepLearningFirstResults.mat')
% 
% cp = my_classperf(cm);ExcelData = {cp.Se, cp.Sp, AUC(1,1), AUC(1,2), AUC(1,3), cp.PPV, cp.NPV, cp.Fmeasure, cp.BA, cp.Gmean, cp.kappa}; %
%     %
% 
% % Write data in Excel file deep_learning_comparison
% 
% % Location and name of data
% filename = 'C:\Users\ADMIN\Dropbox\Profesionales Compartidas\Doctorado Juan Carlos Torres\deep_learning_comparison.xlsx';
% 
% % Specify the place where it going to be write the data
% sheet = 2;
% xlRange = 'D24:N24';
% 
% % Write the data
% xlswrite(filename,ExcelData,sheet,xlRange)
% 
% clear; clc; close all
% 


%%
% 5 Inception Brazil 
% Select the MATLAB file
load('E:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\Inception v3\Brasil Frontal\5ta Corrida\DeepLearningFirstResults.mat')

cp = my_classperf(cm);ExcelData = {cp.Se, cp.Sp, AUC(1,1), AUC(1,2), AUC(1,3), cp.PPV, cp.NPV, cp.Fmeasure, cp.BA, cp.Gmean, cp.kappa}; %
    %

% Write data in Excel file deep_learning_comparison

% Location and name of data
filename = 'C:\Users\ADMIN\Dropbox\Profesionales Compartidas\Doctorado Juan Carlos Torres\deep_learning_comparison.xlsx';

% Specify the place where it going to be write the data
sheet = 2;
xlRange = 'D26:N26';

% Write the data
xlswrite(filename,ExcelData,sheet,xlRange)

clear; clc; close all





% 1 27
% 1 Inception Brazil + IJC
% Select the MATLAB file
load('E:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\Inception v3\Brasil + IJC Frontal\1ra Corrida\DeepLearningFirstResults.mat')

cp = my_classperf(cm);ExcelData = {cp.Se, cp.Sp, AUC(1,1), AUC(1,2), AUC(1,3), cp.PPV, cp.NPV, cp.Fmeasure, cp.BA, cp.Gmean, cp.kappa}; %
    %

% Write data in Excel file deep_learning_comparison

% Location and name of data
filename = 'C:\Users\ADMIN\Dropbox\Profesionales Compartidas\Doctorado Juan Carlos Torres\deep_learning_comparison.xlsx';

% Specify the place where it going to be write the data
sheet = 2;
xlRange = 'D27:N27';

% Write the data
xlswrite(filename,ExcelData,sheet,xlRange)

clear; clc; close all

% 2 28
% 2 Inception Brazil + IJC
% Select the MATLAB file
load('E:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\Inception v3\Brasil + IJC Frontal\2da Corrida\DeepLearningFirstResults.mat')

cp = my_classperf(cm);ExcelData = {cp.Se, cp.Sp, AUC(1,1), AUC(1,2), AUC(1,3), cp.PPV, cp.NPV, cp.Fmeasure, cp.BA, cp.Gmean, cp.kappa}; %
    %

% Write data in Excel file deep_learning_comparison

% Location and name of data
filename = 'C:\Users\ADMIN\Dropbox\Profesionales Compartidas\Doctorado Juan Carlos Torres\deep_learning_comparison.xlsx';

% Specify the place where it going to be write the data
sheet = 2;
xlRange = 'D28:N28';

% Write the data
xlswrite(filename,ExcelData,sheet,xlRange)

clear; clc; close all


% 3 29
% 3 Inception Brazil + IJC
% Select the MATLAB file
load('E:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\Inception v3\Brasil + IJC Frontal\3ra Corrida\DeepLearningFirstResults.mat')

cp = my_classperf(cm);ExcelData = {cp.Se, cp.Sp, AUC(1,1), AUC(1,2), AUC(1,3), cp.PPV, cp.NPV, cp.Fmeasure, cp.BA, cp.Gmean, cp.kappa}; %
    %

% Write data in Excel file deep_learning_comparison

% Location and name of data
filename = 'C:\Users\ADMIN\Dropbox\Profesionales Compartidas\Doctorado Juan Carlos Torres\deep_learning_comparison.xlsx';

% Specify the place where it going to be write the data
sheet = 2;
xlRange = 'D29:N29';

% Write the data
xlswrite(filename,ExcelData,sheet,xlRange)

clear; clc; close all



% 4 30
% 4 Inception Brazil + IJC
% Select the MATLAB file
load('E:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\Inception v3\Brasil + IJC Frontal\4ta Corrida\DeepLearningFirstResults.mat')

cp = my_classperf(cm);ExcelData = {cp.Se, cp.Sp, AUC(1,1), AUC(1,2), AUC(1,3), cp.PPV, cp.NPV, cp.Fmeasure, cp.BA, cp.Gmean, cp.kappa}; %
    %

% Write data in Excel file deep_learning_comparison

% Location and name of data
filename = 'C:\Users\ADMIN\Dropbox\Profesionales Compartidas\Doctorado Juan Carlos Torres\deep_learning_comparison.xlsx';

% Specify the place where it going to be write the data
sheet = 2;
xlRange = 'D30:N30';

% Write the data
xlswrite(filename,ExcelData,sheet,xlRange)

clear; clc; close all



% 5 Inception Brazil + IJC
% Select the MATLAB file
load('E:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\Inception v3\Brasil + IJC Frontal\5ta Corrida\DeepLearningFirstResults.mat')

cp = my_classperf(cm);ExcelData = {cp.Se, cp.Sp, AUC(1,1), AUC(1,2), AUC(1,3), cp.PPV, cp.NPV, cp.Fmeasure, cp.BA, cp.Gmean, cp.kappa}; %
    %

% Write data in Excel file deep_learning_comparison

% Location and name of data
filename = 'C:\Users\ADMIN\Dropbox\Profesionales Compartidas\Doctorado Juan Carlos Torres\deep_learning_comparison.xlsx';

% Specify the place where it going to be write the data
sheet = 2;
xlRange = 'D31:N31';

% Write the data
xlswrite(filename,ExcelData,sheet,xlRange)

clear; clc; close all


%% ResNet-101

% 1 32
% 1 ResNet-101 Brazil 
% Select the MATLAB file
load('E:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\ResNet 101\Brasil Frontal\1ra Corrida\DeepLearningFirstResults.mat')

cp = my_classperf(cm);ExcelData = {cp.Se, cp.Sp, AUC(1,1), AUC(1,2), AUC(1,3), cp.PPV, cp.NPV, cp.Fmeasure, cp.BA, cp.Gmean, cp.kappa}; %
    %

% Write data in Excel file deep_learning_comparison

% Location and name of data
filename = 'C:\Users\ADMIN\Dropbox\Profesionales Compartidas\Doctorado Juan Carlos Torres\deep_learning_comparison.xlsx';

% Specify the place where it going to be write the data
sheet = 2;
xlRange = 'D32:N32';

% Write the data
xlswrite(filename,ExcelData,sheet,xlRange)

clear; clc; close all



% 2 33
% 2 ResNet-101 Brazil 
% Select the MATLAB file
load('E:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\ResNet 101\Brasil Frontal\2da Corrida\DeepLearningFirstResults.mat')

cp = my_classperf(cm);ExcelData = {cp.Se, cp.Sp, AUC(1,1), AUC(1,2), AUC(1,3), cp.PPV, cp.NPV, cp.Fmeasure, cp.BA, cp.Gmean, cp.kappa}; %
    %

% Write data in Excel file deep_learning_comparison

% Location and name of data
filename = 'C:\Users\ADMIN\Dropbox\Profesionales Compartidas\Doctorado Juan Carlos Torres\deep_learning_comparison.xlsx';

% Specify the place where it going to be write the data
sheet = 2;
xlRange = 'D33:N33';

% Write the data
xlswrite(filename,ExcelData,sheet,xlRange)

clear; clc; close all


% 3 34
% 3 ResNet-101 Brazil 
% Select the MATLAB file
load('E:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\ResNet 101\Brasil Frontal\3ra Corrida\DeepLearningFirstResults.mat')

cp = my_classperf(cm);ExcelData = {cp.Se, cp.Sp, AUC(1,1), AUC(1,2), AUC(1,3), cp.PPV, cp.NPV, cp.Fmeasure, cp.BA, cp.Gmean, cp.kappa}; %
    %

% Write data in Excel file deep_learning_comparison

% Location and name of data
filename = 'C:\Users\ADMIN\Dropbox\Profesionales Compartidas\Doctorado Juan Carlos Torres\deep_learning_comparison.xlsx';

% Specify the place where it going to be write the data
sheet = 2;
xlRange = 'D34:N34';

% Write the data
xlswrite(filename,ExcelData,sheet,xlRange)

clear; clc; close all


% 4 35
% 4 ResNet-101 Brazil 
% Select the MATLAB file
load('E:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\ResNet 101\Brasil Frontal\4ta Corrida\DeepLearningFirstResults.mat')

cp = my_classperf(cm);ExcelData = {cp.Se, cp.Sp, AUC(1,1), AUC(1,2), AUC(1,3), cp.PPV, cp.NPV, cp.Fmeasure, cp.BA, cp.Gmean, cp.kappa}; %
    %

% Write data in Excel file deep_learning_comparison

% Location and name of data
filename = 'C:\Users\ADMIN\Dropbox\Profesionales Compartidas\Doctorado Juan Carlos Torres\deep_learning_comparison.xlsx';

% Specify the place where it going to be write the data
sheet = 2;
xlRange = 'D35:N35';

% Write the data
xlswrite(filename,ExcelData,sheet,xlRange)

clear; clc; close all


% 5 36
% 5 ResNet-101 Brazil 
% Select the MATLAB file
load('E:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\ResNet 101\Brasil Frontal\5ta Corrida\DeepLearningFirstResults.mat')

cp = my_classperf(cm);ExcelData = {cp.Se, cp.Sp, AUC(1,1), AUC(1,2), AUC(1,3), cp.PPV, cp.NPV, cp.Fmeasure, cp.BA, cp.Gmean, cp.kappa}; %
    %

% Write data in Excel file deep_learning_comparison

% Location and name of data
filename = 'C:\Users\ADMIN\Dropbox\Profesionales Compartidas\Doctorado Juan Carlos Torres\deep_learning_comparison.xlsx';

% Specify the place where it going to be write the data
sheet = 2;
xlRange = 'D36:N36';

% Write the data
xlswrite(filename,ExcelData,sheet,xlRange)

clear; clc; close all


% 1 37
% 1 ResNet-101 Brazil + IJC
% Select the MATLAB file
load('E:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\ResNet 101\Brasil + IJC Frontal\1ra Corrida\DeepLearningFirstResults.mat')

cp = my_classperf(cm);ExcelData = {cp.Se, cp.Sp, AUC(1,1), AUC(1,2), AUC(1,3), cp.PPV, cp.NPV, cp.Fmeasure, cp.BA, cp.Gmean, cp.kappa}; %
    %

% Write data in Excel file deep_learning_comparison

% Location and name of data
filename = 'C:\Users\ADMIN\Dropbox\Profesionales Compartidas\Doctorado Juan Carlos Torres\deep_learning_comparison.xlsx';

% Specify the place where it going to be write the data
sheet = 2;
xlRange = 'D37:N37';

% Write the data
xlswrite(filename,ExcelData,sheet,xlRange)

clear; clc; close all




% 2 38
% 2 ResNet-101 Brazil + IJC
% Select the MATLAB file
load('E:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\ResNet 101\Brasil + IJC Frontal\2da Corrida\DeepLearningFirstResults.mat')

cp = my_classperf(cm);ExcelData = {cp.Se, cp.Sp, AUC(1,1), AUC(1,2), AUC(1,3), cp.PPV, cp.NPV, cp.Fmeasure, cp.BA, cp.Gmean, cp.kappa}; %
    %

% Write data in Excel file deep_learning_comparison

% Location and name of data
filename = 'C:\Users\ADMIN\Dropbox\Profesionales Compartidas\Doctorado Juan Carlos Torres\deep_learning_comparison.xlsx';

% Specify the place where it going to be write the data
sheet = 2;
xlRange = 'D38:N38';

% Write the data
xlswrite(filename,ExcelData,sheet,xlRange)

clear; clc; close all




% 3 39
% 3 ResNet-101 Brazil + IJC
% Select the MATLAB file
load('E:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\ResNet 101\Brasil + IJC Frontal\3ra Corrida\DeepLearningFirstResults.mat')

cp = my_classperf(cm);ExcelData = {cp.Se, cp.Sp, AUC(1,1), AUC(1,2), AUC(1,3), cp.PPV, cp.NPV, cp.Fmeasure, cp.BA, cp.Gmean, cp.kappa}; %
    %

% Write data in Excel file deep_learning_comparison

% Location and name of data
filename = 'C:\Users\ADMIN\Dropbox\Profesionales Compartidas\Doctorado Juan Carlos Torres\deep_learning_comparison.xlsx';

% Specify the place where it going to be write the data
sheet = 2;
xlRange = 'D39:N39';

% Write the data
xlswrite(filename,ExcelData,sheet,xlRange)

clear; clc; close all




% 4 40
% 4 ResNet-101 Brazil + IJC
% Select the MATLAB file
load('E:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\ResNet 101\Brasil + IJC Frontal\4ta Corrida\DeepLearningFirstResults.mat')

cp = my_classperf(cm);ExcelData = {cp.Se, cp.Sp, AUC(1,1), AUC(1,2), AUC(1,3), cp.PPV, cp.NPV, cp.Fmeasure, cp.BA, cp.Gmean, cp.kappa}; %
    %

% Write data in Excel file deep_learning_comparison

% Location and name of data
filename = 'C:\Users\ADMIN\Dropbox\Profesionales Compartidas\Doctorado Juan Carlos Torres\deep_learning_comparison.xlsx';

% Specify the place where it going to be write the data
sheet = 2;
xlRange = 'D40:N40';

% Write the data
xlswrite(filename,ExcelData,sheet,xlRange)

clear; clc; close all




% 5 41
% 5 ResNet-101 Brazil + IJC
% Select the MATLAB file
load('E:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\ResNet 101\Brasil + IJC Frontal\5ta Corrida\DeepLearningFirstResults.mat')

cp = my_classperf(cm);ExcelData = {cp.Se, cp.Sp, AUC(1,1), AUC(1,2), AUC(1,3), cp.PPV, cp.NPV, cp.Fmeasure, cp.BA, cp.Gmean, cp.kappa}; %
    %

% Write data in Excel file deep_learning_comparison

% Location and name of data
filename = 'C:\Users\ADMIN\Dropbox\Profesionales Compartidas\Doctorado Juan Carlos Torres\deep_learning_comparison.xlsx';

% Specify the place where it going to be write the data
sheet = 2;
xlRange = 'D41:N41';

% Write the data
xlswrite(filename,ExcelData,sheet,xlRange)

clear; clc; close all



%% ResNet-50
% 1 42
% 1 ResNet-50 Brazil
% Select the MATLAB file
load('E:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\ResNet 50\Brasil Frontal\1ra Corrida\DeepLearningFirstResults.mat')

cp = my_classperf(cm);ExcelData = {cp.Se, cp.Sp, AUC(1,1), AUC(1,2), AUC(1,3), cp.PPV, cp.NPV, cp.Fmeasure, cp.BA, cp.Gmean, cp.kappa}; %
    %

% Write data in Excel file deep_learning_comparison

% Location and name of data
filename = 'C:\Users\ADMIN\Dropbox\Profesionales Compartidas\Doctorado Juan Carlos Torres\deep_learning_comparison.xlsx';

% Specify the place where it going to be write the data
sheet = 2;
xlRange = 'D42:N42';

% Write the data
xlswrite(filename,ExcelData,sheet,xlRange)

clear; clc; close all



% 2 43
% 2 ResNet-50 Brazil
% Select the MATLAB file
load('E:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\ResNet 50\Brasil Frontal\2da Corrida\DeepLearningFirstResults.mat')

cp = my_classperf(cm);ExcelData = {cp.Se, cp.Sp, AUC(1,1), AUC(1,2), AUC(1,3), cp.PPV, cp.NPV, cp.Fmeasure, cp.BA, cp.Gmean, cp.kappa}; %
    %

% Write data in Excel file deep_learning_comparison

% Location and name of data
filename = 'C:\Users\ADMIN\Dropbox\Profesionales Compartidas\Doctorado Juan Carlos Torres\deep_learning_comparison.xlsx';

% Specify the place where it going to be write the data
sheet = 2;
xlRange = 'D43:N43';

% Write the data
xlswrite(filename,ExcelData,sheet,xlRange)

clear; clc; close all


% 3 44
% 3 ResNet-50 Brazil
% Select the MATLAB file
load('E:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\ResNet 50\Brasil Frontal\3ra Corrida\DeepLearningFirstResults.mat')

cp = my_classperf(cm);ExcelData = {cp.Se, cp.Sp, AUC(1,1), AUC(1,2), AUC(1,3), cp.PPV, cp.NPV, cp.Fmeasure, cp.BA, cp.Gmean, cp.kappa}; %
    %

% Write data in Excel file deep_learning_comparison

% Location and name of data
filename = 'C:\Users\ADMIN\Dropbox\Profesionales Compartidas\Doctorado Juan Carlos Torres\deep_learning_comparison.xlsx';

% Specify the place where it going to be write the data
sheet = 2;
xlRange = 'D44:N44';

% Write the data
xlswrite(filename,ExcelData,sheet,xlRange)

clear; clc; close all


% 4 45
% 4 ResNet-50 Brazil
% Select the MATLAB file
load('E:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\ResNet 50\Brasil Frontal\4ta Corrida\DeepLearningFirstResults.mat')

cp = my_classperf(cm);ExcelData = {cp.Se, cp.Sp, AUC(1,1), AUC(1,2), AUC(1,3), cp.PPV, cp.NPV, cp.Fmeasure, cp.BA, cp.Gmean, cp.kappa}; %
    %

% Write data in Excel file deep_learning_comparison

% Location and name of data
filename = 'C:\Users\ADMIN\Dropbox\Profesionales Compartidas\Doctorado Juan Carlos Torres\deep_learning_comparison.xlsx';

% Specify the place where it going to be write the data
sheet = 2;
xlRange = 'D45:N45';

% Write the data
xlswrite(filename,ExcelData,sheet,xlRange)

clear; clc; close all




% 5 46
% 5 ResNet-50 Brazil
% Select the MATLAB file
load('E:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\ResNet 50\Brasil Frontal\5ta Corrida\DeepLearningFirstResults.mat')

cp = my_classperf(cm);ExcelData = {cp.Se, cp.Sp, AUC(1,1), AUC(1,2), AUC(1,3), cp.PPV, cp.NPV, cp.Fmeasure, cp.BA, cp.Gmean, cp.kappa}; %
    %

% Write data in Excel file deep_learning_comparison

% Location and name of data
filename = 'C:\Users\ADMIN\Dropbox\Profesionales Compartidas\Doctorado Juan Carlos Torres\deep_learning_comparison.xlsx';

% Specify the place where it going to be write the data
sheet = 2;
xlRange = 'D46:N46';

% Write the data
xlswrite(filename,ExcelData,sheet,xlRange)

clear; clc; close all



% 1 47
% 1 ResNet-50 Brazil + IJC
% Select the MATLAB file
load('E:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\ResNet 50\Brasil + IJC Frontal\1ra Corrida\DeepLearningFirstResults.mat')

cp = my_classperf(cm);ExcelData = {cp.Se, cp.Sp, AUC(1,1), AUC(1,2), AUC(1,3), cp.PPV, cp.NPV, cp.Fmeasure, cp.BA, cp.Gmean, cp.kappa}; %
    %

% Write data in Excel file deep_learning_comparison

% Location and name of data
filename = 'C:\Users\ADMIN\Dropbox\Profesionales Compartidas\Doctorado Juan Carlos Torres\deep_learning_comparison.xlsx';

% Specify the place where it going to be write the data
sheet = 2;
xlRange = 'D47:N47';

% Write the data
xlswrite(filename,ExcelData,sheet,xlRange)

clear; clc; close all




% 2 48
% 2 ResNet-50 Brazil + IJC
% Select the MATLAB file
load('E:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\ResNet 50\Brasil + IJC Frontal\2da Corrida\DeepLearningFirstResults.mat')

cp = my_classperf(cm);ExcelData = {cp.Se, cp.Sp, AUC(1,1), AUC(1,2), AUC(1,3), cp.PPV, cp.NPV, cp.Fmeasure, cp.BA, cp.Gmean, cp.kappa}; %
    %

% Write data in Excel file deep_learning_comparison

% Location and name of data
filename = 'C:\Users\ADMIN\Dropbox\Profesionales Compartidas\Doctorado Juan Carlos Torres\deep_learning_comparison.xlsx';

% Specify the place where it going to be write the data
sheet = 2;
xlRange = 'D48:N48';

% Write the data
xlswrite(filename,ExcelData,sheet,xlRange)

clear; clc; close all


% 3 49
% 3 ResNet-50 Brazil + IJC
% Select the MATLAB file
load('E:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\ResNet 50\Brasil + IJC Frontal\3ra Corrida\DeepLearningFirstResults.mat')

cp = my_classperf(cm);ExcelData = {cp.Se, cp.Sp, AUC(1,1), AUC(1,2), AUC(1,3), cp.PPV, cp.NPV, cp.Fmeasure, cp.BA, cp.Gmean, cp.kappa}; %
    %

% Write data in Excel file deep_learning_comparison

% Location and name of data
filename = 'C:\Users\ADMIN\Dropbox\Profesionales Compartidas\Doctorado Juan Carlos Torres\deep_learning_comparison.xlsx';

% Specify the place where it going to be write the data
sheet = 2;
xlRange = 'D49:N49';

% Write the data
xlswrite(filename,ExcelData,sheet,xlRange)

clear; clc; close all


% 4 50
% 4 ResNet-50 Brazil + IJC
% Select the MATLAB file
load('E:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\ResNet 50\Brasil + IJC Frontal\4ta Corrida\DeepLearningFirstResults.mat')

cp = my_classperf(cm);ExcelData = {cp.Se, cp.Sp, AUC(1,1), AUC(1,2), AUC(1,3), cp.PPV, cp.NPV, cp.Fmeasure, cp.BA, cp.Gmean, cp.kappa}; %
    %

% Write data in Excel file deep_learning_comparison

% Location and name of data
filename = 'C:\Users\ADMIN\Dropbox\Profesionales Compartidas\Doctorado Juan Carlos Torres\deep_learning_comparison.xlsx';

% Specify the place where it going to be write the data
sheet = 2;
xlRange = 'D50:N50';

% Write the data
xlswrite(filename,ExcelData,sheet,xlRange)

clear; clc; close all


% 5 51
% 5 ResNet-50 Brazil + IJC
% Select the MATLAB file
load('E:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\ResNet 50\Brasil + IJC Frontal\5ta Corrida\DeepLearningFirstResults.mat')

cp = my_classperf(cm);ExcelData = {cp.Se, cp.Sp, AUC(1,1), AUC(1,2), AUC(1,3), cp.PPV, cp.NPV, cp.Fmeasure, cp.BA, cp.Gmean, cp.kappa}; %
    %

% Write data in Excel file deep_learning_comparison

% Location and name of data
filename = 'C:\Users\ADMIN\Dropbox\Profesionales Compartidas\Doctorado Juan Carlos Torres\deep_learning_comparison.xlsx';

% Specify the place where it going to be write the data
sheet = 2;
xlRange = 'D51:N51';

% Write the data
xlswrite(filename,ExcelData,sheet,xlRange)

clear; clc; close all


%% VGG 16

% 1 52
% 1 VGG 16 Brazil 
% Select the MATLAB file
load('E:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\VGG 16\Brasil Frontal\1ra Corrida\DeepLearningFirstResults.mat')

cp = my_classperf(cm);ExcelData = {cp.Se, cp.Sp, AUC(1,1), AUC(1,2), AUC(1,3), cp.PPV, cp.NPV, cp.Fmeasure, cp.BA, cp.Gmean, cp.kappa}; %
    %

% Write data in Excel file deep_learning_comparison

% Location and name of data
filename = 'C:\Users\ADMIN\Dropbox\Profesionales Compartidas\Doctorado Juan Carlos Torres\deep_learning_comparison.xlsx';

% Specify the place where it going to be write the data
sheet = 2;
xlRange = 'D52:N52';

% Write the data
xlswrite(filename,ExcelData,sheet,xlRange)

clear; clc; close all


% 2 53
% 2 VGG 16 Brazil 
% Select the MATLAB file
load('E:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\VGG 16\Brasil Frontal\2da Corrida\DeepLearningFirstResults.mat')

cp = my_classperf(cm);ExcelData = {cp.Se, cp.Sp, AUC(1,1), AUC(1,2), AUC(1,3), cp.PPV, cp.NPV, cp.Fmeasure, cp.BA, cp.Gmean, cp.kappa}; %
    %

% Write data in Excel file deep_learning_comparison

% Location and name of data
filename = 'C:\Users\ADMIN\Dropbox\Profesionales Compartidas\Doctorado Juan Carlos Torres\deep_learning_comparison.xlsx';

% Specify the place where it going to be write the data
sheet = 2;
xlRange = 'D53:N53';

% Write the data
xlswrite(filename,ExcelData,sheet,xlRange)

clear; clc; close all



% 3 54
% 3 VGG 16 Brazil 
% Select the MATLAB file
load('E:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\VGG 16\Brasil Frontal\3ra Corrida\DeepLearningFirstResults.mat')

cp = my_classperf(cm);ExcelData = {cp.Se, cp.Sp, AUC(1,1), AUC(1,2), AUC(1,3), cp.PPV, cp.NPV, cp.Fmeasure, cp.BA, cp.Gmean, cp.kappa}; %
    %

% Write data in Excel file deep_learning_comparison

% Location and name of data
filename = 'C:\Users\ADMIN\Dropbox\Profesionales Compartidas\Doctorado Juan Carlos Torres\deep_learning_comparison.xlsx';

% Specify the place where it going to be write the data
sheet = 2;
xlRange = 'D54:N54';

% Write the data
xlswrite(filename,ExcelData,sheet,xlRange)

clear; clc; close all


% 4 55
% 4 VGG 16 Brazil 
% Select the MATLAB file
load('E:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\VGG 16\Brasil Frontal\4ta Corrida\DeepLearningFirstResults.mat')

cp = my_classperf(cm);ExcelData = {cp.Se, cp.Sp, AUC(1,1), AUC(1,2), AUC(1,3), cp.PPV, cp.NPV, cp.Fmeasure, cp.BA, cp.Gmean, cp.kappa}; %
    %

% Write data in Excel file deep_learning_comparison

% Location and name of data
filename = 'C:\Users\ADMIN\Dropbox\Profesionales Compartidas\Doctorado Juan Carlos Torres\deep_learning_comparison.xlsx';

% Specify the place where it going to be write the data
sheet = 2;
xlRange = 'D55:N55';

% Write the data
xlswrite(filename,ExcelData,sheet,xlRange)

clear; clc; close all

% 5 56
% 5 VGG 16 Brazil 
% Select the MATLAB file
load('E:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\VGG 16\Brasil Frontal\5ta Corrida\DeepLearningFirstResults.mat')

cp = my_classperf(cm);ExcelData = {cp.Se, cp.Sp, AUC(1,1), AUC(1,2), AUC(1,3), cp.PPV, cp.NPV, cp.Fmeasure, cp.BA, cp.Gmean, cp.kappa}; %
    %

% Write data in Excel file deep_learning_comparison

% Location and name of data
filename = 'C:\Users\ADMIN\Dropbox\Profesionales Compartidas\Doctorado Juan Carlos Torres\deep_learning_comparison.xlsx';

% Specify the place where it going to be write the data
sheet = 2;
xlRange = 'D56:N56';

% Write the data
xlswrite(filename,ExcelData,sheet,xlRange)

clear; clc; close all


% 1 57
% 1 VGG 16 Brazil 
% Select the MATLAB file
load('E:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\VGG 16\Brasil + IJC Frontal\1ra Corrida\DeepLearningFirstResults.mat')

cp = my_classperf(cm);ExcelData = {cp.Se, cp.Sp, AUC(1,1), AUC(1,2), AUC(1,3), cp.PPV, cp.NPV, cp.Fmeasure, cp.BA, cp.Gmean, cp.kappa}; %
    %

% Write data in Excel file deep_learning_comparison

% Location and name of data
filename = 'C:\Users\ADMIN\Dropbox\Profesionales Compartidas\Doctorado Juan Carlos Torres\deep_learning_comparison.xlsx';

% Specify the place where it going to be write the data
sheet = 2;
xlRange = 'D57:N57';

% Write the data
xlswrite(filename,ExcelData,sheet,xlRange)

clear; clc; close all



% 2 58
% 2 VGG 16 Brazil 
% Select the MATLAB file
load('E:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\VGG 16\Brasil + IJC Frontal\2da Corrida\DeepLearningFirstResults.mat')

cp = my_classperf(cm);ExcelData = {cp.Se, cp.Sp, AUC(1,1), AUC(1,2), AUC(1,3), cp.PPV, cp.NPV, cp.Fmeasure, cp.BA, cp.Gmean, cp.kappa}; %
    %

% Write data in Excel file deep_learning_comparison

% Location and name of data
filename = 'C:\Users\ADMIN\Dropbox\Profesionales Compartidas\Doctorado Juan Carlos Torres\deep_learning_comparison.xlsx';

% Specify the place where it going to be write the data
sheet = 2;
xlRange = 'D58:N58';

% Write the data
xlswrite(filename,ExcelData,sheet,xlRange)

clear; clc; close all


% 3 59
% 3 VGG 16 Brazil 
% Select the MATLAB file
load('E:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\VGG 16\Brasil + IJC Frontal\3ra Corrida\DeepLearningFirstResults.mat')

cp = my_classperf(cm);ExcelData = {cp.Se, cp.Sp, AUC(1,1), AUC(1,2), AUC(1,3), cp.PPV, cp.NPV, cp.Fmeasure, cp.BA, cp.Gmean, cp.kappa}; %
    %

% Write data in Excel file deep_learning_comparison

% Location and name of data
filename = 'C:\Users\ADMIN\Dropbox\Profesionales Compartidas\Doctorado Juan Carlos Torres\deep_learning_comparison.xlsx';

% Specify the place where it going to be write the data
sheet = 2;
xlRange = 'D59:N59';

% Write the data
xlswrite(filename,ExcelData,sheet,xlRange)

clear; clc; close all


% 4 60
% 4 VGG 16 Brazil 
% Select the MATLAB file
load('E:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\VGG 16\Brasil + IJC Frontal\4ta Corrida\DeepLearningFirstResults.mat')

cp = my_classperf(cm);ExcelData = {cp.Se, cp.Sp, AUC(1,1), AUC(1,2), AUC(1,3), cp.PPV, cp.NPV, cp.Fmeasure, cp.BA, cp.Gmean, cp.kappa}; %
    %

% Write data in Excel file deep_learning_comparison

% Location and name of data
filename = 'C:\Users\ADMIN\Dropbox\Profesionales Compartidas\Doctorado Juan Carlos Torres\deep_learning_comparison.xlsx';

% Specify the place where it going to be write the data
sheet = 2;
xlRange = 'D60:N60';

% Write the data
xlswrite(filename,ExcelData,sheet,xlRange)

clear; clc; close all


% 5 61
% 5 VGG 16 Brazil 
% Select the MATLAB file
load('E:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\VGG 16\Brasil + IJC Frontal\5ta Corrida\DeepLearningFirstResults.mat')

cp = my_classperf(cm);ExcelData = {cp.Se, cp.Sp, AUC(1,1), AUC(1,2), AUC(1,3), cp.PPV, cp.NPV, cp.Fmeasure, cp.BA, cp.Gmean, cp.kappa}; %
    %

% Write data in Excel file deep_learning_comparison

% Location and name of data
filename = 'C:\Users\ADMIN\Dropbox\Profesionales Compartidas\Doctorado Juan Carlos Torres\deep_learning_comparison.xlsx';

% Specify the place where it going to be write the data
sheet = 2;
xlRange = 'D61:N61';

% Write the data
xlswrite(filename,ExcelData,sheet,xlRange)

clear; clc; close all


%% VGG 19
% 1 62
% 1 VGG 19 Brazil 
% Select the MATLAB file
load('E:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\VGG 19\Brasil Frontal\1ra Corrida\DeepLearningFirstResults.mat')

cp = my_classperf(cm);ExcelData = {cp.Se, cp.Sp, AUC(1,1), AUC(1,2), AUC(1,3), cp.PPV, cp.NPV, cp.Fmeasure, cp.BA, cp.Gmean, cp.kappa}; %
    %

% Write data in Excel file deep_learning_comparison

% Location and name of data
filename = 'C:\Users\ADMIN\Dropbox\Profesionales Compartidas\Doctorado Juan Carlos Torres\deep_learning_comparison.xlsx';

% Specify the place where it going to be write the data
sheet = 2;
xlRange = 'D62:N62';

% Write the data
xlswrite(filename,ExcelData,sheet,xlRange)

clear; clc; close all


% 2 63
% 2 VGG 19 Brazil 
% Select the MATLAB file
load('E:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\VGG 19\Brasil Frontal\2da Corrida\DeepLearningFirstResults.mat')

cp = my_classperf(cm);ExcelData = {cp.Se, cp.Sp, AUC(1,1), AUC(1,2), AUC(1,3), cp.PPV, cp.NPV, cp.Fmeasure, cp.BA, cp.Gmean, cp.kappa}; %
    %

% Write data in Excel file deep_learning_comparison

% Location and name of data
filename = 'C:\Users\ADMIN\Dropbox\Profesionales Compartidas\Doctorado Juan Carlos Torres\deep_learning_comparison.xlsx';

% Specify the place where it going to be write the data
sheet = 2;
xlRange = 'D63:N63';

% Write the data
xlswrite(filename,ExcelData,sheet,xlRange)

clear; clc; close all


% 3 64
% 3 VGG 19 Brazil 
% Select the MATLAB file
load('E:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\VGG 19\Brasil Frontal\3ra Corrida\DeepLearningFirstResults.mat')

cp = my_classperf(cm);ExcelData = {cp.Se, cp.Sp, AUC(1,1), AUC(1,2), AUC(1,3), cp.PPV, cp.NPV, cp.Fmeasure, cp.BA, cp.Gmean, cp.kappa}; %
    %

% Write data in Excel file deep_learning_comparison

% Location and name of data
filename = 'C:\Users\ADMIN\Dropbox\Profesionales Compartidas\Doctorado Juan Carlos Torres\deep_learning_comparison.xlsx';

% Specify the place where it going to be write the data
sheet = 2;
xlRange = 'D64:N64';

% Write the data
xlswrite(filename,ExcelData,sheet,xlRange)

clear; clc; close all


% 4 65
% 4 VGG 19 Brazil 
% Select the MATLAB file
load('E:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\VGG 19\Brasil Frontal\4ta Corrida\DeepLearningFirstResults.mat')

cp = my_classperf(cm);ExcelData = {cp.Se, cp.Sp, AUC(1,1), AUC(1,2), AUC(1,3), cp.PPV, cp.NPV, cp.Fmeasure, cp.BA, cp.Gmean, cp.kappa}; %
    %

% Write data in Excel file deep_learning_comparison

% Location and name of data
filename = 'C:\Users\ADMIN\Dropbox\Profesionales Compartidas\Doctorado Juan Carlos Torres\deep_learning_comparison.xlsx';

% Specify the place where it going to be write the data
sheet = 2;
xlRange = 'D65:N65';

% Write the data
xlswrite(filename,ExcelData,sheet,xlRange)

clear; clc; close all


% 5 66
% 5 VGG 19 Brazil 
% Select the MATLAB file
load('E:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\VGG 19\Brasil Frontal\5ta Corrida\DeepLearningFirstResults.mat')

cp = my_classperf(cm);ExcelData = {cp.Se, cp.Sp, AUC(1,1), AUC(1,2), AUC(1,3), cp.PPV, cp.NPV, cp.Fmeasure, cp.BA, cp.Gmean, cp.kappa}; %
    %

% Write data in Excel file deep_learning_comparison

% Location and name of data
filename = 'C:\Users\ADMIN\Dropbox\Profesionales Compartidas\Doctorado Juan Carlos Torres\deep_learning_comparison.xlsx';

% Specify the place where it going to be write the data
sheet = 2;
xlRange = 'D66:N66';

% Write the data
xlswrite(filename,ExcelData,sheet,xlRange)

clear; clc; close all


% 1 67
% 1 VGG 19 Brazil + IJC
% Select the MATLAB file
load('E:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\VGG 19\Brasil + IJC Frontal\1ra Corrida\DeepLearningFirstResults.mat')

cp = my_classperf(cm);ExcelData = {cp.Se, cp.Sp, AUC(1,1), AUC(1,2), AUC(1,3), cp.PPV, cp.NPV, cp.Fmeasure, cp.BA, cp.Gmean, cp.kappa}; %
    %

% Write data in Excel file deep_learning_comparison

% Location and name of data
filename = 'C:\Users\ADMIN\Dropbox\Profesionales Compartidas\Doctorado Juan Carlos Torres\deep_learning_comparison.xlsx';

% Specify the place where it going to be write the data
sheet = 2;
xlRange = 'D67:N67';

% Write the data
xlswrite(filename,ExcelData,sheet,xlRange)

clear; clc; close all



% 2 68
% 2 VGG 19 Brazil + IJC
% Select the MATLAB file
load('E:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\VGG 19\Brasil + IJC Frontal\2da Corrida\DeepLearningFirstResults.mat')

cp = my_classperf(cm);ExcelData = {cp.Se, cp.Sp, AUC(1,1), AUC(1,2), AUC(1,3), cp.PPV, cp.NPV, cp.Fmeasure, cp.BA, cp.Gmean, cp.kappa}; %
    %

% Write data in Excel file deep_learning_comparison

% Location and name of data
filename = 'C:\Users\ADMIN\Dropbox\Profesionales Compartidas\Doctorado Juan Carlos Torres\deep_learning_comparison.xlsx';

% Specify the place where it going to be write the data
sheet = 2;
xlRange = 'D68:N68';

% Write the data
xlswrite(filename,ExcelData,sheet,xlRange)

clear; clc; close all


% 3 69
% 3 VGG 19 Brazil + IJC
% Select the MATLAB file
load('E:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\VGG 19\Brasil + IJC Frontal\3ra Corrida\DeepLearningFirstResults.mat')

cp = my_classperf(cm);ExcelData = {cp.Se, cp.Sp, AUC(1,1), AUC(1,2), AUC(1,3), cp.PPV, cp.NPV, cp.Fmeasure, cp.BA, cp.Gmean, cp.kappa}; %
    %

% Write data in Excel file deep_learning_comparison

% Location and name of data
filename = 'C:\Users\ADMIN\Dropbox\Profesionales Compartidas\Doctorado Juan Carlos Torres\deep_learning_comparison.xlsx';

% Specify the place where it going to be write the data
sheet = 2;
xlRange = 'D69:N69';

% Write the data
xlswrite(filename,ExcelData,sheet,xlRange)

clear; clc; close all


% 4 70
% 4 VGG 19 Brazil + IJC
% Select the MATLAB file
load('E:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\VGG 19\Brasil + IJC Frontal\4ta Corrida\DeepLearningFirstResults.mat')

cp = my_classperf(cm);ExcelData = {cp.Se, cp.Sp, AUC(1,1), AUC(1,2), AUC(1,3), cp.PPV, cp.NPV, cp.Fmeasure, cp.BA, cp.Gmean, cp.kappa}; %
    %

% Write data in Excel file deep_learning_comparison

% Location and name of data
filename = 'C:\Users\ADMIN\Dropbox\Profesionales Compartidas\Doctorado Juan Carlos Torres\deep_learning_comparison.xlsx';

% Specify the place where it going to be write the data
sheet = 2;
xlRange = 'D70:N70';

% Write the data
xlswrite(filename,ExcelData,sheet,xlRange)

clear; clc; close all


% 5 71
% 5 VGG 19 Brazil + IJC
% Select the MATLAB file
load('E:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\VGG 19\Brasil + IJC Frontal\5ta Corrida\DeepLearningFirstResults.mat')

cp = my_classperf(cm);ExcelData = {cp.Se, cp.Sp, AUC(1,1), AUC(1,2), AUC(1,3), cp.PPV, cp.NPV, cp.Fmeasure, cp.BA, cp.Gmean, cp.kappa}; %
    %

% Write data in Excel file deep_learning_comparison

% Location and name of data
filename = 'C:\Users\ADMIN\Dropbox\Profesionales Compartidas\Doctorado Juan Carlos Torres\deep_learning_comparison.xlsx';

% Specify the place where it going to be write the data
sheet = 2;
xlRange = 'D71:N71';

% Write the data
xlswrite(filename,ExcelData,sheet,xlRange)

clear; clc; close all


