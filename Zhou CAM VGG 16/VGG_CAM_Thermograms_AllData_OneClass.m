%% CAM Generation for all the Thermograms

% Application of the CAM structure to all the thermograms in my data base
% for breast cancer and only appears one class

format compact
clear; clc;


% Validation Folders
AbnormalValFolder = 'K:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\Data Augmented Thermography\train-val brasil + ijc frontal\Validation\Abnormal';
resultsAbnormalValFolder = 'K:\Archivos Juan Carlos Torres\MATLAB\CAM Test\VGG Thermograms One Class\Abnormal Validation';
NormalValFolder = 'K:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\Data Augmented Thermography\train-val brasil + ijc frontal\Validation\Normal';
resultsNormalValFolder = 'K:\Archivos Juan Carlos Torres\MATLAB\CAM Test\VGG Thermograms One Class\Normal Validation';

% Training Folders
AbnormalTrainFolder = 'K:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\Data Augmented Thermography\train-val brasil + ijc frontal\Training\Abnormal';
resultsAbnormalTrainFolder = 'K:\Archivos Juan Carlos Torres\MATLAB\CAM Test\VGG Thermograms One Class\Abnormal Training';
NormalTrainFolder = 'K:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\Data Augmented Thermography\train-val brasil + ijc frontal\Training\Normal';
resultsNormalTrainFolder = 'K:\Archivos Juan Carlos Torres\MATLAB\CAM Test\VGG Thermograms One Class\Normal Training';

resultsFolder = 'K:\Archivos Juan Carlos Torres\MATLAB\CAM Test\VGG Thermograms One Class';
codeFolder = 'K:\Archivos Juan Carlos Torres\MATLAB\MATLAB Codigos\Deep Learning\Zhou CAM VGG 16';
addpath('K:\Archivos Juan Carlos Torres\MATLAB\MATLAB Codigos');

% Load the VGG16_CAM Architecture
load ('K:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\Zhou CAM\VGG16_CAM_Modified.mat')

cd(codeFolder);

%% Abnormal Validation Folder
% List all .tif files
dirStruct = dir(fullfile(AbnormalValFolder , '*.tif'));
AbnormValFiles = numel(dirStruct);

% Initialize progress bar
ioi_text_waitbar(0, 'Please wait...');
for iFiles = 1:AbnormValFiles,
    currentFile = dirStruct(iFiles).name;
    FileName = currentFile(1:end-4);
    
                    inputSize = netTransfer.Layers(1).InputSize;
                    classNames = netTransfer.Layers(end).ClassNames;
                    numOfFinalConvFilters = 512; % Being honest it's a matrix [512 512]

                    %% Put the image to analyze
                    
                    cd(AbnormalValFolder)                    
                    TheImg = imread(currentFile);
                    
                    TheImgSize = size(TheImg);
                    TheImg_ = imresize(TheImg, inputSize(1:2));
                    TheResizedImg = TheImg_;

                    TheImg_ = bsxfun(@minus,single(TheImg_ ), mean2(TheImg_ ));

                    [label, scores] = classify(netTransfer, TheResizedImg);
                    label;

                    [s,idx] = sort(scores(:),'descend');
                    idx = idx(1:1:2);
                    classNamesTop = netTransfer.Layers(end).ClassNames(idx);

                    scoresTop = scores(idx)*100;
                    scoresTop = round(scoresTop,2);
                    scoresTop = num2cell(scoresTop);


                    sprintf('Preparation of the Image applied, now obtaining activations of the image, please wait')

                    %% Activations of the Image 
                    HolisFeaturesActivation = activations(netTransfer, TheImg, 'CAM_conv', 'OutputAs', 'channels', 'ExecutionEnvironment', 'cpu');
                    HolisFeaturesActivationSize = size(HolisFeaturesActivation);

                    sprintf('Activations of the layer obtained')


                    HolisActivHeight = size(HolisFeaturesActivation,1);
                    HolisActivWidth = size(HolisFeaturesActivation,2);
                    HolisActivChnnl = size(HolisFeaturesActivation,3);

                    % Reshape 
                    % HolisFeaturesActivation = reshape(gather(HolisFeaturesActivation), [], numOfFinalConvFilters);
                    HolisFeaturesActivation = reshape(gather(HolisFeaturesActivation), [], HolisActivChnnl);
                    HolisFeaturesActivationSize = size(HolisFeaturesActivation);

                    sprintf('Reshape applied, now obtaining the Weights of the layer, please wait')

                    HolisFulConWeights = netTransfer.Layers(36).Weights;
                    size(HolisFulConWeights)
                    % HolisFulConWeights = reshape(gather(HolisFulConWeights), numOfFinalConvFilters, []);
                    HolisFulConWeights = (HolisFulConWeights)'
                    size(HolisFulConWeights)

                    %% 
                    CAM_Thermogram = figure;
                    subplot(1, 2, 1);
                    imshow(TheImg);
                    size(TheImg);
                    % d=cell(1,5);
                    for ii = 1:1   
                        HolisFCA = HolisFeaturesActivation * HolisFulConWeights(:, idx(ii)); 
                        HolisFCA = reshape(HolisFCA, HolisActivHeight, HolisActivWidth);
                        HolisFCA = imresize(HolisFCA,  TheImgSize(1:2));
                        HolisFCASize = size(HolisFCA);
                        mapIm = mat2im(HolisFCA, jet(200), [0 max(HolisFCA(:))]);
                            ClassActivationMap = mapIm*0.6+(single(TheImg)/255)*0.2;


                        subplot(1, 2, ii+1);
                        imshow(mat2gray(imresize (ClassActivationMap, TheImgSize(1:2) )));
                        title (sprintf('%s %G %%', classNamesTop{ii}, scoresTop{ii} ));
                       %title(string('%s, %d %%', classNamesTop{ii}, scoresTop{ii} ) + ", " + num2str(100*scores(classNames == label),3) + "%");
                    % title({d)
                    end
                    
%                     ThermogramName = []
%                     CopyNumber = [' Copy(', num2str(kCopies),').jpg'];
%          nameFile = [currentFile2 CopyNumber];
%          movefile(currentFile, nameFile) 
                    
                    cd(resultsAbnormalValFolder)
                    saveas(CAM_Thermogram, [currentFile , '.jpg']);
                    close all  
    
    
    
    
    % Update progress bar
    ioi_text_waitbar(iFiles/AbnormValFiles, sprintf('Reading file %d from %d', iFiles, AbnormValFiles));
end
% Close progress bar
ioi_text_waitbar('Clear');



%% Normal Validation Folder

cd(codeFolder)

% List all .tif files
dirStruct = dir(fullfile(NormalValFolder , '*.tif'));
NormValFiles = numel(dirStruct);

% Initialize progress bar
ioi_text_waitbar(0, 'Please wait...');
for iFiles = 1:NormValFiles 
    currentFile = dirStruct(iFiles).name;
    FileName = currentFile(1:end-4);
    
                    inputSize = netTransfer.Layers(1).InputSize;
                    classNames = netTransfer.Layers(end).ClassNames;
                    numOfFinalConvFilters = 512; % Being honest it's a matrix [512 512]

                    %% Put the image to analyze
                    
                    cd(NormalValFolder)                    
                    TheImg = imread(currentFile);
                    
                    TheImgSize = size(TheImg);
                    TheImg_ = imresize(TheImg, inputSize(1:2));
                    TheResizedImg = TheImg_;

                    TheImg_ = bsxfun(@minus,single(TheImg_ ), mean2(TheImg_ ));

                    [label, scores] = classify(netTransfer, TheResizedImg);
                    label;

                    [s,idx] = sort(scores(:),'descend');
                    idx = idx(1:1:2);
                    classNamesTop = netTransfer.Layers(end).ClassNames(idx);

                    scoresTop = scores(idx)*100;
                    scoresTop = round(scoresTop,2);
                    scoresTop = num2cell(scoresTop);


                    sprintf('Preparation of the Image applied, now obtaining activations of the image, please wait')

                    %% Activations of the Image 
                    HolisFeaturesActivation = activations(netTransfer, TheImg, 'CAM_conv', 'OutputAs', 'channels', 'ExecutionEnvironment', 'cpu');
                    HolisFeaturesActivationSize = size(HolisFeaturesActivation);

                    sprintf('Activations of the layer obtained')


                    HolisActivHeight = size(HolisFeaturesActivation,1);
                    HolisActivWidth = size(HolisFeaturesActivation,2);
                    HolisActivChnnl = size(HolisFeaturesActivation,3);

                    % Reshape 
                    % HolisFeaturesActivation = reshape(gather(HolisFeaturesActivation), [], numOfFinalConvFilters);
                    HolisFeaturesActivation = reshape(gather(HolisFeaturesActivation), [], HolisActivChnnl);
                    HolisFeaturesActivationSize = size(HolisFeaturesActivation);

                    sprintf('Reshape applied, now obtaining the Weights of the layer, please wait')

                    HolisFulConWeights = netTransfer.Layers(36).Weights;
                    size(HolisFulConWeights)
                    % HolisFulConWeights = reshape(gather(HolisFulConWeights), numOfFinalConvFilters, []);
                    HolisFulConWeights = (HolisFulConWeights)'
                    size(HolisFulConWeights)

                    %% 
                    CAM_Thermogram = figure;
                    subplot(1, 2, 1);
                    imshow(TheImg);
                    size(TheImg);
                    % d=cell(1,5);
                    for ii = 1:1   
                        HolisFCA = HolisFeaturesActivation * HolisFulConWeights(:, idx(ii)); 
                        HolisFCA = reshape(HolisFCA, HolisActivHeight, HolisActivWidth);
                        HolisFCA = imresize(HolisFCA,  TheImgSize(1:2));
                        HolisFCASize = size(HolisFCA);
                        mapIm = mat2im(HolisFCA, jet(200), [0 max(HolisFCA(:))]);
                            ClassActivationMap = mapIm*0.6+(single(TheImg)/255)*0.2;


                        subplot(1, 2, ii+1);
                        imshow(mat2gray(imresize (ClassActivationMap, TheImgSize(1:2) )));
                        title (sprintf('%s %G %%', classNamesTop{ii}, scoresTop{ii} ));
                       %title(string('%s, %d %%', classNamesTop{ii}, scoresTop{ii} ) + ", " + num2str(100*scores(classNames == label),3) + "%");
                    % title({d)
                    end
                    
%                     ThermogramName = []
%                     CopyNumber = [' Copy(', num2str(kCopies),').jpg'];
%          nameFile = [currentFile2 CopyNumber];
%          movefile(currentFile, nameFile) 
                    
                    cd(resultsNormalValFolder)
                    saveas(CAM_Thermogram, [FileName , '.jpg']);
                    close all  
    
    
    
    
    % Update progress bar
    ioi_text_waitbar(iFiles/NormValFiles, sprintf('Reading file %d from %d', iFiles, NormValFiles));
end
% Close progress bar
ioi_text_waitbar('Clear');


%% Abnormal Training Folder

cd(codeFolder)

% List all .tif files
dirStruct = dir(fullfile(AbnormalTrainFolder, '*.tif'));
AbnormalTrainFiles = numel(dirStruct);

% Initialize progress bar
ioi_text_waitbar(0, 'Please wait...');
for iFiles = 1:AbnormalTrainFiles 
    currentFile = dirStruct(iFiles).name;
    FileName = currentFile(1:end-4);
    
                    inputSize = netTransfer.Layers(1).InputSize;
                    classNames = netTransfer.Layers(end).ClassNames;
                    numOfFinalConvFilters = 512; % Being honest it's a matrix [512 512]

                    %% Put the image to analyze
                    
                    cd(AbnormalTrainFolder)                    
                    TheImg = imread(currentFile);
                    
                    TheImgSize = size(TheImg);
                    TheImg_ = imresize(TheImg, inputSize(1:2));
                    TheResizedImg = TheImg_;

                    TheImg_ = bsxfun(@minus,single(TheImg_ ), mean2(TheImg_ ));

                    [label, scores] = classify(netTransfer, TheResizedImg);
                    label;

                    [s,idx] = sort(scores(:),'descend');
                    idx = idx(1:1:2);
                    classNamesTop = netTransfer.Layers(end).ClassNames(idx);

                    scoresTop = scores(idx)*100;
                    scoresTop = round(scoresTop,2);
                    scoresTop = num2cell(scoresTop);


                    sprintf('Preparation of the Image applied, now obtaining activations of the image, please wait')

                    %% Activations of the Image 
                    HolisFeaturesActivation = activations(netTransfer, TheImg, 'CAM_conv', 'OutputAs', 'channels', 'ExecutionEnvironment', 'cpu');
                    HolisFeaturesActivationSize = size(HolisFeaturesActivation);

                    sprintf('Activations of the layer obtained')


                    HolisActivHeight = size(HolisFeaturesActivation,1);
                    HolisActivWidth = size(HolisFeaturesActivation,2);
                    HolisActivChnnl = size(HolisFeaturesActivation,3);

                    % Reshape 
                    % HolisFeaturesActivation = reshape(gather(HolisFeaturesActivation), [], numOfFinalConvFilters);
                    HolisFeaturesActivation = reshape(gather(HolisFeaturesActivation), [], HolisActivChnnl);
                    HolisFeaturesActivationSize = size(HolisFeaturesActivation);

                    sprintf('Reshape applied, now obtaining the Weights of the layer, please wait')

                    HolisFulConWeights = netTransfer.Layers(36).Weights;
                    size(HolisFulConWeights)
                    % HolisFulConWeights = reshape(gather(HolisFulConWeights), numOfFinalConvFilters, []);
                    HolisFulConWeights = (HolisFulConWeights)'
                    size(HolisFulConWeights)

                    %% 
                    CAM_Thermogram = figure;
                    subplot(1, 2, 1);
                    imshow(TheImg);
                    size(TheImg);
                    % d=cell(1,5);
                    for ii = 1:1   
                        HolisFCA = HolisFeaturesActivation * HolisFulConWeights(:, idx(ii)); 
                        HolisFCA = reshape(HolisFCA, HolisActivHeight, HolisActivWidth);
                        HolisFCA = imresize(HolisFCA,  TheImgSize(1:2));
                        HolisFCASize = size(HolisFCA);
                        mapIm = mat2im(HolisFCA, jet(200), [0 max(HolisFCA(:))]);
                            ClassActivationMap = mapIm*0.6+(single(TheImg)/255)*0.2;


                        subplot(1, 2, ii+1);
                        imshow(mat2gray(imresize (ClassActivationMap, TheImgSize(1:2) )));
                        title (sprintf('%s %G %%', classNamesTop{ii}, scoresTop{ii} ));
                       %title(string('%s, %d %%', classNamesTop{ii}, scoresTop{ii} ) + ", " + num2str(100*scores(classNames == label),3) + "%");
                    % title({d)
                    end
                    
%                     ThermogramName = []
%                     CopyNumber = [' Copy(', num2str(kCopies),').jpg'];
%          nameFile = [currentFile2 CopyNumber];
%          movefile(currentFile, nameFile) 
                    
                    cd(resultsAbnormalTrainFolder)
                    saveas(CAM_Thermogram, [FileName , '.jpg']);
                    close all  
    
    
    
    
    % Update progress bar
    ioi_text_waitbar(iFiles/AbnormalTrainFiles, sprintf('Reading file %d from %d', iFiles, AbnormalTrainFiles));
end
% Close progress bar
ioi_text_waitbar('Clear');


%% Normal Training Folder

cd(codeFolder)

% List all .tif files
dirStruct = dir(fullfile(NormalTrainFolder, '*.tif'));
NormalTrainFiles = numel(dirStruct);

% Initialize progress bar
ioi_text_waitbar(0, 'Please wait...');
for iFiles = 1:NormalTrainFiles
    currentFile = dirStruct(iFiles).name;
    FileName = currentFile(1:end-4);
    
                    inputSize = netTransfer.Layers(1).InputSize;
                    classNames = netTransfer.Layers(end).ClassNames;
                    numOfFinalConvFilters = 512; % Being honest it's a matrix [512 512]

                    %% Put the image to analyze
                    
                    cd(NormalTrainFolder)                    
                    TheImg = imread(currentFile);
                    
                    TheImgSize = size(TheImg);
                    TheImg_ = imresize(TheImg, inputSize(1:2));
                    TheResizedImg = TheImg_;

                    TheImg_ = bsxfun(@minus,single(TheImg_ ), mean2(TheImg_ ));

                    [label, scores] = classify(netTransfer, TheResizedImg);
                    label;

                    [s,idx] = sort(scores(:),'descend');
                    idx = idx(1:1:2);
                    classNamesTop = netTransfer.Layers(end).ClassNames(idx);

                    scoresTop = scores(idx)*100;
                    scoresTop = round(scoresTop,2);
                    scoresTop = num2cell(scoresTop);


                    sprintf('Preparation of the Image applied, now obtaining activations of the image, please wait')

                    %% Activations of the Image 
                    HolisFeaturesActivation = activations(netTransfer, TheImg, 'CAM_conv', 'OutputAs', 'channels', 'ExecutionEnvironment', 'cpu');
                    HolisFeaturesActivationSize = size(HolisFeaturesActivation);

                    sprintf('Activations of the layer obtained')


                    HolisActivHeight = size(HolisFeaturesActivation,1);
                    HolisActivWidth = size(HolisFeaturesActivation,2);
                    HolisActivChnnl = size(HolisFeaturesActivation,3);

                    % Reshape 
                    % HolisFeaturesActivation = reshape(gather(HolisFeaturesActivation), [], numOfFinalConvFilters);
                    HolisFeaturesActivation = reshape(gather(HolisFeaturesActivation), [], HolisActivChnnl);
                    HolisFeaturesActivationSize = size(HolisFeaturesActivation);

                    sprintf('Reshape applied, now obtaining the Weights of the layer, please wait')

                    HolisFulConWeights = netTransfer.Layers(36).Weights;
                    size(HolisFulConWeights)
                    % HolisFulConWeights = reshape(gather(HolisFulConWeights), numOfFinalConvFilters, []);
                    HolisFulConWeights = (HolisFulConWeights)'
                    size(HolisFulConWeights)

                    %% 
                    CAM_Thermogram = figure;
                    subplot(1, 2, 1);
                    imshow(TheImg);
                    size(TheImg);
                    % d=cell(1,5);
                    for ii = 1:1   
                        HolisFCA = HolisFeaturesActivation * HolisFulConWeights(:, idx(ii)); 
                        HolisFCA = reshape(HolisFCA, HolisActivHeight, HolisActivWidth);
                        HolisFCA = imresize(HolisFCA,  TheImgSize(1:2));
                        HolisFCASize = size(HolisFCA);
                        mapIm = mat2im(HolisFCA, jet(200), [0 max(HolisFCA(:))]);
                            ClassActivationMap = mapIm*0.6+(single(TheImg)/255)*0.2;


                        subplot(1, 2, ii+1);
                        imshow(mat2gray(imresize (ClassActivationMap, TheImgSize(1:2) )));
                        title (sprintf('%s %G %%', classNamesTop{ii}, scoresTop{ii} ));
                       %title(string('%s, %d %%', classNamesTop{ii}, scoresTop{ii} ) + ", " + num2str(100*scores(classNames == label),3) + "%");
                    % title({d)
                    end
                    
%                     ThermogramName = []
%                     CopyNumber = [' Copy(', num2str(kCopies),').jpg'];
%          nameFile = [currentFile2 CopyNumber];
%          movefile(currentFile, nameFile) 
                    
                    cd(resultsNormalTrainFolder)
                    saveas(CAM_Thermogram, [FileName , '.jpg']);
                    close all  
    
    
    
    
    % Update progress bar
    ioi_text_waitbar(iFiles/NormalTrainFiles, sprintf('Reading file %d from %d', iFiles, NormalTrainFiles));
end
% Close progress bar
ioi_text_waitbar('Clear');

