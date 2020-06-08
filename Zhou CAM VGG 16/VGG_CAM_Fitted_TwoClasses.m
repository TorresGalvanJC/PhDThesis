%% CAM Generation for all the Thermograms Two Classes

% Application of the CAM structure to all the thermograms in my data base
% for breast cancer

format compact
clear; clc;


% Folders
AbnormalValFolder = 'K:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\Termogramas\Brasil + IJC Fitted\Abnormal';
resultsAbnormalValFolder = 'K:\Archivos Juan Carlos Torres\MATLAB\CAM Test\VGG Thermograms Fitted\Two Classes\Abnormal';
NormalValFolder = 'K:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\Termogramas\Brasil + IJC Fitted\Normal';
resultsNormalValFolder = 'K:\Archivos Juan Carlos Torres\MATLAB\CAM Test\VGG Thermograms Fitted\Two Classes\Normal';


resultsFolder = 'K:\Archivos Juan Carlos Torres\MATLAB\CAM Test\VGG Thermograms';
codeFolder = 'K:\Archivos Juan Carlos Torres\MATLAB\MATLAB Codigos\Deep Learning\Zhou CAM VGG 16';
addpath('K:\Archivos Juan Carlos Torres\MATLAB\MATLAB Codigos');

% Load the VGG16_CAM Architecture
load ('K:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\Zhou CAM\VGG16_CAM_Modified.mat')

cd(codeFolder);

%% Abnormal Folder
% List all .tif files
dirStruct = dir(fullfile(AbnormalValFolder , '*.tif'));
AbnormValFiles = numel(dirStruct);

% Initialize progress bar
ioi_text_waitbar(0, 'Please wait...')
for iFiles = 1:AbnormValFiles
    currentFile = dirStruct(iFiles).name;
    FileName = currentFile(1:end-4);
    
                    inputSize = netTransfer.Layers(1).InputSize;
                    classNames = netTransfer.Layers(end).ClassNames;
                    numOfFinalConvFilters = 512; % Being honest it's a matrix [512 512]

                    %% Put the image to analyze
                    
                    cd(AbnormalValFolder)                    
                    TheImg = imread(currentFile);
                    TheImg = padarray(TheImg,[224 224],0,'both');
                                        
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
                    size(HolisFulConWeights);
                    % HolisFulConWeights = reshape(gather(HolisFulConWeights), numOfFinalConvFilters, []);
                    HolisFulConWeights = (HolisFulConWeights)';
                    size(HolisFulConWeights);

                    %% 
                    CAM_Thermogram = figure;
                    subplot(1, 3, 1);
                    imshow(TheImg);
                    size(TheImg);
                    % d=cell(1,5);
                    for ii = 1:2   
                        HolisFCA = HolisFeaturesActivation * HolisFulConWeights(:, idx(ii)); 
                        HolisFCA = reshape(HolisFCA, HolisActivHeight, HolisActivWidth);
                        HolisFCA = imresize(HolisFCA,  TheImgSize(1:2));
                        HolisFCASize = size(HolisFCA);
                        mapIm = mat2im(HolisFCA, jet(200), [0 max(HolisFCA(:))]);
                            ClassActivationMap = mapIm*0.6+(single(TheImg)/255)*0.2;


                        subplot(1, 3, ii+1);
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
    ioi_text_waitbar(iFiles/AbnormValFiles, sprintf('Reading file %d from %d', iFiles, AbnormValFiles))
end
% Close progress bar
ioi_text_waitbar('Clear')



%% Normal Folder

cd(codeFolder)

% List all .tif files
dirStruct = dir(fullfile(NormalValFolder , '*.tif'));
NormValFiles = numel(dirStruct);

% Initialize progress bar
ioi_text_waitbar(0, 'Please wait...')
for iFiles = 1:NormValFiles 
    currentFile = dirStruct(iFiles).name;
    FileName = currentFile(1:end-4);
    
                    inputSize = netTransfer.Layers(1).InputSize;
                    classNames = netTransfer.Layers(end).ClassNames;
                    numOfFinalConvFilters = 512; % Being honest it's a matrix [512 512]

                    %% Put the image to analyze
                    
                    cd(NormalValFolder)                    
                    
                    TheImg = imread(currentFile);
                    TheImg = padarray(TheImg,[224 224],0,'both');
                    
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
                    size(HolisFulConWeights);
                    % HolisFulConWeights = reshape(gather(HolisFulConWeights), numOfFinalConvFilters, []);
                    HolisFulConWeights = (HolisFulConWeights)';
                    size(HolisFulConWeights);

                    %% 
                    CAM_Thermogram = figure;
                    subplot(1, 3, 1);
                    imshow(TheImg);
                    size(TheImg);
                    % d=cell(1,5);
                    for ii = 1:2   
                        HolisFCA = HolisFeaturesActivation * HolisFulConWeights(:, idx(ii)); 
                        HolisFCA = reshape(HolisFCA, HolisActivHeight, HolisActivWidth);
                        HolisFCA = imresize(HolisFCA,  TheImgSize(1:2));
                        HolisFCASize = size(HolisFCA);
                        mapIm = mat2im(HolisFCA, jet(200), [0 max(HolisFCA(:))]);
                            ClassActivationMap = mapIm*0.6+(single(TheImg)/255)*0.2;


                        subplot(1, 3, ii+1);
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
    ioi_text_waitbar(iFiles/NormValFiles, sprintf('Reading file %d from %d', iFiles, NormValFiles))
end
% Close progress bar
ioi_text_waitbar('Clear');
