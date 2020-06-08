format compact
clear; clc;

load ('K:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\Zhou CAM\VGG16_CAM_Modified.mat')

addpath('K:\Archivos Juan Carlos Torres\MATLAB\MATLAB Codigos');
inputSize = netTransfer.Layers(1).InputSize;
classNames = netTransfer.Layers(end).ClassNames;

numOfFinalConvFilters = 512; % Being honest it's a matrix [512 512]

%% Put the image to analyze

TheImg = imread('K:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\Data Augmented Thermography\train-val brasil + ijc frontal\Validation\Normal\Normal_Bra_013.tif');
% TheImg = imread('K:\Archivos Juan Carlos Torres\MATLAB\matconvnet-1.0-beta25\MatConvNet-Saliency-Visualization-master\Images\cat_dog.jpg');

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
figure;
subplot(1, 3, 1);
imshow(TheImg);
size(TheImg)
% d=cell(1,5);
for ii = 1:2   
    HolisFCA = HolisFeaturesActivation * HolisFulConWeights(:, idx(ii)); 
    HolisFCA = reshape(HolisFCA, HolisActivHeight, HolisActivWidth);
    HolisFCA = imresize(HolisFCA,  TheImgSize(1:2));
    HolisFCASize = size(HolisFCA)
    mapIm = mat2im(HolisFCA, jet(256), [0 max(HolisFCA(:))]);
        ClassActivationMap = mapIm*0.6+(single(TheImg)/255)*0.2;


    subplot(1, 3, ii+1);
    imshow(mat2gray(imresize (ClassActivationMap, TheImgSize(1:2) )));
    title (sprintf('%s %G %%', classNamesTop{ii}, scoresTop{ii} ));
   %title(string('%s, %d %%', classNamesTop{ii}, scoresTop{ii} ) + ", " + num2str(100*scores(classNames == label),3) + "%");
% title({d)
end

% cd(resultsAbnormalValFolder)
% saveas(CAM_Thermogram, [currentFile , '.jpg']);
% close all  

