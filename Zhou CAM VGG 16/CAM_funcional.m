%% Zhou VGG-16 CAM Architecture working

% Use of the Zhou's network that makes possible the application of Class
% Activation Mapping in MATLAB

% The architecture has a little modification charging the categories 

clc; clear; format compact

protofile = 'K:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\Zhou CAM\deploy_vgg16CAM.prototxt';
datafile = 'K:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\Zhou CAM\vgg16CAM_train_iter_90000.caffemodel';
load('K:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\Zhou CAM\categories1000.mat')
net = importCaffeNetwork(protofile,datafile)

addpath('K:\Archivos Juan Carlos Torres\MATLAB\MATLAB Codigos');
addpath('K:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\Zhou CAM\Test Images');
inputSize = net.Layers(1).InputSize;
classNames = net.Layers(end).ClassNames;

numOfFinalConvFilters = 512; % Being honest it's a matrix [512 512]

%% Put the image to analyze

% TheImg = imread('zebra_elephant.jpg');
% TheImg = imread('peppers.png');
% TheImg = imread('Sdla.jpg');
% TheImg = imread('1383239551554.jpeg');
% TheImg = imread('Capitol_at_Dusk_2.jpg');
% TheImg = imread('istock-103332974.jpg');
 TheImg = imread('chocolatecalienteconvino.jpg');

TheImgSize = size(TheImg);
TheImg_ = imresize(TheImg, inputSize(1:2));
TheResizedImg = TheImg_;

TheImg_ = bsxfun(@minus,single(TheImg_ ), mean2(TheImg_ ));

[label, scores] = classify(net, TheResizedImg);
label;

[s,idx] = sort(scores(:),'descend');
idx = idx(1:1:5);
classNamesTop = net.Layers(end).ClassNames(idx);

scoresTop = scores(idx)*100;
scoresTop = round(scoresTop,2);
scoresTop = num2cell(scoresTop);


sprintf('Preparation of the Image applied, now obtaining activations of the image, please wait')

%% Activations of the Image 
HolisFeaturesActivation = activations(net, TheImg, 'CAM_conv', 'OutputAs', 'channels', 'ExecutionEnvironment', 'cpu');
HolisFeaturesActivationSize = size(HolisFeaturesActivation);

sprintf('Activations of the layer obtained')


HolisActivHeight = size(HolisFeaturesActivation,1);
HolisActivWidth = size(HolisFeaturesActivation,2);
HolisActivChnnl = size(HolisFeaturesActivation,3);

% Reshape 
% HolisFeaturesActivation = reshape(gather(HolisFeaturesActivation), [], numOfFinalConvFilters);
HolisFeaturesActivation2 = reshape(gather(HolisFeaturesActivation), [], HolisActivChnnl);
HolisFeaturesActivationSize = size(HolisFeaturesActivation2);

sprintf('Reshape applied, now obtaining the Weights of the layer, please wait')

HolisFulConWeights = net.Layers(36).Weights;
size(HolisFulConWeights)
% HolisFulConWeights = reshape(gather(HolisFulConWeights), numOfFinalConvFilters, []);
HolisFulConWeights = (HolisFulConWeights)';
size(HolisFulConWeights)

%% 
figure;
subplot(2, 3, 1);
imshow(TheImg);
size(TheImg)
% d=cell(1,5);
for ii = 1:5   
    HolisFCA = HolisFeaturesActivation2 * HolisFulConWeights(:, idx(ii)); 
    HolisFCA = reshape(HolisFCA, HolisActivHeight, HolisActivWidth);
    HolisFCA = imresize(HolisFCA,  TheImgSize(1:2));
    HolisFCASize = size(HolisFCA)
    mapIm = mat2im(HolisFCA, jet(256), [0 max(HolisFCA(:))]);
    
    ClassActivationMap = mapIm*0.6+(single(TheImg)/255)*0.2;
    subplot(2, 3, ii+1);
    
    %imagesc(mat2gray(imresize (ClassActivationMap, TheImgSize(1:2) )), [0 1]);
    imshow(mat2gray(imresize (ClassActivationMap, TheImgSize(1:2) )));
    % colormap(jet(256))
    % colorbar
    
    classNumber = (classNamesTop{ii}(6:end));
    classNumber = str2num(classNumber);
    NameOfClass = categories{classNumber}(10:end);
    
    title (sprintf('%s %G %%', NameOfClass, scoresTop{ii} ));
   %title(string('%s, %d %%', classNamesTop{ii}, scoresTop{ii} ) + ", " + num2str(100*scores(classNames == label),3) + "%");
% title({d)
end

% cd(resultsAbnormalValFolder)
% saveas(CAM_Thermogram, [currentFile , '.jpg']);
% close all  