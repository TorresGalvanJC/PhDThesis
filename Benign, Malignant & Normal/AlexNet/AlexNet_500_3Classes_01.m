%% *AlexNet Data Augmented 500 Brasil+IJC *

clear, close all, clc;

% Folders Location
codeFolder = 'K:\Archivos Juan Carlos Torres\MATLAB\MATLAB Codigos\Benign, Malignant & Normal\AlexNet';
resultsFolder = 'K:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\Termogramas\Tumors\AlexNet\500 Times';
addpath('K:\Archivos Juan Carlos Torres\MATLAB\MATLAB Codigos')

TrainingFolder = 'K:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\Termogramas\Tumors\Data Augmented\500 Times';
ValidationFolder = 'K:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\Termogramas\Tumors\Thermograms\Validation 500 Times';

nLoops = 30
ioi_text_waitbar(0, 'Please wait...');

for qq = 1:nLoops
    close all
    %% imageDatastore
    
    % Two different folders, one for each imageDatastore
    
    trainingImages = imageDatastore(TrainingFolder, ...
        'IncludeSubfolders',true,...
        'FileExtensions','.tif','LabelSource','foldernames');
    
    trainingImages.ReadFcn = @(loc)imresize(imread(loc),[227, 227]);
    
    validationImages = imageDatastore(ValidationFolder, ...
        'IncludeSubfolders',true,...
        'FileExtensions','.tif','LabelSource','foldernames');
    
    validationImages.ReadFcn = @(loc)imresize(imread(loc),[227, 227]);
    
    
    %% Data Augmenter
    
    % Preprocesing of the images
    
    augmenter = imageDataAugmenter(...
        'FillValue', [12 12 12],...
        'RandXReflection', true,...
        'RandYReflection', true,...
        'RandRotation', [0 359],...
        'RandXShear', [-90 90],...
        'RandYShear', [-90 90],...
        'RandXTranslation', [0 30],...
        'RandYTranslation', [0 30])
    
    % Generate batches of augmented image data
    ImageSource = augmentedImageSource([227 227], trainingImages,'DataAugmentation', augmenter)
    
    %%
    numTrainImages = numel(trainingImages.Labels);
    idx = randperm(numTrainImages,16);
    figure
    for i = 1:16
        subplot(4,4,i)
        I = readimage(trainingImages,idx(i));
        label = trainingImages.Labels(idx(i));
        imshow(I)
        title(char(label))
    end
    %%
    net = alexnet;
    net.Layers
    
    layersTransfer = net.Layers(1:end-3);
    
    numClasses = numel(categories(trainingImages.Labels))
    layers = [
        layersTransfer
        fullyConnectedLayer(numClasses,...
        'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
        softmaxLayer
        classificationLayer];
    
    
    % Train Network, specify the training options
    miniBatchSize = 10;
    % numIterationsPerEpoch = floor(numel(trainingImages.Labels)/miniBatchSize);
    % numIterationsPerEpoch = ;
    
    options = trainingOptions('sgdm',...
        'ExecutionEnvironment', 'gpu',...
        'MiniBatchSize',miniBatchSize,...
        'MaxEpochs',20,...
        'InitialLearnRate',1e-4,...
        'Verbose',false,...
        'Plots','training-progress',...
        'Shuffle','every-epoch',...
        'ValidationData',validationImages,...
        'ValidationFrequency',50);
    netTransfer = trainNetwork(trainingImages,layers,options);
    
    % currentfig=findall(groot,'Type','Figure');
    % fileTrainingImage=strcat(resultsFolder, 'TrainingProgress.png');
    % saveas(currentfig,fileTrainingImage)
    
    % Classify the validation images using the fine-tuned network.
    [predictedLabels,scores] = classify(netTransfer,validationImages, 'ExecutionEnvironment','cpu');
    % Display four sample validation images with their predicted labels.
    
    idx = randperm(20,4);
    figure
    for i = 1:numel(idx)
        subplot(2,2,i)
        I = readimage(validationImages,idx(i));
        label = predictedLabels(idx(i));
        imshow(I)
        title(char(label))
    end
    %% Calculate the classification accuracy on the validation set.
    % Accuracy is the fraction of labels that the network predicts correctly.
    
    valLabels = validationImages.Labels;
    accuracy = mean(predictedLabels == valLabels);
    targets = grp2idx(valLabels);
    targets(targets==3) = 0; % Normal= 0; 1= Benign 2 = Malignant
    % targets = ~targets;
    outputs = grp2idx(predictedLabels);
    
    outputs(outputs==3) = 0; % Normal= 0; 1= Benign 2 = Malignant
    
    % outputs = ~outputs;
    [cm, order] = confusionmat(targets', outputs');
    
    %% Pretty Confusion Matrix
    tt = table(validationImages.Labels,predictedLabels,'VariableNames',{'Actual','Predicted'});
    Confusion_Matrix1 = figure;
    heatmap(tt,'Predicted','Actual');
    
    cd(resultsFolder)
    saveas(Confusion_Matrix1, 'Confusion_Matrix_1.jpg');
    cd(codeFolder)
    
    %% Or create a more 'sophisticated' confusion matrix
    tbl = validationImages.countEachLabel;
    t = zeros(3,length(predictedLabels));
    y = t;
    for ii = 1:3
        y(ii,:) = predictedLabels == tbl.Label(ii);
        t(ii,:) = validationImages.Labels == tbl.Label(ii);
    end
    
    Confusion_Matrix2=figure;
    plotconfusion(t,y);
    cd(resultsFolder)
    saveas(Confusion_Matrix2, 'Confusion_Matrix2.jpg');
    cd(codeFolder)
    
    %%
    
    addpath(genpath(resultsFolder))
    cp = getValuesMultiClass(cm);
    
    %%
    % ROC_1=figure; plotroc(targets', scores(:,2)');
    %
    % cd(resultsFolder)
    % saveas(ROC_1, 'ROC_1.jpg');
    % cd(codeFolder)
    %
    % [X,Y,T,AUC,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(targets', scores(:,2)',1,...
    %     'NBoot',1000,'TVals',0:0.05:1);
    % ROC_2=figure; errorbar(X(:,1),Y(:,1),Y(:,1)-Y(:,2),Y(:,3)-Y(:,1));
    %
    % cd(resultsFolder)
    % saveas(ROC_2, 'ROC_2.jpg');
    % cd(codeFolder)
    
    
    idx = randperm(20,9);
    Second_Figure = figure
    for i = 1:numel(idx)
        subplot(3,3,i)
        I = readimage(validationImages,idx(i));
        label = predictedLabels(idx(i));
        imshow(I)
        title(char(label))
    end
    
    cd(resultsFolder)
    saveas(Second_Figure, 'Second_Figure.jpg');
    
    
    cd(resultsFolder)
    save(['Calis_' num2str(qq) '.mat']);
    
    cd(codeFolder)
    close all
    ioi_text_waitbar(qq/nLoops, sprintf('AlexNet 500, Running loop %d from %d', qq, nLoops));
end
% Close progress bar
ioi_text_waitbar('Clear');

% End of Script