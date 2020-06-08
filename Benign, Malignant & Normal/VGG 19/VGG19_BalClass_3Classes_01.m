%%

clear, close all, clc;

% Folders Location
codeFolder = 'K:\Archivos Juan Carlos Torres\MATLAB\MATLAB Codigos\Benign, Malignant & Normal\VGG 19';
resultsFolder = 'K:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\Termogramas\Tumors\VGG 19\Balanced Class';
addpath('K:\Archivos Juan Carlos Torres\MATLAB\MATLAB Codigos')

TrainingFolder = 'K:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\Termogramas\Tumors\Data Augmented\Balanced Class';
ValidationFolder = 'K:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\Termogramas\Tumors\Thermograms\Validation Balanced Class';

nLoops = 30
ioi_text_waitbar(0, 'Please wait...');

for qq = 1:nLoops
    close all
    
    %% imageDatastore
    
    % Two different folders, one for each imageDatastore
    
    trainingImages = imageDatastore(TrainingFolder, ...
        'IncludeSubfolders',true,...
        'FileExtensions','.tif','LabelSource','foldernames');
    
    trainingImages.ReadFcn = @(loc)imresize(imread(loc),[224, 224]);
    
    validationImages = imageDatastore(ValidationFolder, ...
        'IncludeSubfolders',true,...
        'FileExtensions','.tif','LabelSource','foldernames');
    
    validationImages.ReadFcn = @(loc)imresize(imread(loc),[224, 224]);
    
    %% Data Augmenter
    
    % Preprocesing of the images
    
    augmenter = imageDataAugmenter('FillValue', [12 12 12],...
        'RandXReflection', true,...
        'RandYReflection', true,...
        'RandRotation', [0 359],...
        'RandXShear', [-90 90],...
        'RandYShear', [-90 90],...
        'RandXTranslation', [0 30],...
        'RandYTranslation', [0 30])
    
    % Generate batches of augmented image data
    ImageSource = augmentedImageSource([224 224], trainingImages,'DataAugmentation', augmenter)
    
    %%
    numTrainImages = numel(trainingImages.Labels);
%     idx = randperm(numTrainImages, 16);
%     First_Figure = figure
%     for i = 1:16
%         subplot(4, 4, i)
%         I = readimage(trainingImages, idx(i));
%         label = trainingImages.Labels(idx(i));
%         imshow(I)
%         title(char(label))
%     end
%     
%     cd(resultsFolder)
%     saveas(First_Figure, 'First_Figure.jpg');
%     cd(codeFolder)
    
    %%
    % Load the pretrained vgg16 network
    
    net = vgg19;
    net.Layers
    %% Modify Layers
    % We have to modify the last three layers of the network and add three other
    % layers
    
    % Take care that te final fully conected layer to have the same size as the
    % number of classes in the new data set
    
    % To learn faster in the new layers thatn in the transferred layers, increase
    % the learning rate factors of the fully connected layer
    layersTransfer = net.Layers(1:end-3)
    
    
    
    numClasses = numel(categories(trainingImages.Labels))
    
    
    
    ioi_text_waitbar(qq/nLoops, sprintf('VGG 19 Balanced Class, Running loop %d from %d', qq, nLoops));
    
    layers = [
        layersTransfer
        fullyConnectedLayer(numClasses,...
        'WeightLearnRateFactor', 20,...
        'BiasLearnRateFactor', 20)
        softmaxLayer
        classificationLayer];
    
    miniBatchSize = 10
    numIterationsPerEpoch = 50;
    options = trainingOptions('sgdm',...
        'ExecutionEnvironment', 'gpu',...
        'MiniBatchSize',miniBatchSize,...
        'MaxEpochs',4,...
        'InitialLearnRate',1e-4,...
        'VerboseFrequency',1,...
        'Plots','training-progress',...
        'ValidationData',validationImages,...
        'ValidationFrequency',numIterationsPerEpoch);
    netTransfer = trainNetwork(trainingImages,layers,options);
    
    
    %% Classify the validation images using the fine-tuned network.
    
    [predictedLabels, scores] = classify(netTransfer,validationImages, 'ExecutionEnvironment','cpu');
    
%     idx = randperm(20,4);
%     figure
%     for i = 1:numel(idx)
%         subplot(2,2,i)
%         I = readimage(validationImages,idx(i));
%         label = predictedLabels(idx(i));
%         imshow(I)
%         title(char(label))
%     end
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
    
    
%     idx = randperm(20,9);
%     Second_Figure = figure
%     for i = 1:numel(idx)
%         subplot(3,3,i)
%         I = readimage(validationImages,idx(i));
%         label = predictedLabels(idx(i));
%         imshow(I)
%         title(char(label))
%     end
%     
%     cd(resultsFolder)
%     saveas(Second_Figure, 'Second_Figure.jpg');
    
    cd(resultsFolder)
    
    save(['Calis_' num2str(qq) '.mat'])
    
    cd(codeFolder)
    close all
    
end
% Close progress bar
ioi_text_waitbar('Clear');

% End of Script