% *ResNet 101 for Database of Brasil, IJC & ISSSTE*

% Unbalanced Class
% MiniBatch = 10

% Analysis of three databases is realized for compare the results, only the
% results that has a Sensitivity bigger than 80% is saved

clear; close all; clc;

% Folders Location
codeFolder = 'C:\Users\LANCYTT\Documents\Juan Carlos Torres\Thermograms Analysis';
% addpath('I:\Archivos Juan Carlos Torres\MATLAB\MATLAB Codigos');

resultsFolder = 'C:\Users\LANCYTT\Documents\Juan Carlos Torres\Results\ResNet101\Iteraciones\Balanced';
addpath('C:\Users\LANCYTT\Documents\Juan Carlos Torres\Thermograms Analysis');

TrainingFolder = 'C:\Users\LANCYTT\Documents\Juan Carlos Torres\Data Augmented Thermography\Data Augmented Bra+IJC+ISSSTE\Balanced Augmented Modify';
ValidationFolder = 'C:\Users\LANCYTT\Documents\Juan Carlos Torres\Data Augmented Thermography\Train - Valid Bra+IJC+ISSSTE\Validation Balanced Class';

pat_sendmail

ioi_text_waitbar(0, 'Running ResNet 101, Balanced MiniBatch 10... Please wait...');

Name=sprintf('ResNet101 Balanced');
nLoops = 1090;

for qq = 1081:nLoops
    
      
    ioi_text_waitbar(qq/nLoops, sprintf('ResNet101 Balanced MiniBatch=10, Running loop %d from %d', qq, 1440))
    
    close all
    trainingImages = imageDatastore(TrainingFolder, ...
        'IncludeSubfolders',true,...
        'FileExtensions','.tif','LabelSource','foldernames');
    
    trainingImages.ReadFcn = @(loc)imresize(imread(loc),[224, 224]);
    
    validationImages = imageDatastore(ValidationFolder, ...
        'IncludeSubfolders',true,...
        'FileExtensions','.tif','LabelSource','foldernames');
    
    validationImages.ReadFcn = @(loc)imresize(imread(loc),[224, 224]);
    
    
    % Load the pretrained GoogleNet network
    
    net = resnet101;
    net.Layers;
    
    % Extract the layer graph from the trained network and plot the layer graph.
    
    lgraph = layerGraph(net);
    Architecture_1 = figure('Units','normalized','Position',[0.1 0.1 0.8 0.8]);
    plot(lgraph);
    
       
    lgraph = removeLayers(lgraph, {'fc1000', 'prob', 'ClassificationLayer_predictions'});
    
    numClasses = numel(categories(trainingImages.Labels));
    newLayers = [
        fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',1,'BiasLearnRateFactor', 1)
        softmaxLayer('Name','softmax')
        classificationLayer('Name','classoutput')];
    lgraph = addLayers(lgraph,newLayers);
    
    lgraph = connectLayers(lgraph,'pool5','fc');
    
    Architecture_2 = figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
    plot(lgraph);
    ylim([0,10]);
    
    miniBatchSize = 10;
    
    %numIterationsPerEpoch = floor(numel(trainingImages.Labels)/miniBatchSize);
    options = trainingOptions('sgdm',...
        'ExecutionEnvironment','gpu',...
        'MiniBatchSize',miniBatchSize,...
        'MaxEpochs',100,...
        'InitialLearnRate',1e-5,...
        'VerboseFrequency',1,...
        'Plots','training-progress',...
        'ValidationData',validationImages,...
        'ValidationFrequency',5);
    netTransfer = trainNetwork(trainingImages,lgraph,options);
    
    %% Classify the validation images using the fine-tuned network.
    
    [predictedLabels, scores] = classify(netTransfer,validationImages,'ExecutionEnvironment','cpu');
    accuracy = mean(predictedLabels == validationImages.Labels);
    targets = grp2idx(validationImages.Labels);
    targets(targets==2) = 0;
    targets = ~targets;
    outputs = grp2idx(predictedLabels);
    outputs(outputs==2) = 0;
    outputs = ~outputs;
    [c,cm,ind,per] = confusion(targets', outputs');
    
    Confusion_Matrix=figure; plotconfusion(targets', outputs');
    
    %%
    
    addpath(genpath(resultsFolder))
    cp = my_classperf(cm);
    ROC_1=figure; plotroc(targets', scores(:,2)');
    
    [X,Y,T,AUC,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(targets', scores(:,2)',1,...
        'NBoot',1000,'TVals',0:0.05:1);
    ROC_2=figure; errorbar(X(:,3),Y(:,1),Y(:,1)-Y(:,2),Y(:,3)-Y(:,1));
    TheTableNice = table(validationImages.Labels,predictedLabels,'VariableNames',{'Actual','Predicted'});
    Confusion_Matrix1 = figure;
    PrettyCM = heatmap(TheTableNice,'Predicted','Actual');
    
    %     idx = randperm(20,4);
    %     Second_Figure = figure
    %     for i = 1:numel(idx)
    %         subplot(2,2,i)
    %         I = readimage(validationImages,idx(i));
    %         label = predictedLabels(idx(i));
    %         imshow(I)
    %         title(char(label))
    %     end
    
    try
        msg=sprintf('%s loop %d from %d MiniBatchSize %d with the results %d', Name, qq, nLoops, miniBatchSize, cp);
        sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
    catch
    end
    cd(resultsFolder);
    if  cp.Se > 0.79
        
        saveas(PrettyCM, ['BalCl_MB10_PrettyCM_' num2str(qq) '.jpg']);
        
        saveas(ROC_1, ['BalCl_MB10_ROC_1_' num2str(qq) '.jpg']);
        saveas(ROC_2, ['BalCl_MB10_ROC_2_' num2str(qq) '.jpg']);
        
        TrainingNeuNetw = findall(groot, 'Type', 'Figure');
        
        save(['RN_BalCl_MinBat10_' num2str(qq) '.mat'],'accuracy', 'AUC',...
            'c', 'cm', 'Confusion_Matrix', 'cp', ...
            'ind', 'miniBatchSize', 'netTransfer',...
            'numClasses', 'OPTROCPT', 'outputs', 'per', 'predictedLabels',...
            'scores','SUBY', 'SUBYNAMES', 'T', 'targets',...
            'TrainingNeuNetw','TheTableNice', 'validationImages', 'X', 'Y');
        %saveas(Architecture_1, ['UnCl_MB10_Architecture_1_' num2str(qq) '.jpg']);
        %saveas(Architecture_2, ['UnCl_MB10_Architecture_2_' num2str(qq) '.jpg']);
        
        %saveas(Second_Figure, ['UnCl_MB10_2nd_Figure_' num2str(qq) '.jpg']);
        
        
        
        try
            msg=sprintf('%s loop %d from %d MiniBatchSize %d , Sensitivity = %d , Balanced Accuracy = %d',...
                Name, qq, nLoops, miniBatchSize, cp.Se, cp.BA);
            sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
        catch
        end
        
%         close all
        % Save the figure of Training Progress in PNG format
        % saveas(TrainingNeuNetw,['RN_UnCl_MinBat10_' num2str(qq) '.png']);
        
        
%     else
%         save(['RN_BalCl_MinBat10_' num2str(qq) '.mat'],'accuracy', 'AUC',...
%             'cm', 'Confusion_Matrix', 'cp',...
%             'outputs', 'per', 'scores','SUBY', 'SUBYNAMES', 'T', 'targets',...
%             'TheTableNice', 'validationImages', 'X', 'Y');
%         saveas(PrettyCM, ['BalCl_MB10_PrettyCM_' num2str(qq) '.jpg']);
%         
%         try
%             msg=sprintf('%s loop %d from %d MiniBatchSize %d , Another one bits the dust, Sensitivity = %d',...
%                 Name, qq, nLoops, miniBatchSize, cp.Se);
%             sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
%         catch
%         end
        % Close all the figures
        
    end
    
    close all
    delete(findall(0));
    cd(codeFolder);
    % clearvars -except codeFolder resultsFolder TrainingFolder ValidationFolder nLoops qq
    
end

% Close progress bar
ioi_text_waitbar('Clear');

nLoops = 1100;

for qq = 1091:nLoops
    
      
    ioi_text_waitbar(qq/nLoops, sprintf('ResNet101 Balanced MiniBatch=10, Running loop %d from %d', qq, 1440))
    
    close all
    trainingImages = imageDatastore(TrainingFolder, ...
        'IncludeSubfolders',true,...
        'FileExtensions','.tif','LabelSource','foldernames');
    
    trainingImages.ReadFcn = @(loc)imresize(imread(loc),[224, 224]);
    
    validationImages = imageDatastore(ValidationFolder, ...
        'IncludeSubfolders',true,...
        'FileExtensions','.tif','LabelSource','foldernames');
    
    validationImages.ReadFcn = @(loc)imresize(imread(loc),[224, 224]);
    
    
    % Load the pretrained GoogleNet network
    
    net = resnet101;
    net.Layers;
    
    % Extract the layer graph from the trained network and plot the layer graph.
    
    lgraph = layerGraph(net);
    Architecture_1 = figure('Units','normalized','Position',[0.1 0.1 0.8 0.8]);
    plot(lgraph);
    
       
    lgraph = removeLayers(lgraph, {'fc1000', 'prob', 'ClassificationLayer_predictions'});
    
    numClasses = numel(categories(trainingImages.Labels));
    newLayers = [
        fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',1,'BiasLearnRateFactor', 5)
        softmaxLayer('Name','softmax')
        classificationLayer('Name','classoutput')];
    lgraph = addLayers(lgraph,newLayers);
    
    lgraph = connectLayers(lgraph,'pool5','fc');
    
    Architecture_2 = figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
    plot(lgraph);
    ylim([0,10]);
    
    miniBatchSize = 10;
    
    %numIterationsPerEpoch = floor(numel(trainingImages.Labels)/miniBatchSize);
    options = trainingOptions('sgdm',...
        'ExecutionEnvironment','gpu',...
        'MiniBatchSize',miniBatchSize,...
        'MaxEpochs',100,...
        'InitialLearnRate',1e-5,...
        'VerboseFrequency',1,...
        'Plots','training-progress',...
        'ValidationData',validationImages,...
        'ValidationFrequency',5);
    netTransfer = trainNetwork(trainingImages,lgraph,options);
    
    %% Classify the validation images using the fine-tuned network.
    
    [predictedLabels, scores] = classify(netTransfer,validationImages,'ExecutionEnvironment','cpu');
    accuracy = mean(predictedLabels == validationImages.Labels);
    targets = grp2idx(validationImages.Labels);
    targets(targets==2) = 0;
    targets = ~targets;
    outputs = grp2idx(predictedLabels);
    outputs(outputs==2) = 0;
    outputs = ~outputs;
    [c,cm,ind,per] = confusion(targets', outputs');
    
    Confusion_Matrix=figure; plotconfusion(targets', outputs');
    
    %%
    
    addpath(genpath(resultsFolder))
    cp = my_classperf(cm);
    ROC_1=figure; plotroc(targets', scores(:,2)');
    
    [X,Y,T,AUC,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(targets', scores(:,2)',1,...
        'NBoot',1000,'TVals',0:0.05:1);
    ROC_2=figure; errorbar(X(:,3),Y(:,1),Y(:,1)-Y(:,2),Y(:,3)-Y(:,1));
    TheTableNice = table(validationImages.Labels,predictedLabels,'VariableNames',{'Actual','Predicted'});
    Confusion_Matrix1 = figure;
    PrettyCM = heatmap(TheTableNice,'Predicted','Actual');
    
    %     idx = randperm(20,4);
    %     Second_Figure = figure
    %     for i = 1:numel(idx)
    %         subplot(2,2,i)
    %         I = readimage(validationImages,idx(i));
    %         label = predictedLabels(idx(i));
    %         imshow(I)
    %         title(char(label))
    %     end
    
    try
        msg=sprintf('%s loop %d from %d MiniBatchSize %d with the results %d', Name, qq, nLoops, miniBatchSize, cp);
        sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
    catch
    end
    cd(resultsFolder);
    if  cp.Se > 0.79
        
        saveas(PrettyCM, ['BalCl_MB10_PrettyCM_' num2str(qq) '.jpg']);
        
        saveas(ROC_1, ['BalCl_MB10_ROC_1_' num2str(qq) '.jpg']);
        saveas(ROC_2, ['BalCl_MB10_ROC_2_' num2str(qq) '.jpg']);
        
        TrainingNeuNetw = findall(groot, 'Type', 'Figure');
        
        save(['RN_BalCl_MinBat10_' num2str(qq) '.mat'],'accuracy', 'AUC',...
            'c', 'cm', 'Confusion_Matrix', 'cp', ...
            'ind', 'miniBatchSize', 'netTransfer',...
            'numClasses', 'OPTROCPT', 'outputs', 'per', 'predictedLabels',...
            'scores','SUBY', 'SUBYNAMES', 'T', 'targets',...
            'TrainingNeuNetw','TheTableNice', 'validationImages', 'X', 'Y');
        %saveas(Architecture_1, ['UnCl_MB10_Architecture_1_' num2str(qq) '.jpg']);
        %saveas(Architecture_2, ['UnCl_MB10_Architecture_2_' num2str(qq) '.jpg']);
        
        %saveas(Second_Figure, ['UnCl_MB10_2nd_Figure_' num2str(qq) '.jpg']);
        
        
        
        try
            msg=sprintf('%s loop %d from %d MiniBatchSize %d , Sensitivity = %d , Balanced Accuracy = %d',...
                Name, qq, nLoops, miniBatchSize, cp.Se, cp.BA);
            sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
        catch
        end
        
%         close all
        % Save the figure of Training Progress in PNG format
        % saveas(TrainingNeuNetw,['RN_UnCl_MinBat10_' num2str(qq) '.png']);
        
        
%     else
%         save(['RN_BalCl_MinBat10_' num2str(qq) '.mat'],'accuracy', 'AUC',...
%             'cm', 'Confusion_Matrix', 'cp',...
%             'outputs', 'per', 'scores','SUBY', 'SUBYNAMES', 'T', 'targets',...
%             'TheTableNice', 'validationImages', 'X', 'Y');
%         saveas(PrettyCM, ['BalCl_MB10_PrettyCM_' num2str(qq) '.jpg']);
%         
%         try
%             msg=sprintf('%s loop %d from %d MiniBatchSize %d , Another one bits the dust, Sensitivity = %d',...
%                 Name, qq, nLoops, miniBatchSize, cp.Se);
%             sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
%         catch
%         end
        % Close all the figures
        
    end
    
    close all
    delete(findall(0));
    cd(codeFolder);
    % clearvars -except codeFolder resultsFolder TrainingFolder ValidationFolder nLoops qq
    
end

% Close progress bar
ioi_text_waitbar('Clear');

%%
nLoops = 1110;

for qq = 1101:nLoops
    
      
    ioi_text_waitbar(qq/nLoops, sprintf('ResNet101 Balanced MiniBatch=10, Running loop %d from %d', qq, 1440))
    
    close all
    trainingImages = imageDatastore(TrainingFolder, ...
        'IncludeSubfolders',true,...
        'FileExtensions','.tif','LabelSource','foldernames');
    
    trainingImages.ReadFcn = @(loc)imresize(imread(loc),[224, 224]);
    
    validationImages = imageDatastore(ValidationFolder, ...
        'IncludeSubfolders',true,...
        'FileExtensions','.tif','LabelSource','foldernames');
    
    validationImages.ReadFcn = @(loc)imresize(imread(loc),[224, 224]);
    
    
    % Load the pretrained GoogleNet network
    
    net = resnet101;
    net.Layers;
    
    % Extract the layer graph from the trained network and plot the layer graph.
    
    lgraph = layerGraph(net);
    Architecture_1 = figure('Units','normalized','Position',[0.1 0.1 0.8 0.8]);
    plot(lgraph);
    
       
    lgraph = removeLayers(lgraph, {'fc1000', 'prob', 'ClassificationLayer_predictions'});
    
    numClasses = numel(categories(trainingImages.Labels));
    newLayers = [
        fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',1,'BiasLearnRateFactor', 10)
        softmaxLayer('Name','softmax')
        classificationLayer('Name','classoutput')];
    lgraph = addLayers(lgraph,newLayers);
    
    lgraph = connectLayers(lgraph,'pool5','fc');
    
    Architecture_2 = figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
    plot(lgraph);
    ylim([0,10]);
    
    miniBatchSize = 10;
    
    %numIterationsPerEpoch = floor(numel(trainingImages.Labels)/miniBatchSize);
    options = trainingOptions('sgdm',...
        'ExecutionEnvironment','gpu',...
        'MiniBatchSize',miniBatchSize,...
        'MaxEpochs',100,...
        'InitialLearnRate',1e-5,...
        'VerboseFrequency',1,...
        'Plots','training-progress',...
        'ValidationData',validationImages,...
        'ValidationFrequency',5);
    netTransfer = trainNetwork(trainingImages,lgraph,options);
    
    %% Classify the validation images using the fine-tuned network.
    
    [predictedLabels, scores] = classify(netTransfer,validationImages,'ExecutionEnvironment','cpu');
    accuracy = mean(predictedLabels == validationImages.Labels);
    targets = grp2idx(validationImages.Labels);
    targets(targets==2) = 0;
    targets = ~targets;
    outputs = grp2idx(predictedLabels);
    outputs(outputs==2) = 0;
    outputs = ~outputs;
    [c,cm,ind,per] = confusion(targets', outputs');
    
    Confusion_Matrix=figure; plotconfusion(targets', outputs');
    
    %%
    
    addpath(genpath(resultsFolder))
    cp = my_classperf(cm);
    ROC_1=figure; plotroc(targets', scores(:,2)');
    
    [X,Y,T,AUC,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(targets', scores(:,2)',1,...
        'NBoot',1000,'TVals',0:0.05:1);
    ROC_2=figure; errorbar(X(:,3),Y(:,1),Y(:,1)-Y(:,2),Y(:,3)-Y(:,1));
    TheTableNice = table(validationImages.Labels,predictedLabels,'VariableNames',{'Actual','Predicted'});
    Confusion_Matrix1 = figure;
    PrettyCM = heatmap(TheTableNice,'Predicted','Actual');
    
    %     idx = randperm(20,4);
    %     Second_Figure = figure
    %     for i = 1:numel(idx)
    %         subplot(2,2,i)
    %         I = readimage(validationImages,idx(i));
    %         label = predictedLabels(idx(i));
    %         imshow(I)
    %         title(char(label))
    %     end
    
    try
        msg=sprintf('%s loop %d from %d MiniBatchSize %d with the results %d', Name, qq, nLoops, miniBatchSize, cp);
        sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
    catch
    end
    cd(resultsFolder);
    if  cp.Se > 0.79
        
        saveas(PrettyCM, ['BalCl_MB10_PrettyCM_' num2str(qq) '.jpg']);
        
        saveas(ROC_1, ['BalCl_MB10_ROC_1_' num2str(qq) '.jpg']);
        saveas(ROC_2, ['BalCl_MB10_ROC_2_' num2str(qq) '.jpg']);
        
        TrainingNeuNetw = findall(groot, 'Type', 'Figure');
        
        save(['RN_BalCl_MinBat10_' num2str(qq) '.mat'],'accuracy', 'AUC',...
            'c', 'cm', 'Confusion_Matrix', 'cp', ...
            'ind', 'miniBatchSize', 'netTransfer',...
            'numClasses', 'OPTROCPT', 'outputs', 'per', 'predictedLabels',...
            'scores','SUBY', 'SUBYNAMES', 'T', 'targets',...
            'TrainingNeuNetw','TheTableNice', 'validationImages', 'X', 'Y');
        %saveas(Architecture_1, ['UnCl_MB10_Architecture_1_' num2str(qq) '.jpg']);
        %saveas(Architecture_2, ['UnCl_MB10_Architecture_2_' num2str(qq) '.jpg']);
        
        %saveas(Second_Figure, ['UnCl_MB10_2nd_Figure_' num2str(qq) '.jpg']);
        
        
        
        try
            msg=sprintf('%s loop %d from %d MiniBatchSize %d , Sensitivity = %d , Balanced Accuracy = %d',...
                Name, qq, nLoops, miniBatchSize, cp.Se, cp.BA);
            sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
        catch
        end
        
%         close all
        % Save the figure of Training Progress in PNG format
        % saveas(TrainingNeuNetw,['RN_UnCl_MinBat10_' num2str(qq) '.png']);
        
        
%     else
%         save(['RN_BalCl_MinBat10_' num2str(qq) '.mat'],'accuracy', 'AUC',...
%             'cm', 'Confusion_Matrix', 'cp',...
%             'outputs', 'per', 'scores','SUBY', 'SUBYNAMES', 'T', 'targets',...
%             'TheTableNice', 'validationImages', 'X', 'Y');
%         saveas(PrettyCM, ['BalCl_MB10_PrettyCM_' num2str(qq) '.jpg']);
%         
%         try
%             msg=sprintf('%s loop %d from %d MiniBatchSize %d , Another one bits the dust, Sensitivity = %d',...
%                 Name, qq, nLoops, miniBatchSize, cp.Se);
%             sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
%         catch
%         end
        % Close all the figures
        
    end
    
    close all
    delete(findall(0));
    cd(codeFolder);
    % clearvars -except codeFolder resultsFolder TrainingFolder ValidationFolder nLoops qq
    
end

% Close progress bar
ioi_text_waitbar('Clear');

%%
nLoops = 1120;

for qq = 1111:nLoops
    
      
    ioi_text_waitbar(qq/nLoops, sprintf('ResNet101 Balanced MiniBatch=10, Running loop %d from %d', qq, 1440))
    
    close all
    trainingImages = imageDatastore(TrainingFolder, ...
        'IncludeSubfolders',true,...
        'FileExtensions','.tif','LabelSource','foldernames');
    
    trainingImages.ReadFcn = @(loc)imresize(imread(loc),[224, 224]);
    
    validationImages = imageDatastore(ValidationFolder, ...
        'IncludeSubfolders',true,...
        'FileExtensions','.tif','LabelSource','foldernames');
    
    validationImages.ReadFcn = @(loc)imresize(imread(loc),[224, 224]);
    
    
    % Load the pretrained GoogleNet network
    
    net = resnet101;
    net.Layers;
    
    % Extract the layer graph from the trained network and plot the layer graph.
    
    lgraph = layerGraph(net);
    Architecture_1 = figure('Units','normalized','Position',[0.1 0.1 0.8 0.8]);
    plot(lgraph);
    
       
    lgraph = removeLayers(lgraph, {'fc1000', 'prob', 'ClassificationLayer_predictions'});
    
    numClasses = numel(categories(trainingImages.Labels));
    newLayers = [
        fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',1,'BiasLearnRateFactor', 20)
        softmaxLayer('Name','softmax')
        classificationLayer('Name','classoutput')];
    lgraph = addLayers(lgraph,newLayers);
    
    lgraph = connectLayers(lgraph,'pool5','fc');
    
    Architecture_2 = figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
    plot(lgraph);
    ylim([0,10]);
    
    miniBatchSize = 10;
    
    %numIterationsPerEpoch = floor(numel(trainingImages.Labels)/miniBatchSize);
    options = trainingOptions('sgdm',...
        'ExecutionEnvironment','gpu',...
        'MiniBatchSize',miniBatchSize,...
        'MaxEpochs',100,...
        'InitialLearnRate',1e-5,...
        'VerboseFrequency',1,...
        'Plots','training-progress',...
        'ValidationData',validationImages,...
        'ValidationFrequency',5);
    netTransfer = trainNetwork(trainingImages,lgraph,options);
    
    %% Classify the validation images using the fine-tuned network.
    
    [predictedLabels, scores] = classify(netTransfer,validationImages,'ExecutionEnvironment','cpu');
    accuracy = mean(predictedLabels == validationImages.Labels);
    targets = grp2idx(validationImages.Labels);
    targets(targets==2) = 0;
    targets = ~targets;
    outputs = grp2idx(predictedLabels);
    outputs(outputs==2) = 0;
    outputs = ~outputs;
    [c,cm,ind,per] = confusion(targets', outputs');
    
    Confusion_Matrix=figure; plotconfusion(targets', outputs');
    
    %%
    
    addpath(genpath(resultsFolder))
    cp = my_classperf(cm);
    ROC_1=figure; plotroc(targets', scores(:,2)');
    
    [X,Y,T,AUC,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(targets', scores(:,2)',1,...
        'NBoot',1000,'TVals',0:0.05:1);
    ROC_2=figure; errorbar(X(:,3),Y(:,1),Y(:,1)-Y(:,2),Y(:,3)-Y(:,1));
    TheTableNice = table(validationImages.Labels,predictedLabels,'VariableNames',{'Actual','Predicted'});
    Confusion_Matrix1 = figure;
    PrettyCM = heatmap(TheTableNice,'Predicted','Actual');
    
    %     idx = randperm(20,4);
    %     Second_Figure = figure
    %     for i = 1:numel(idx)
    %         subplot(2,2,i)
    %         I = readimage(validationImages,idx(i));
    %         label = predictedLabels(idx(i));
    %         imshow(I)
    %         title(char(label))
    %     end
    
    try
        msg=sprintf('%s loop %d from %d MiniBatchSize %d with the results %d', Name, qq, nLoops, miniBatchSize, cp);
        sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
    catch
    end
    cd(resultsFolder);
    if  cp.Se > 0.79
        
        saveas(PrettyCM, ['BalCl_MB10_PrettyCM_' num2str(qq) '.jpg']);
        
        saveas(ROC_1, ['BalCl_MB10_ROC_1_' num2str(qq) '.jpg']);
        saveas(ROC_2, ['BalCl_MB10_ROC_2_' num2str(qq) '.jpg']);
        
        TrainingNeuNetw = findall(groot, 'Type', 'Figure');
        
        save(['RN_BalCl_MinBat10_' num2str(qq) '.mat'],'accuracy', 'AUC',...
            'c', 'cm', 'Confusion_Matrix', 'cp', ...
            'ind', 'miniBatchSize', 'netTransfer',...
            'numClasses', 'OPTROCPT', 'outputs', 'per', 'predictedLabels',...
            'scores','SUBY', 'SUBYNAMES', 'T', 'targets',...
            'TrainingNeuNetw','TheTableNice', 'validationImages', 'X', 'Y');
        %saveas(Architecture_1, ['UnCl_MB10_Architecture_1_' num2str(qq) '.jpg']);
        %saveas(Architecture_2, ['UnCl_MB10_Architecture_2_' num2str(qq) '.jpg']);
        
        %saveas(Second_Figure, ['UnCl_MB10_2nd_Figure_' num2str(qq) '.jpg']);
        
        
        
        try
            msg=sprintf('%s loop %d from %d MiniBatchSize %d , Sensitivity = %d , Balanced Accuracy = %d',...
                Name, qq, nLoops, miniBatchSize, cp.Se, cp.BA);
            sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
        catch
        end
        
%         close all
        % Save the figure of Training Progress in PNG format
        % saveas(TrainingNeuNetw,['RN_UnCl_MinBat10_' num2str(qq) '.png']);
        
        
%     else
%         save(['RN_BalCl_MinBat10_' num2str(qq) '.mat'],'accuracy', 'AUC',...
%             'cm', 'Confusion_Matrix', 'cp',...
%             'outputs', 'per', 'scores','SUBY', 'SUBYNAMES', 'T', 'targets',...
%             'TheTableNice', 'validationImages', 'X', 'Y');
%         saveas(PrettyCM, ['BalCl_MB10_PrettyCM_' num2str(qq) '.jpg']);
%         
%         try
%             msg=sprintf('%s loop %d from %d MiniBatchSize %d , Another one bits the dust, Sensitivity = %d',...
%                 Name, qq, nLoops, miniBatchSize, cp.Se);
%             sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
%         catch
%         end
        % Close all the figures
        
    end
    
    close all
    delete(findall(0));
    cd(codeFolder);
    % clearvars -except codeFolder resultsFolder TrainingFolder ValidationFolder nLoops qq
    
end

% Close progress bar
ioi_text_waitbar('Clear');

%%
nLoops = 1130;

for qq = 1121:nLoops
    
      
    ioi_text_waitbar(qq/nLoops, sprintf('ResNet101 Balanced MiniBatch=10, Running loop %d from %d', qq, 1440))
    
    close all
    trainingImages = imageDatastore(TrainingFolder, ...
        'IncludeSubfolders',true,...
        'FileExtensions','.tif','LabelSource','foldernames');
    
    trainingImages.ReadFcn = @(loc)imresize(imread(loc),[224, 224]);
    
    validationImages = imageDatastore(ValidationFolder, ...
        'IncludeSubfolders',true,...
        'FileExtensions','.tif','LabelSource','foldernames');
    
    validationImages.ReadFcn = @(loc)imresize(imread(loc),[224, 224]);
    
    
    % Load the pretrained GoogleNet network
    
    net = resnet101;
    net.Layers;
    
    % Extract the layer graph from the trained network and plot the layer graph.
    
    lgraph = layerGraph(net);
    Architecture_1 = figure('Units','normalized','Position',[0.1 0.1 0.8 0.8]);
    plot(lgraph);
    
       
    lgraph = removeLayers(lgraph, {'fc1000', 'prob', 'ClassificationLayer_predictions'});
    
    numClasses = numel(categories(trainingImages.Labels));
    newLayers = [
        fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',1,'BiasLearnRateFactor', 40)
        softmaxLayer('Name','softmax')
        classificationLayer('Name','classoutput')];
    lgraph = addLayers(lgraph,newLayers);
    
    lgraph = connectLayers(lgraph,'pool5','fc');
    
    Architecture_2 = figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
    plot(lgraph);
    ylim([0,10]);
    
    miniBatchSize = 10;
    
    %numIterationsPerEpoch = floor(numel(trainingImages.Labels)/miniBatchSize);
    options = trainingOptions('sgdm',...
        'ExecutionEnvironment','gpu',...
        'MiniBatchSize',miniBatchSize,...
        'MaxEpochs',100,...
        'InitialLearnRate',1e-5,...
        'VerboseFrequency',1,...
        'Plots','training-progress',...
        'ValidationData',validationImages,...
        'ValidationFrequency',5);
    netTransfer = trainNetwork(trainingImages,lgraph,options);
    
    %% Classify the validation images using the fine-tuned network.
    
    [predictedLabels, scores] = classify(netTransfer,validationImages,'ExecutionEnvironment','cpu');
    accuracy = mean(predictedLabels == validationImages.Labels);
    targets = grp2idx(validationImages.Labels);
    targets(targets==2) = 0;
    targets = ~targets;
    outputs = grp2idx(predictedLabels);
    outputs(outputs==2) = 0;
    outputs = ~outputs;
    [c,cm,ind,per] = confusion(targets', outputs');
    
    Confusion_Matrix=figure; plotconfusion(targets', outputs');
    
    %%
    
    addpath(genpath(resultsFolder))
    cp = my_classperf(cm);
    ROC_1=figure; plotroc(targets', scores(:,2)');
    
    [X,Y,T,AUC,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(targets', scores(:,2)',1,...
        'NBoot',1000,'TVals',0:0.05:1);
    ROC_2=figure; errorbar(X(:,3),Y(:,1),Y(:,1)-Y(:,2),Y(:,3)-Y(:,1));
    TheTableNice = table(validationImages.Labels,predictedLabels,'VariableNames',{'Actual','Predicted'});
    Confusion_Matrix1 = figure;
    PrettyCM = heatmap(TheTableNice,'Predicted','Actual');
    
    %     idx = randperm(20,4);
    %     Second_Figure = figure
    %     for i = 1:numel(idx)
    %         subplot(2,2,i)
    %         I = readimage(validationImages,idx(i));
    %         label = predictedLabels(idx(i));
    %         imshow(I)
    %         title(char(label))
    %     end
    
    try
        msg=sprintf('%s loop %d from %d MiniBatchSize %d with the results %d', Name, qq, nLoops, miniBatchSize, cp);
        sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
    catch
    end
    cd(resultsFolder);
    if  cp.Se > 0.79
        
        saveas(PrettyCM, ['BalCl_MB10_PrettyCM_' num2str(qq) '.jpg']);
        
        saveas(ROC_1, ['BalCl_MB10_ROC_1_' num2str(qq) '.jpg']);
        saveas(ROC_2, ['BalCl_MB10_ROC_2_' num2str(qq) '.jpg']);
        
        TrainingNeuNetw = findall(groot, 'Type', 'Figure');
        
        save(['RN_BalCl_MinBat10_' num2str(qq) '.mat'],'accuracy', 'AUC',...
            'c', 'cm', 'Confusion_Matrix', 'cp', ...
            'ind', 'miniBatchSize', 'netTransfer',...
            'numClasses', 'OPTROCPT', 'outputs', 'per', 'predictedLabels',...
            'scores','SUBY', 'SUBYNAMES', 'T', 'targets',...
            'TrainingNeuNetw','TheTableNice', 'validationImages', 'X', 'Y');
        %saveas(Architecture_1, ['UnCl_MB10_Architecture_1_' num2str(qq) '.jpg']);
        %saveas(Architecture_2, ['UnCl_MB10_Architecture_2_' num2str(qq) '.jpg']);
        
        %saveas(Second_Figure, ['UnCl_MB10_2nd_Figure_' num2str(qq) '.jpg']);
        
        
        
        try
            msg=sprintf('%s loop %d from %d MiniBatchSize %d , Sensitivity = %d , Balanced Accuracy = %d',...
                Name, qq, nLoops, miniBatchSize, cp.Se, cp.BA);
            sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
        catch
        end
        
%         close all
        % Save the figure of Training Progress in PNG format
        % saveas(TrainingNeuNetw,['RN_UnCl_MinBat10_' num2str(qq) '.png']);
        
        
%     else
%         save(['RN_BalCl_MinBat10_' num2str(qq) '.mat'],'accuracy', 'AUC',...
%             'cm', 'Confusion_Matrix', 'cp',...
%             'outputs', 'per', 'scores','SUBY', 'SUBYNAMES', 'T', 'targets',...
%             'TheTableNice', 'validationImages', 'X', 'Y');
%         saveas(PrettyCM, ['BalCl_MB10_PrettyCM_' num2str(qq) '.jpg']);
%         
%         try
%             msg=sprintf('%s loop %d from %d MiniBatchSize %d , Another one bits the dust, Sensitivity = %d',...
%                 Name, qq, nLoops, miniBatchSize, cp.Se);
%             sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
%         catch
%         end
        % Close all the figures
        
    end
    
    close all
    delete(findall(0));
    cd(codeFolder);
    % clearvars -except codeFolder resultsFolder TrainingFolder ValidationFolder nLoops qq
    
end

% Close progress bar
ioi_text_waitbar('Clear');


nLoops = 1140;

for qq = 1131:nLoops
    
      
    ioi_text_waitbar(qq/nLoops, sprintf('ResNet101 Balanced MiniBatch=10, Running loop %d from %d', qq, 1440))
    
    close all
    trainingImages = imageDatastore(TrainingFolder, ...
        'IncludeSubfolders',true,...
        'FileExtensions','.tif','LabelSource','foldernames');
    
    trainingImages.ReadFcn = @(loc)imresize(imread(loc),[224, 224]);
    
    validationImages = imageDatastore(ValidationFolder, ...
        'IncludeSubfolders',true,...
        'FileExtensions','.tif','LabelSource','foldernames');
    
    validationImages.ReadFcn = @(loc)imresize(imread(loc),[224, 224]);
    
    
    % Load the pretrained GoogleNet network
    
    net = resnet101;
    net.Layers;
    
    % Extract the layer graph from the trained network and plot the layer graph.
    
    lgraph = layerGraph(net);
    Architecture_1 = figure('Units','normalized','Position',[0.1 0.1 0.8 0.8]);
    plot(lgraph);
    
       
    lgraph = removeLayers(lgraph, {'fc1000', 'prob', 'ClassificationLayer_predictions'});
    
    numClasses = numel(categories(trainingImages.Labels));
    newLayers = [
        fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',1,'BiasLearnRateFactor', 60)
        softmaxLayer('Name','softmax')
        classificationLayer('Name','classoutput')];
    lgraph = addLayers(lgraph,newLayers);
    
    lgraph = connectLayers(lgraph,'pool5','fc');
    
    Architecture_2 = figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
    plot(lgraph);
    ylim([0,10]);
    
    miniBatchSize = 10;
    
    %numIterationsPerEpoch = floor(numel(trainingImages.Labels)/miniBatchSize);
    options = trainingOptions('sgdm',...
        'ExecutionEnvironment','gpu',...
        'MiniBatchSize',miniBatchSize,...
        'MaxEpochs',100,...
        'InitialLearnRate',1e-5,...
        'VerboseFrequency',1,...
        'Plots','training-progress',...
        'ValidationData',validationImages,...
        'ValidationFrequency',5);
    netTransfer = trainNetwork(trainingImages,lgraph,options);
    
    %% Classify the validation images using the fine-tuned network.
    
    [predictedLabels, scores] = classify(netTransfer,validationImages,'ExecutionEnvironment','cpu');
    accuracy = mean(predictedLabels == validationImages.Labels);
    targets = grp2idx(validationImages.Labels);
    targets(targets==2) = 0;
    targets = ~targets;
    outputs = grp2idx(predictedLabels);
    outputs(outputs==2) = 0;
    outputs = ~outputs;
    [c,cm,ind,per] = confusion(targets', outputs');
    
    Confusion_Matrix=figure; plotconfusion(targets', outputs');
    
    %%
    
    addpath(genpath(resultsFolder))
    cp = my_classperf(cm);
    ROC_1=figure; plotroc(targets', scores(:,2)');
    
    [X,Y,T,AUC,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(targets', scores(:,2)',1,...
        'NBoot',1000,'TVals',0:0.05:1);
    ROC_2=figure; errorbar(X(:,3),Y(:,1),Y(:,1)-Y(:,2),Y(:,3)-Y(:,1));
    TheTableNice = table(validationImages.Labels,predictedLabels,'VariableNames',{'Actual','Predicted'});
    Confusion_Matrix1 = figure;
    PrettyCM = heatmap(TheTableNice,'Predicted','Actual');
    
    %     idx = randperm(20,4);
    %     Second_Figure = figure
    %     for i = 1:numel(idx)
    %         subplot(2,2,i)
    %         I = readimage(validationImages,idx(i));
    %         label = predictedLabels(idx(i));
    %         imshow(I)
    %         title(char(label))
    %     end
    
    try
        msg=sprintf('%s loop %d from %d MiniBatchSize %d with the results %d', Name, qq, nLoops, miniBatchSize, cp);
        sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
    catch
    end
    cd(resultsFolder);
    if  cp.Se > 0.79
        
        saveas(PrettyCM, ['BalCl_MB10_PrettyCM_' num2str(qq) '.jpg']);
        
        saveas(ROC_1, ['BalCl_MB10_ROC_1_' num2str(qq) '.jpg']);
        saveas(ROC_2, ['BalCl_MB10_ROC_2_' num2str(qq) '.jpg']);
        
        TrainingNeuNetw = findall(groot, 'Type', 'Figure');
        
        save(['RN_BalCl_MinBat10_' num2str(qq) '.mat'],'accuracy', 'AUC',...
            'c', 'cm', 'Confusion_Matrix', 'cp', ...
            'ind', 'miniBatchSize', 'netTransfer',...
            'numClasses', 'OPTROCPT', 'outputs', 'per', 'predictedLabels',...
            'scores','SUBY', 'SUBYNAMES', 'T', 'targets',...
            'TrainingNeuNetw','TheTableNice', 'validationImages', 'X', 'Y');
        %saveas(Architecture_1, ['UnCl_MB10_Architecture_1_' num2str(qq) '.jpg']);
        %saveas(Architecture_2, ['UnCl_MB10_Architecture_2_' num2str(qq) '.jpg']);
        
        %saveas(Second_Figure, ['UnCl_MB10_2nd_Figure_' num2str(qq) '.jpg']);
        
        
        
        try
            msg=sprintf('%s loop %d from %d MiniBatchSize %d , Sensitivity = %d , Balanced Accuracy = %d',...
                Name, qq, nLoops, miniBatchSize, cp.Se, cp.BA);
            sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
        catch
        end
        
%         close all
        % Save the figure of Training Progress in PNG format
        % saveas(TrainingNeuNetw,['RN_UnCl_MinBat10_' num2str(qq) '.png']);
        
        
%     else
%         save(['RN_BalCl_MinBat10_' num2str(qq) '.mat'],'accuracy', 'AUC',...
%             'cm', 'Confusion_Matrix', 'cp',...
%             'outputs', 'per', 'scores','SUBY', 'SUBYNAMES', 'T', 'targets',...
%             'TheTableNice', 'validationImages', 'X', 'Y');
%         saveas(PrettyCM, ['BalCl_MB10_PrettyCM_' num2str(qq) '.jpg']);
%         
%         try
%             msg=sprintf('%s loop %d from %d MiniBatchSize %d , Another one bits the dust, Sensitivity = %d',...
%                 Name, qq, nLoops, miniBatchSize, cp.Se);
%             sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
%         catch
%         end
        % Close all the figures
        
    end
    
    close all
    delete(findall(0));
    cd(codeFolder);
    % clearvars -except codeFolder resultsFolder TrainingFolder ValidationFolder nLoops qq
    
end

% Close progress bar
ioi_text_waitbar('Clear');

% *ResNet 101 for Database of Brasil, IJC & ISSSTE*

% Unbalanced Class
% MiniBatch = 10

% Analysis of three databases is realized for compare the results, only the
% results that has a Sensitivity bigger than 80% is saved

clear; close all; clc;

% Folders Location
codeFolder = 'C:\Users\LANCYTT\Documents\Juan Carlos Torres\Thermograms Analysis';
% addpath('I:\Archivos Juan Carlos Torres\MATLAB\MATLAB Codigos');

resultsFolder = 'C:\Users\LANCYTT\Documents\Juan Carlos Torres\Results\ResNet101\Iteraciones\Balanced';
addpath('C:\Users\LANCYTT\Documents\Juan Carlos Torres\Thermograms Analysis');

TrainingFolder = 'C:\Users\LANCYTT\Documents\Juan Carlos Torres\Data Augmented Thermography\Data Augmented Bra+IJC+ISSSTE\Balanced Augmented Modify';
ValidationFolder = 'C:\Users\LANCYTT\Documents\Juan Carlos Torres\Data Augmented Thermography\Train - Valid Bra+IJC+ISSSTE\Validation Balanced Class';

pat_sendmail

ioi_text_waitbar(0, 'Running ResNet 101, Balanced MiniBatch 10... Please wait...');

Name=sprintf('ResNet101 Balanced');
nLoops = 1150;

for qq = 1141:nLoops
    
      
    ioi_text_waitbar(qq/nLoops, sprintf('ResNet101 Balanced MiniBatch=10, Running loop %d from %d', qq, 1440))
    
    close all
    trainingImages = imageDatastore(TrainingFolder, ...
        'IncludeSubfolders',true,...
        'FileExtensions','.tif','LabelSource','foldernames');
    
    trainingImages.ReadFcn = @(loc)imresize(imread(loc),[224, 224]);
    
    validationImages = imageDatastore(ValidationFolder, ...
        'IncludeSubfolders',true,...
        'FileExtensions','.tif','LabelSource','foldernames');
    
    validationImages.ReadFcn = @(loc)imresize(imread(loc),[224, 224]);
    
    
    % Load the pretrained GoogleNet network
    
    net = resnet101;
    net.Layers;
    
    % Extract the layer graph from the trained network and plot the layer graph.
    
    lgraph = layerGraph(net);
    Architecture_1 = figure('Units','normalized','Position',[0.1 0.1 0.8 0.8]);
    plot(lgraph);
    
       
    lgraph = removeLayers(lgraph, {'fc1000', 'prob', 'ClassificationLayer_predictions'});
    
    numClasses = numel(categories(trainingImages.Labels));
    newLayers = [
        fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',5,'BiasLearnRateFactor', 1)
        softmaxLayer('Name','softmax')
        classificationLayer('Name','classoutput')];
    lgraph = addLayers(lgraph,newLayers);
    
    lgraph = connectLayers(lgraph,'pool5','fc');
    
    Architecture_2 = figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
    plot(lgraph);
    ylim([0,10]);
    
    miniBatchSize = 10;
    
    %numIterationsPerEpoch = floor(numel(trainingImages.Labels)/miniBatchSize);
    options = trainingOptions('sgdm',...
        'ExecutionEnvironment','gpu',...
        'MiniBatchSize',miniBatchSize,...
        'MaxEpochs',100,...
        'InitialLearnRate',1e-5,...
        'VerboseFrequency',1,...
        'Plots','training-progress',...
        'ValidationData',validationImages,...
        'ValidationFrequency',5);
    netTransfer = trainNetwork(trainingImages,lgraph,options);
    
    %% Classify the validation images using the fine-tuned network.
    
    [predictedLabels, scores] = classify(netTransfer,validationImages,'ExecutionEnvironment','cpu');
    accuracy = mean(predictedLabels == validationImages.Labels);
    targets = grp2idx(validationImages.Labels);
    targets(targets==2) = 0;
    targets = ~targets;
    outputs = grp2idx(predictedLabels);
    outputs(outputs==2) = 0;
    outputs = ~outputs;
    [c,cm,ind,per] = confusion(targets', outputs');
    
    Confusion_Matrix=figure; plotconfusion(targets', outputs');
    
    %%
    
    addpath(genpath(resultsFolder))
    cp = my_classperf(cm);
    ROC_1=figure; plotroc(targets', scores(:,2)');
    
    [X,Y,T,AUC,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(targets', scores(:,2)',1,...
        'NBoot',1000,'TVals',0:0.05:1);
    ROC_2=figure; errorbar(X(:,3),Y(:,1),Y(:,1)-Y(:,2),Y(:,3)-Y(:,1));
    TheTableNice = table(validationImages.Labels,predictedLabels,'VariableNames',{'Actual','Predicted'});
    Confusion_Matrix1 = figure;
    PrettyCM = heatmap(TheTableNice,'Predicted','Actual');
    
    %     idx = randperm(20,4);
    %     Second_Figure = figure
    %     for i = 1:numel(idx)
    %         subplot(2,2,i)
    %         I = readimage(validationImages,idx(i));
    %         label = predictedLabels(idx(i));
    %         imshow(I)
    %         title(char(label))
    %     end
    
    try
        msg=sprintf('%s loop %d from %d MiniBatchSize %d with the results %d', Name, qq, nLoops, miniBatchSize, cp);
        sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
    catch
    end
    cd(resultsFolder);
    if  cp.Se > 0.79
        
        saveas(PrettyCM, ['BalCl_MB10_PrettyCM_' num2str(qq) '.jpg']);
        
        saveas(ROC_1, ['BalCl_MB10_ROC_1_' num2str(qq) '.jpg']);
        saveas(ROC_2, ['BalCl_MB10_ROC_2_' num2str(qq) '.jpg']);
        
        TrainingNeuNetw = findall(groot, 'Type', 'Figure');
        
        save(['RN_BalCl_MinBat10_' num2str(qq) '.mat'],'accuracy', 'AUC',...
            'c', 'cm', 'Confusion_Matrix', 'cp', ...
            'ind', 'miniBatchSize', 'netTransfer',...
            'numClasses', 'OPTROCPT', 'outputs', 'per', 'predictedLabels',...
            'scores','SUBY', 'SUBYNAMES', 'T', 'targets',...
            'TrainingNeuNetw','TheTableNice', 'validationImages', 'X', 'Y');
        %saveas(Architecture_1, ['UnCl_MB10_Architecture_1_' num2str(qq) '.jpg']);
        %saveas(Architecture_2, ['UnCl_MB10_Architecture_2_' num2str(qq) '.jpg']);
        
        %saveas(Second_Figure, ['UnCl_MB10_2nd_Figure_' num2str(qq) '.jpg']);
        
        
        
        try
            msg=sprintf('%s loop %d from %d MiniBatchSize %d , Sensitivity = %d , Balanced Accuracy = %d',...
                Name, qq, nLoops, miniBatchSize, cp.Se, cp.BA);
            sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
        catch
        end
        
%         close all
        % Save the figure of Training Progress in PNG format
        % saveas(TrainingNeuNetw,['RN_UnCl_MinBat10_' num2str(qq) '.png']);
        
        
%     else
%         save(['RN_BalCl_MinBat10_' num2str(qq) '.mat'],'accuracy', 'AUC',...
%             'cm', 'Confusion_Matrix', 'cp',...
%             'outputs', 'per', 'scores','SUBY', 'SUBYNAMES', 'T', 'targets',...
%             'TheTableNice', 'validationImages', 'X', 'Y');
%         saveas(PrettyCM, ['BalCl_MB10_PrettyCM_' num2str(qq) '.jpg']);
%         
%         try
%             msg=sprintf('%s loop %d from %d MiniBatchSize %d , Another one bits the dust, Sensitivity = %d',...
%                 Name, qq, nLoops, miniBatchSize, cp.Se);
%             sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
%         catch
%         end
        % Close all the figures
        
    end
    
    close all
    delete(findall(0));
    cd(codeFolder);
    % clearvars -except codeFolder resultsFolder TrainingFolder ValidationFolder nLoops qq
    
end

% Close progress bar
ioi_text_waitbar('Clear');

nLoops = 1160;

for qq = 1151:nLoops
    
      
    ioi_text_waitbar(qq/nLoops, sprintf('ResNet101 Balanced MiniBatch=10, Running loop %d from %d', qq, 1440))
    
    close all
    trainingImages = imageDatastore(TrainingFolder, ...
        'IncludeSubfolders',true,...
        'FileExtensions','.tif','LabelSource','foldernames');
    
    trainingImages.ReadFcn = @(loc)imresize(imread(loc),[224, 224]);
    
    validationImages = imageDatastore(ValidationFolder, ...
        'IncludeSubfolders',true,...
        'FileExtensions','.tif','LabelSource','foldernames');
    
    validationImages.ReadFcn = @(loc)imresize(imread(loc),[224, 224]);
    
    
    % Load the pretrained GoogleNet network
    
    net = resnet101;
    net.Layers;
    
    % Extract the layer graph from the trained network and plot the layer graph.
    
    lgraph = layerGraph(net);
    Architecture_1 = figure('Units','normalized','Position',[0.1 0.1 0.8 0.8]);
    plot(lgraph);
    
       
    lgraph = removeLayers(lgraph, {'fc1000', 'prob', 'ClassificationLayer_predictions'});
    
    numClasses = numel(categories(trainingImages.Labels));
    newLayers = [
        fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',5,'BiasLearnRateFactor', 5)
        softmaxLayer('Name','softmax')
        classificationLayer('Name','classoutput')];
    lgraph = addLayers(lgraph,newLayers);
    
    lgraph = connectLayers(lgraph,'pool5','fc');
    
    Architecture_2 = figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
    plot(lgraph);
    ylim([0,10]);
    
    miniBatchSize = 10;
    
    %numIterationsPerEpoch = floor(numel(trainingImages.Labels)/miniBatchSize);
    options = trainingOptions('sgdm',...
        'ExecutionEnvironment','gpu',...
        'MiniBatchSize',miniBatchSize,...
        'MaxEpochs',100,...
        'InitialLearnRate',1e-5,...
        'VerboseFrequency',1,...
        'Plots','training-progress',...
        'ValidationData',validationImages,...
        'ValidationFrequency',5);
    netTransfer = trainNetwork(trainingImages,lgraph,options);
    
    %% Classify the validation images using the fine-tuned network.
    
    [predictedLabels, scores] = classify(netTransfer,validationImages,'ExecutionEnvironment','cpu');
    accuracy = mean(predictedLabels == validationImages.Labels);
    targets = grp2idx(validationImages.Labels);
    targets(targets==2) = 0;
    targets = ~targets;
    outputs = grp2idx(predictedLabels);
    outputs(outputs==2) = 0;
    outputs = ~outputs;
    [c,cm,ind,per] = confusion(targets', outputs');
    
    Confusion_Matrix=figure; plotconfusion(targets', outputs');
    
    %%
    
    addpath(genpath(resultsFolder))
    cp = my_classperf(cm);
    ROC_1=figure; plotroc(targets', scores(:,2)');
    
    [X,Y,T,AUC,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(targets', scores(:,2)',1,...
        'NBoot',1000,'TVals',0:0.05:1);
    ROC_2=figure; errorbar(X(:,3),Y(:,1),Y(:,1)-Y(:,2),Y(:,3)-Y(:,1));
    TheTableNice = table(validationImages.Labels,predictedLabels,'VariableNames',{'Actual','Predicted'});
    Confusion_Matrix1 = figure;
    PrettyCM = heatmap(TheTableNice,'Predicted','Actual');
    
    %     idx = randperm(20,4);
    %     Second_Figure = figure
    %     for i = 1:numel(idx)
    %         subplot(2,2,i)
    %         I = readimage(validationImages,idx(i));
    %         label = predictedLabels(idx(i));
    %         imshow(I)
    %         title(char(label))
    %     end
    
    try
        msg=sprintf('%s loop %d from %d MiniBatchSize %d with the results %d', Name, qq, nLoops, miniBatchSize, cp);
        sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
    catch
    end
    cd(resultsFolder);
    if  cp.Se > 0.79
        
        saveas(PrettyCM, ['BalCl_MB10_PrettyCM_' num2str(qq) '.jpg']);
        
        saveas(ROC_1, ['BalCl_MB10_ROC_1_' num2str(qq) '.jpg']);
        saveas(ROC_2, ['BalCl_MB10_ROC_2_' num2str(qq) '.jpg']);
        
        TrainingNeuNetw = findall(groot, 'Type', 'Figure');
        
        save(['RN_BalCl_MinBat10_' num2str(qq) '.mat'],'accuracy', 'AUC',...
            'c', 'cm', 'Confusion_Matrix', 'cp', ...
            'ind', 'miniBatchSize', 'netTransfer',...
            'numClasses', 'OPTROCPT', 'outputs', 'per', 'predictedLabels',...
            'scores','SUBY', 'SUBYNAMES', 'T', 'targets',...
            'TrainingNeuNetw','TheTableNice', 'validationImages', 'X', 'Y');
        %saveas(Architecture_1, ['UnCl_MB10_Architecture_1_' num2str(qq) '.jpg']);
        %saveas(Architecture_2, ['UnCl_MB10_Architecture_2_' num2str(qq) '.jpg']);
        
        %saveas(Second_Figure, ['UnCl_MB10_2nd_Figure_' num2str(qq) '.jpg']);
        
        
        
        try
            msg=sprintf('%s loop %d from %d MiniBatchSize %d , Sensitivity = %d , Balanced Accuracy = %d',...
                Name, qq, nLoops, miniBatchSize, cp.Se, cp.BA);
            sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
        catch
        end
        
%         close all
        % Save the figure of Training Progress in PNG format
        % saveas(TrainingNeuNetw,['RN_UnCl_MinBat10_' num2str(qq) '.png']);
        
        
%     else
%         save(['RN_BalCl_MinBat10_' num2str(qq) '.mat'],'accuracy', 'AUC',...
%             'cm', 'Confusion_Matrix', 'cp',...
%             'outputs', 'per', 'scores','SUBY', 'SUBYNAMES', 'T', 'targets',...
%             'TheTableNice', 'validationImages', 'X', 'Y');
%         saveas(PrettyCM, ['BalCl_MB10_PrettyCM_' num2str(qq) '.jpg']);
%         
%         try
%             msg=sprintf('%s loop %d from %d MiniBatchSize %d , Another one bits the dust, Sensitivity = %d',...
%                 Name, qq, nLoops, miniBatchSize, cp.Se);
%             sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
%         catch
%         end
        % Close all the figures
        
    end
    
    close all
    delete(findall(0));
    cd(codeFolder);
    % clearvars -except codeFolder resultsFolder TrainingFolder ValidationFolder nLoops qq
    
end

% Close progress bar
ioi_text_waitbar('Clear');

%%
nLoops = 1170;

for qq = 1161:nLoops
    
      
    ioi_text_waitbar(qq/nLoops, sprintf('ResNet101 Balanced MiniBatch=10, Running loop %d from %d', qq, 1440))
    
    close all
    trainingImages = imageDatastore(TrainingFolder, ...
        'IncludeSubfolders',true,...
        'FileExtensions','.tif','LabelSource','foldernames');
    
    trainingImages.ReadFcn = @(loc)imresize(imread(loc),[224, 224]);
    
    validationImages = imageDatastore(ValidationFolder, ...
        'IncludeSubfolders',true,...
        'FileExtensions','.tif','LabelSource','foldernames');
    
    validationImages.ReadFcn = @(loc)imresize(imread(loc),[224, 224]);
    
    
    % Load the pretrained GoogleNet network
    
    net = resnet101;
    net.Layers;
    
    % Extract the layer graph from the trained network and plot the layer graph.
    
    lgraph = layerGraph(net);
    Architecture_1 = figure('Units','normalized','Position',[0.1 0.1 0.8 0.8]);
    plot(lgraph);
    
       
    lgraph = removeLayers(lgraph, {'fc1000', 'prob', 'ClassificationLayer_predictions'});
    
    numClasses = numel(categories(trainingImages.Labels));
    newLayers = [
        fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',5,'BiasLearnRateFactor', 10)
        softmaxLayer('Name','softmax')
        classificationLayer('Name','classoutput')];
    lgraph = addLayers(lgraph,newLayers);
    
    lgraph = connectLayers(lgraph,'pool5','fc');
    
    Architecture_2 = figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
    plot(lgraph);
    ylim([0,10]);
    
    miniBatchSize = 10;
    
    %numIterationsPerEpoch = floor(numel(trainingImages.Labels)/miniBatchSize);
    options = trainingOptions('sgdm',...
        'ExecutionEnvironment','gpu',...
        'MiniBatchSize',miniBatchSize,...
        'MaxEpochs',100,...
        'InitialLearnRate',1e-5,...
        'VerboseFrequency',1,...
        'Plots','training-progress',...
        'ValidationData',validationImages,...
        'ValidationFrequency',5);
    netTransfer = trainNetwork(trainingImages,lgraph,options);
    
    %% Classify the validation images using the fine-tuned network.
    
    [predictedLabels, scores] = classify(netTransfer,validationImages,'ExecutionEnvironment','cpu');
    accuracy = mean(predictedLabels == validationImages.Labels);
    targets = grp2idx(validationImages.Labels);
    targets(targets==2) = 0;
    targets = ~targets;
    outputs = grp2idx(predictedLabels);
    outputs(outputs==2) = 0;
    outputs = ~outputs;
    [c,cm,ind,per] = confusion(targets', outputs');
    
    Confusion_Matrix=figure; plotconfusion(targets', outputs');
    
    %%
    
    addpath(genpath(resultsFolder))
    cp = my_classperf(cm);
    ROC_1=figure; plotroc(targets', scores(:,2)');
    
    [X,Y,T,AUC,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(targets', scores(:,2)',1,...
        'NBoot',1000,'TVals',0:0.05:1);
    ROC_2=figure; errorbar(X(:,3),Y(:,1),Y(:,1)-Y(:,2),Y(:,3)-Y(:,1));
    TheTableNice = table(validationImages.Labels,predictedLabels,'VariableNames',{'Actual','Predicted'});
    Confusion_Matrix1 = figure;
    PrettyCM = heatmap(TheTableNice,'Predicted','Actual');
    
    %     idx = randperm(20,4);
    %     Second_Figure = figure
    %     for i = 1:numel(idx)
    %         subplot(2,2,i)
    %         I = readimage(validationImages,idx(i));
    %         label = predictedLabels(idx(i));
    %         imshow(I)
    %         title(char(label))
    %     end
    
    try
        msg=sprintf('%s loop %d from %d MiniBatchSize %d with the results %d', Name, qq, nLoops, miniBatchSize, cp);
        sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
    catch
    end
    cd(resultsFolder);
    if  cp.Se > 0.79
        
        saveas(PrettyCM, ['BalCl_MB10_PrettyCM_' num2str(qq) '.jpg']);
        
        saveas(ROC_1, ['BalCl_MB10_ROC_1_' num2str(qq) '.jpg']);
        saveas(ROC_2, ['BalCl_MB10_ROC_2_' num2str(qq) '.jpg']);
        
        TrainingNeuNetw = findall(groot, 'Type', 'Figure');
        
        save(['RN_BalCl_MinBat10_' num2str(qq) '.mat'],'accuracy', 'AUC',...
            'c', 'cm', 'Confusion_Matrix', 'cp', ...
            'ind', 'miniBatchSize', 'netTransfer',...
            'numClasses', 'OPTROCPT', 'outputs', 'per', 'predictedLabels',...
            'scores','SUBY', 'SUBYNAMES', 'T', 'targets',...
            'TrainingNeuNetw','TheTableNice', 'validationImages', 'X', 'Y');
        %saveas(Architecture_1, ['UnCl_MB10_Architecture_1_' num2str(qq) '.jpg']);
        %saveas(Architecture_2, ['UnCl_MB10_Architecture_2_' num2str(qq) '.jpg']);
        
        %saveas(Second_Figure, ['UnCl_MB10_2nd_Figure_' num2str(qq) '.jpg']);
        
        
        
        try
            msg=sprintf('%s loop %d from %d MiniBatchSize %d , Sensitivity = %d , Balanced Accuracy = %d',...
                Name, qq, nLoops, miniBatchSize, cp.Se, cp.BA);
            sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
        catch
        end
        
%         close all
        % Save the figure of Training Progress in PNG format
        % saveas(TrainingNeuNetw,['RN_UnCl_MinBat10_' num2str(qq) '.png']);
        
        
%     else
%         save(['RN_BalCl_MinBat10_' num2str(qq) '.mat'],'accuracy', 'AUC',...
%             'cm', 'Confusion_Matrix', 'cp',...
%             'outputs', 'per', 'scores','SUBY', 'SUBYNAMES', 'T', 'targets',...
%             'TheTableNice', 'validationImages', 'X', 'Y');
%         saveas(PrettyCM, ['BalCl_MB10_PrettyCM_' num2str(qq) '.jpg']);
%         
%         try
%             msg=sprintf('%s loop %d from %d MiniBatchSize %d , Another one bits the dust, Sensitivity = %d',...
%                 Name, qq, nLoops, miniBatchSize, cp.Se);
%             sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
%         catch
%         end
        % Close all the figures
        
    end
    
    close all
    delete(findall(0));
    cd(codeFolder);
    % clearvars -except codeFolder resultsFolder TrainingFolder ValidationFolder nLoops qq
    
end

% Close progress bar
ioi_text_waitbar('Clear');

%%
nLoops = 1180;

for qq = 1171:nLoops
    
      
    ioi_text_waitbar(qq/nLoops, sprintf('ResNet101 Balanced MiniBatch=10, Running loop %d from %d', qq, 1440))
    
    close all
    trainingImages = imageDatastore(TrainingFolder, ...
        'IncludeSubfolders',true,...
        'FileExtensions','.tif','LabelSource','foldernames');
    
    trainingImages.ReadFcn = @(loc)imresize(imread(loc),[224, 224]);
    
    validationImages = imageDatastore(ValidationFolder, ...
        'IncludeSubfolders',true,...
        'FileExtensions','.tif','LabelSource','foldernames');
    
    validationImages.ReadFcn = @(loc)imresize(imread(loc),[224, 224]);
    
    
    % Load the pretrained GoogleNet network
    
    net = resnet101;
    net.Layers;
    
    % Extract the layer graph from the trained network and plot the layer graph.
    
    lgraph = layerGraph(net);
    Architecture_1 = figure('Units','normalized','Position',[0.1 0.1 0.8 0.8]);
    plot(lgraph);
    
       
    lgraph = removeLayers(lgraph, {'fc1000', 'prob', 'ClassificationLayer_predictions'});
    
    numClasses = numel(categories(trainingImages.Labels));
    newLayers = [
        fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',5,'BiasLearnRateFactor', 20)
        softmaxLayer('Name','softmax')
        classificationLayer('Name','classoutput')];
    lgraph = addLayers(lgraph,newLayers);
    
    lgraph = connectLayers(lgraph,'pool5','fc');
    
    Architecture_2 = figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
    plot(lgraph);
    ylim([0,10]);
    
    miniBatchSize = 10;
    
    %numIterationsPerEpoch = floor(numel(trainingImages.Labels)/miniBatchSize);
    options = trainingOptions('sgdm',...
        'ExecutionEnvironment','gpu',...
        'MiniBatchSize',miniBatchSize,...
        'MaxEpochs',100,...
        'InitialLearnRate',1e-5,...
        'VerboseFrequency',1,...
        'Plots','training-progress',...
        'ValidationData',validationImages,...
        'ValidationFrequency',5);
    netTransfer = trainNetwork(trainingImages,lgraph,options);
    
    %% Classify the validation images using the fine-tuned network.
    
    [predictedLabels, scores] = classify(netTransfer,validationImages,'ExecutionEnvironment','cpu');
    accuracy = mean(predictedLabels == validationImages.Labels);
    targets = grp2idx(validationImages.Labels);
    targets(targets==2) = 0;
    targets = ~targets;
    outputs = grp2idx(predictedLabels);
    outputs(outputs==2) = 0;
    outputs = ~outputs;
    [c,cm,ind,per] = confusion(targets', outputs');
    
    Confusion_Matrix=figure; plotconfusion(targets', outputs');
    
    %%
    
    addpath(genpath(resultsFolder))
    cp = my_classperf(cm);
    ROC_1=figure; plotroc(targets', scores(:,2)');
    
    [X,Y,T,AUC,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(targets', scores(:,2)',1,...
        'NBoot',1000,'TVals',0:0.05:1);
    ROC_2=figure; errorbar(X(:,3),Y(:,1),Y(:,1)-Y(:,2),Y(:,3)-Y(:,1));
    TheTableNice = table(validationImages.Labels,predictedLabels,'VariableNames',{'Actual','Predicted'});
    Confusion_Matrix1 = figure;
    PrettyCM = heatmap(TheTableNice,'Predicted','Actual');
    
    %     idx = randperm(20,4);
    %     Second_Figure = figure
    %     for i = 1:numel(idx)
    %         subplot(2,2,i)
    %         I = readimage(validationImages,idx(i));
    %         label = predictedLabels(idx(i));
    %         imshow(I)
    %         title(char(label))
    %     end
    
    try
        msg=sprintf('%s loop %d from %d MiniBatchSize %d with the results %d', Name, qq, nLoops, miniBatchSize, cp);
        sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
    catch
    end
    cd(resultsFolder);
    if  cp.Se > 0.79
        
        saveas(PrettyCM, ['BalCl_MB10_PrettyCM_' num2str(qq) '.jpg']);
        
        saveas(ROC_1, ['BalCl_MB10_ROC_1_' num2str(qq) '.jpg']);
        saveas(ROC_2, ['BalCl_MB10_ROC_2_' num2str(qq) '.jpg']);
        
        TrainingNeuNetw = findall(groot, 'Type', 'Figure');
        
        save(['RN_BalCl_MinBat10_' num2str(qq) '.mat'],'accuracy', 'AUC',...
            'c', 'cm', 'Confusion_Matrix', 'cp', ...
            'ind', 'miniBatchSize', 'netTransfer',...
            'numClasses', 'OPTROCPT', 'outputs', 'per', 'predictedLabels',...
            'scores','SUBY', 'SUBYNAMES', 'T', 'targets',...
            'TrainingNeuNetw','TheTableNice', 'validationImages', 'X', 'Y');
        %saveas(Architecture_1, ['UnCl_MB10_Architecture_1_' num2str(qq) '.jpg']);
        %saveas(Architecture_2, ['UnCl_MB10_Architecture_2_' num2str(qq) '.jpg']);
        
        %saveas(Second_Figure, ['UnCl_MB10_2nd_Figure_' num2str(qq) '.jpg']);
        
        
        
        try
            msg=sprintf('%s loop %d from %d MiniBatchSize %d , Sensitivity = %d , Balanced Accuracy = %d',...
                Name, qq, nLoops, miniBatchSize, cp.Se, cp.BA);
            sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
        catch
        end
        
%         close all
        % Save the figure of Training Progress in PNG format
        % saveas(TrainingNeuNetw,['RN_UnCl_MinBat10_' num2str(qq) '.png']);
        
        
%     else
%         save(['RN_BalCl_MinBat10_' num2str(qq) '.mat'],'accuracy', 'AUC',...
%             'cm', 'Confusion_Matrix', 'cp',...
%             'outputs', 'per', 'scores','SUBY', 'SUBYNAMES', 'T', 'targets',...
%             'TheTableNice', 'validationImages', 'X', 'Y');
%         saveas(PrettyCM, ['BalCl_MB10_PrettyCM_' num2str(qq) '.jpg']);
%         
%         try
%             msg=sprintf('%s loop %d from %d MiniBatchSize %d , Another one bits the dust, Sensitivity = %d',...
%                 Name, qq, nLoops, miniBatchSize, cp.Se);
%             sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
%         catch
%         end
        % Close all the figures
        
    end
    
    close all
    delete(findall(0));
    cd(codeFolder);
    % clearvars -except codeFolder resultsFolder TrainingFolder ValidationFolder nLoops qq
    
end

% Close progress bar
ioi_text_waitbar('Clear');

%%
nLoops = 1190;

for qq = 1181:nLoops
    
      
    ioi_text_waitbar(qq/nLoops, sprintf('ResNet101 Balanced MiniBatch=10, Running loop %d from %d', qq, 1440))
    
    close all
    trainingImages = imageDatastore(TrainingFolder, ...
        'IncludeSubfolders',true,...
        'FileExtensions','.tif','LabelSource','foldernames');
    
    trainingImages.ReadFcn = @(loc)imresize(imread(loc),[224, 224]);
    
    validationImages = imageDatastore(ValidationFolder, ...
        'IncludeSubfolders',true,...
        'FileExtensions','.tif','LabelSource','foldernames');
    
    validationImages.ReadFcn = @(loc)imresize(imread(loc),[224, 224]);
    
    
    % Load the pretrained GoogleNet network
    
    net = resnet101;
    net.Layers;
    
    % Extract the layer graph from the trained network and plot the layer graph.
    
    lgraph = layerGraph(net);
    Architecture_1 = figure('Units','normalized','Position',[0.1 0.1 0.8 0.8]);
    plot(lgraph);
    
       
    lgraph = removeLayers(lgraph, {'fc1000', 'prob', 'ClassificationLayer_predictions'});
    
    numClasses = numel(categories(trainingImages.Labels));
    newLayers = [
        fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',5,'BiasLearnRateFactor', 40)
        softmaxLayer('Name','softmax')
        classificationLayer('Name','classoutput')];
    lgraph = addLayers(lgraph,newLayers);
    
    lgraph = connectLayers(lgraph,'pool5','fc');
    
    Architecture_2 = figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
    plot(lgraph);
    ylim([0,10]);
    
    miniBatchSize = 10;
    
    %numIterationsPerEpoch = floor(numel(trainingImages.Labels)/miniBatchSize);
    options = trainingOptions('sgdm',...
        'ExecutionEnvironment','gpu',...
        'MiniBatchSize',miniBatchSize,...
        'MaxEpochs',100,...
        'InitialLearnRate',1e-5,...
        'VerboseFrequency',1,...
        'Plots','training-progress',...
        'ValidationData',validationImages,...
        'ValidationFrequency',5);
    netTransfer = trainNetwork(trainingImages,lgraph,options);
    
    %% Classify the validation images using the fine-tuned network.
    
    [predictedLabels, scores] = classify(netTransfer,validationImages,'ExecutionEnvironment','cpu');
    accuracy = mean(predictedLabels == validationImages.Labels);
    targets = grp2idx(validationImages.Labels);
    targets(targets==2) = 0;
    targets = ~targets;
    outputs = grp2idx(predictedLabels);
    outputs(outputs==2) = 0;
    outputs = ~outputs;
    [c,cm,ind,per] = confusion(targets', outputs');
    
    Confusion_Matrix=figure; plotconfusion(targets', outputs');
    
    %%
    
    addpath(genpath(resultsFolder))
    cp = my_classperf(cm);
    ROC_1=figure; plotroc(targets', scores(:,2)');
    
    [X,Y,T,AUC,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(targets', scores(:,2)',1,...
        'NBoot',1000,'TVals',0:0.05:1);
    ROC_2=figure; errorbar(X(:,3),Y(:,1),Y(:,1)-Y(:,2),Y(:,3)-Y(:,1));
    TheTableNice = table(validationImages.Labels,predictedLabels,'VariableNames',{'Actual','Predicted'});
    Confusion_Matrix1 = figure;
    PrettyCM = heatmap(TheTableNice,'Predicted','Actual');
    
    %     idx = randperm(20,4);
    %     Second_Figure = figure
    %     for i = 1:numel(idx)
    %         subplot(2,2,i)
    %         I = readimage(validationImages,idx(i));
    %         label = predictedLabels(idx(i));
    %         imshow(I)
    %         title(char(label))
    %     end
    
    try
        msg=sprintf('%s loop %d from %d MiniBatchSize %d with the results %d', Name, qq, nLoops, miniBatchSize, cp);
        sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
    catch
    end
    cd(resultsFolder);
    if  cp.Se > 0.79
        
        saveas(PrettyCM, ['BalCl_MB10_PrettyCM_' num2str(qq) '.jpg']);
        
        saveas(ROC_1, ['BalCl_MB10_ROC_1_' num2str(qq) '.jpg']);
        saveas(ROC_2, ['BalCl_MB10_ROC_2_' num2str(qq) '.jpg']);
        
        TrainingNeuNetw = findall(groot, 'Type', 'Figure');
        
        save(['RN_BalCl_MinBat10_' num2str(qq) '.mat'],'accuracy', 'AUC',...
            'c', 'cm', 'Confusion_Matrix', 'cp', ...
            'ind', 'miniBatchSize', 'netTransfer',...
            'numClasses', 'OPTROCPT', 'outputs', 'per', 'predictedLabels',...
            'scores','SUBY', 'SUBYNAMES', 'T', 'targets',...
            'TrainingNeuNetw','TheTableNice', 'validationImages', 'X', 'Y');
        %saveas(Architecture_1, ['UnCl_MB10_Architecture_1_' num2str(qq) '.jpg']);
        %saveas(Architecture_2, ['UnCl_MB10_Architecture_2_' num2str(qq) '.jpg']);
        
        %saveas(Second_Figure, ['UnCl_MB10_2nd_Figure_' num2str(qq) '.jpg']);
        
        
        
        try
            msg=sprintf('%s loop %d from %d MiniBatchSize %d , Sensitivity = %d , Balanced Accuracy = %d',...
                Name, qq, nLoops, miniBatchSize, cp.Se, cp.BA);
            sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
        catch
        end
        
%         close all
        % Save the figure of Training Progress in PNG format
        % saveas(TrainingNeuNetw,['RN_UnCl_MinBat10_' num2str(qq) '.png']);
        
        
%     else
%         save(['RN_BalCl_MinBat10_' num2str(qq) '.mat'],'accuracy', 'AUC',...
%             'cm', 'Confusion_Matrix', 'cp',...
%             'outputs', 'per', 'scores','SUBY', 'SUBYNAMES', 'T', 'targets',...
%             'TheTableNice', 'validationImages', 'X', 'Y');
%         saveas(PrettyCM, ['BalCl_MB10_PrettyCM_' num2str(qq) '.jpg']);
%         
%         try
%             msg=sprintf('%s loop %d from %d MiniBatchSize %d , Another one bits the dust, Sensitivity = %d',...
%                 Name, qq, nLoops, miniBatchSize, cp.Se);
%             sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
%         catch
%         end
        % Close all the figures
        
    end
    
    close all
    delete(findall(0));
    cd(codeFolder);
    % clearvars -except codeFolder resultsFolder TrainingFolder ValidationFolder nLoops qq
    
end

% Close progress bar
ioi_text_waitbar('Clear');


nLoops = 1200;

for qq = 1191:nLoops
    
      
    ioi_text_waitbar(qq/nLoops, sprintf('ResNet101 Balanced MiniBatch=10, Running loop %d from %d', qq, 1440))
    
    close all
    trainingImages = imageDatastore(TrainingFolder, ...
        'IncludeSubfolders',true,...
        'FileExtensions','.tif','LabelSource','foldernames');
    
    trainingImages.ReadFcn = @(loc)imresize(imread(loc),[224, 224]);
    
    validationImages = imageDatastore(ValidationFolder, ...
        'IncludeSubfolders',true,...
        'FileExtensions','.tif','LabelSource','foldernames');
    
    validationImages.ReadFcn = @(loc)imresize(imread(loc),[224, 224]);
    
    
    % Load the pretrained GoogleNet network
    
    net = resnet101;
    net.Layers;
    
    % Extract the layer graph from the trained network and plot the layer graph.
    
    lgraph = layerGraph(net);
    Architecture_1 = figure('Units','normalized','Position',[0.1 0.1 0.8 0.8]);
    plot(lgraph);
    
       
    lgraph = removeLayers(lgraph, {'fc1000', 'prob', 'ClassificationLayer_predictions'});
    
    numClasses = numel(categories(trainingImages.Labels));
    newLayers = [
        fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',5,'BiasLearnRateFactor', 60)
        softmaxLayer('Name','softmax')
        classificationLayer('Name','classoutput')];
    lgraph = addLayers(lgraph,newLayers);
    
    lgraph = connectLayers(lgraph,'pool5','fc');
    
    Architecture_2 = figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
    plot(lgraph);
    ylim([0,10]);
    
    miniBatchSize = 10;
    
    %numIterationsPerEpoch = floor(numel(trainingImages.Labels)/miniBatchSize);
    options = trainingOptions('sgdm',...
        'ExecutionEnvironment','gpu',...
        'MiniBatchSize',miniBatchSize,...
        'MaxEpochs',100,...
        'InitialLearnRate',1e-5,...
        'VerboseFrequency',1,...
        'Plots','training-progress',...
        'ValidationData',validationImages,...
        'ValidationFrequency',5);
    netTransfer = trainNetwork(trainingImages,lgraph,options);
    
    %% Classify the validation images using the fine-tuned network.
    
    [predictedLabels, scores] = classify(netTransfer,validationImages,'ExecutionEnvironment','cpu');
    accuracy = mean(predictedLabels == validationImages.Labels);
    targets = grp2idx(validationImages.Labels);
    targets(targets==2) = 0;
    targets = ~targets;
    outputs = grp2idx(predictedLabels);
    outputs(outputs==2) = 0;
    outputs = ~outputs;
    [c,cm,ind,per] = confusion(targets', outputs');
    
    Confusion_Matrix=figure; plotconfusion(targets', outputs');
    
    %%
    
    addpath(genpath(resultsFolder))
    cp = my_classperf(cm);
    ROC_1=figure; plotroc(targets', scores(:,2)');
    
    [X,Y,T,AUC,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(targets', scores(:,2)',1,...
        'NBoot',1000,'TVals',0:0.05:1);
    ROC_2=figure; errorbar(X(:,3),Y(:,1),Y(:,1)-Y(:,2),Y(:,3)-Y(:,1));
    TheTableNice = table(validationImages.Labels,predictedLabels,'VariableNames',{'Actual','Predicted'});
    Confusion_Matrix1 = figure;
    PrettyCM = heatmap(TheTableNice,'Predicted','Actual');
    
    %     idx = randperm(20,4);
    %     Second_Figure = figure
    %     for i = 1:numel(idx)
    %         subplot(2,2,i)
    %         I = readimage(validationImages,idx(i));
    %         label = predictedLabels(idx(i));
    %         imshow(I)
    %         title(char(label))
    %     end
    
    try
        msg=sprintf('%s loop %d from %d MiniBatchSize %d with the results %d', Name, qq, nLoops, miniBatchSize, cp);
        sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
    catch
    end
    cd(resultsFolder);
    if  cp.Se > 0.79
        
        saveas(PrettyCM, ['BalCl_MB10_PrettyCM_' num2str(qq) '.jpg']);
        
        saveas(ROC_1, ['BalCl_MB10_ROC_1_' num2str(qq) '.jpg']);
        saveas(ROC_2, ['BalCl_MB10_ROC_2_' num2str(qq) '.jpg']);
        
        TrainingNeuNetw = findall(groot, 'Type', 'Figure');
        
        save(['RN_BalCl_MinBat10_' num2str(qq) '.mat'],'accuracy', 'AUC',...
            'c', 'cm', 'Confusion_Matrix', 'cp', ...
            'ind', 'miniBatchSize', 'netTransfer',...
            'numClasses', 'OPTROCPT', 'outputs', 'per', 'predictedLabels',...
            'scores','SUBY', 'SUBYNAMES', 'T', 'targets',...
            'TrainingNeuNetw','TheTableNice', 'validationImages', 'X', 'Y');
        %saveas(Architecture_1, ['UnCl_MB10_Architecture_1_' num2str(qq) '.jpg']);
        %saveas(Architecture_2, ['UnCl_MB10_Architecture_2_' num2str(qq) '.jpg']);
        
        %saveas(Second_Figure, ['UnCl_MB10_2nd_Figure_' num2str(qq) '.jpg']);
        
        
        
        try
            msg=sprintf('%s loop %d from %d MiniBatchSize %d , Sensitivity = %d , Balanced Accuracy = %d',...
                Name, qq, nLoops, miniBatchSize, cp.Se, cp.BA);
            sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
        catch
        end
        
%         close all
        % Save the figure of Training Progress in PNG format
        % saveas(TrainingNeuNetw,['RN_UnCl_MinBat10_' num2str(qq) '.png']);
        
        
%     else
%         save(['RN_BalCl_MinBat10_' num2str(qq) '.mat'],'accuracy', 'AUC',...
%             'cm', 'Confusion_Matrix', 'cp',...
%             'outputs', 'per', 'scores','SUBY', 'SUBYNAMES', 'T', 'targets',...
%             'TheTableNice', 'validationImages', 'X', 'Y');
%         saveas(PrettyCM, ['BalCl_MB10_PrettyCM_' num2str(qq) '.jpg']);
%         
%         try
%             msg=sprintf('%s loop %d from %d MiniBatchSize %d , Another one bits the dust, Sensitivity = %d',...
%                 Name, qq, nLoops, miniBatchSize, cp.Se);
%             sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
%         catch
%         end
        % Close all the figures
        
    end
    
    close all
    delete(findall(0));
    cd(codeFolder);
    % clearvars -except codeFolder resultsFolder TrainingFolder ValidationFolder nLoops qq
    
end

% Close progress bar
ioi_text_waitbar('Clear');

% *ResNet 101 for Database of Brasil, IJC & ISSSTE*

% Unbalanced Class
% MiniBatch = 10

% Analysis of three databases is realized for compare the results, only the
% results that has a Sensitivity bigger than 80% is saved

clear; close all; clc;

% Folders Location
codeFolder = 'C:\Users\LANCYTT\Documents\Juan Carlos Torres\Thermograms Analysis';
% addpath('I:\Archivos Juan Carlos Torres\MATLAB\MATLAB Codigos');

resultsFolder = 'C:\Users\LANCYTT\Documents\Juan Carlos Torres\Results\ResNet101\Iteraciones\Balanced';
addpath('C:\Users\LANCYTT\Documents\Juan Carlos Torres\Thermograms Analysis');

TrainingFolder = 'C:\Users\LANCYTT\Documents\Juan Carlos Torres\Data Augmented Thermography\Data Augmented Bra+IJC+ISSSTE\Balanced Augmented Modify';
ValidationFolder = 'C:\Users\LANCYTT\Documents\Juan Carlos Torres\Data Augmented Thermography\Train - Valid Bra+IJC+ISSSTE\Validation Balanced Class';

pat_sendmail

ioi_text_waitbar(0, 'Running ResNet 101, Balanced MiniBatch 10... Please wait...');

Name=sprintf('ResNet101 Balanced');
nLoops = 1210;

for qq = 1211:nLoops
    
      
    ioi_text_waitbar(qq/nLoops, sprintf('ResNet101 Balanced MiniBatch=10, Running loop %d from %d', qq, 1440))
    
    close all
    trainingImages = imageDatastore(TrainingFolder, ...
        'IncludeSubfolders',true,...
        'FileExtensions','.tif','LabelSource','foldernames');
    
    trainingImages.ReadFcn = @(loc)imresize(imread(loc),[224, 224]);
    
    validationImages = imageDatastore(ValidationFolder, ...
        'IncludeSubfolders',true,...
        'FileExtensions','.tif','LabelSource','foldernames');
    
    validationImages.ReadFcn = @(loc)imresize(imread(loc),[224, 224]);
    
    
    % Load the pretrained GoogleNet network
    
    net = resnet101;
    net.Layers;
    
    % Extract the layer graph from the trained network and plot the layer graph.
    
    lgraph = layerGraph(net);
    Architecture_1 = figure('Units','normalized','Position',[0.1 0.1 0.8 0.8]);
    plot(lgraph);
    
       
    lgraph = removeLayers(lgraph, {'fc1000', 'prob', 'ClassificationLayer_predictions'});
    
    numClasses = numel(categories(trainingImages.Labels));
    newLayers = [
        fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',10,'BiasLearnRateFactor', 1)
        softmaxLayer('Name','softmax')
        classificationLayer('Name','classoutput')];
    lgraph = addLayers(lgraph,newLayers);
    
    lgraph = connectLayers(lgraph,'pool5','fc');
    
    Architecture_2 = figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
    plot(lgraph);
    ylim([0,10]);
    
    miniBatchSize = 10;
    
    %numIterationsPerEpoch = floor(numel(trainingImages.Labels)/miniBatchSize);
    options = trainingOptions('sgdm',...
        'ExecutionEnvironment','gpu',...
        'MiniBatchSize',miniBatchSize,...
        'MaxEpochs',100,...
        'InitialLearnRate',1e-5,...
        'VerboseFrequency',1,...
        'Plots','training-progress',...
        'ValidationData',validationImages,...
        'ValidationFrequency',5);
    netTransfer = trainNetwork(trainingImages,lgraph,options);
    
    %% Classify the validation images using the fine-tuned network.
    
    [predictedLabels, scores] = classify(netTransfer,validationImages,'ExecutionEnvironment','cpu');
    accuracy = mean(predictedLabels == validationImages.Labels);
    targets = grp2idx(validationImages.Labels);
    targets(targets==2) = 0;
    targets = ~targets;
    outputs = grp2idx(predictedLabels);
    outputs(outputs==2) = 0;
    outputs = ~outputs;
    [c,cm,ind,per] = confusion(targets', outputs');
    
    Confusion_Matrix=figure; plotconfusion(targets', outputs');
    
    %%
    
    addpath(genpath(resultsFolder))
    cp = my_classperf(cm);
    ROC_1=figure; plotroc(targets', scores(:,2)');
    
    [X,Y,T,AUC,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(targets', scores(:,2)',1,...
        'NBoot',1000,'TVals',0:0.05:1);
    ROC_2=figure; errorbar(X(:,3),Y(:,1),Y(:,1)-Y(:,2),Y(:,3)-Y(:,1));
    TheTableNice = table(validationImages.Labels,predictedLabels,'VariableNames',{'Actual','Predicted'});
    Confusion_Matrix1 = figure;
    PrettyCM = heatmap(TheTableNice,'Predicted','Actual');
    
    %     idx = randperm(20,4);
    %     Second_Figure = figure
    %     for i = 1:numel(idx)
    %         subplot(2,2,i)
    %         I = readimage(validationImages,idx(i));
    %         label = predictedLabels(idx(i));
    %         imshow(I)
    %         title(char(label))
    %     end
    
    try
        msg=sprintf('%s loop %d from %d MiniBatchSize %d with the results %d', Name, qq, nLoops, miniBatchSize, cp);
        sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
    catch
    end
    cd(resultsFolder);
    if  cp.Se > 0.79
        
        saveas(PrettyCM, ['BalCl_MB10_PrettyCM_' num2str(qq) '.jpg']);
        
        saveas(ROC_1, ['BalCl_MB10_ROC_1_' num2str(qq) '.jpg']);
        saveas(ROC_2, ['BalCl_MB10_ROC_2_' num2str(qq) '.jpg']);
        
        TrainingNeuNetw = findall(groot, 'Type', 'Figure');
        
        save(['RN_BalCl_MinBat10_' num2str(qq) '.mat'],'accuracy', 'AUC',...
            'c', 'cm', 'Confusion_Matrix', 'cp', ...
            'ind', 'miniBatchSize', 'netTransfer',...
            'numClasses', 'OPTROCPT', 'outputs', 'per', 'predictedLabels',...
            'scores','SUBY', 'SUBYNAMES', 'T', 'targets',...
            'TrainingNeuNetw','TheTableNice', 'validationImages', 'X', 'Y');
        %saveas(Architecture_1, ['UnCl_MB10_Architecture_1_' num2str(qq) '.jpg']);
        %saveas(Architecture_2, ['UnCl_MB10_Architecture_2_' num2str(qq) '.jpg']);
        
        %saveas(Second_Figure, ['UnCl_MB10_2nd_Figure_' num2str(qq) '.jpg']);
        
        
        
        try
            msg=sprintf('%s loop %d from %d MiniBatchSize %d , Sensitivity = %d , Balanced Accuracy = %d',...
                Name, qq, nLoops, miniBatchSize, cp.Se, cp.BA);
            sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
        catch
        end
        
%         close all
        % Save the figure of Training Progress in PNG format
        % saveas(TrainingNeuNetw,['RN_UnCl_MinBat10_' num2str(qq) '.png']);
        
        
%     else
%         save(['RN_BalCl_MinBat10_' num2str(qq) '.mat'],'accuracy', 'AUC',...
%             'cm', 'Confusion_Matrix', 'cp',...
%             'outputs', 'per', 'scores','SUBY', 'SUBYNAMES', 'T', 'targets',...
%             'TheTableNice', 'validationImages', 'X', 'Y');
%         saveas(PrettyCM, ['BalCl_MB10_PrettyCM_' num2str(qq) '.jpg']);
%         
%         try
%             msg=sprintf('%s loop %d from %d MiniBatchSize %d , Another one bits the dust, Sensitivity = %d',...
%                 Name, qq, nLoops, miniBatchSize, cp.Se);
%             sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
%         catch
%         end
        % Close all the figures
        
    end
    
    close all
    delete(findall(0));
    cd(codeFolder);
    % clearvars -except codeFolder resultsFolder TrainingFolder ValidationFolder nLoops qq
    
end

% Close progress bar
ioi_text_waitbar('Clear');

nLoops = 1220;

for qq = 1211:nLoops
    
      
    ioi_text_waitbar(qq/nLoops, sprintf('ResNet101 Balanced MiniBatch=10, Running loop %d from %d', qq, 1440))
    
    close all
    trainingImages = imageDatastore(TrainingFolder, ...
        'IncludeSubfolders',true,...
        'FileExtensions','.tif','LabelSource','foldernames');
    
    trainingImages.ReadFcn = @(loc)imresize(imread(loc),[224, 224]);
    
    validationImages = imageDatastore(ValidationFolder, ...
        'IncludeSubfolders',true,...
        'FileExtensions','.tif','LabelSource','foldernames');
    
    validationImages.ReadFcn = @(loc)imresize(imread(loc),[224, 224]);
    
    
    % Load the pretrained GoogleNet network
    
    net = resnet101;
    net.Layers;
    
    % Extract the layer graph from the trained network and plot the layer graph.
    
    lgraph = layerGraph(net);
    Architecture_1 = figure('Units','normalized','Position',[0.1 0.1 0.8 0.8]);
    plot(lgraph);
    
       
    lgraph = removeLayers(lgraph, {'fc1000', 'prob', 'ClassificationLayer_predictions'});
    
    numClasses = numel(categories(trainingImages.Labels));
    newLayers = [
        fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',10,'BiasLearnRateFactor', 5)
        softmaxLayer('Name','softmax')
        classificationLayer('Name','classoutput')];
    lgraph = addLayers(lgraph,newLayers);
    
    lgraph = connectLayers(lgraph,'pool5','fc');
    
    Architecture_2 = figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
    plot(lgraph);
    ylim([0,10]);
    
    miniBatchSize = 10;
    
    %numIterationsPerEpoch = floor(numel(trainingImages.Labels)/miniBatchSize);
    options = trainingOptions('sgdm',...
        'ExecutionEnvironment','gpu',...
        'MiniBatchSize',miniBatchSize,...
        'MaxEpochs',100,...
        'InitialLearnRate',1e-5,...
        'VerboseFrequency',1,...
        'Plots','training-progress',...
        'ValidationData',validationImages,...
        'ValidationFrequency',5);
    netTransfer = trainNetwork(trainingImages,lgraph,options);
    
    %% Classify the validation images using the fine-tuned network.
    
    [predictedLabels, scores] = classify(netTransfer,validationImages,'ExecutionEnvironment','cpu');
    accuracy = mean(predictedLabels == validationImages.Labels);
    targets = grp2idx(validationImages.Labels);
    targets(targets==2) = 0;
    targets = ~targets;
    outputs = grp2idx(predictedLabels);
    outputs(outputs==2) = 0;
    outputs = ~outputs;
    [c,cm,ind,per] = confusion(targets', outputs');
    
    Confusion_Matrix=figure; plotconfusion(targets', outputs');
    
    %%
    
    addpath(genpath(resultsFolder))
    cp = my_classperf(cm);
    ROC_1=figure; plotroc(targets', scores(:,2)');
    
    [X,Y,T,AUC,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(targets', scores(:,2)',1,...
        'NBoot',1000,'TVals',0:0.05:1);
    ROC_2=figure; errorbar(X(:,3),Y(:,1),Y(:,1)-Y(:,2),Y(:,3)-Y(:,1));
    TheTableNice = table(validationImages.Labels,predictedLabels,'VariableNames',{'Actual','Predicted'});
    Confusion_Matrix1 = figure;
    PrettyCM = heatmap(TheTableNice,'Predicted','Actual');
    
    %     idx = randperm(20,4);
    %     Second_Figure = figure
    %     for i = 1:numel(idx)
    %         subplot(2,2,i)
    %         I = readimage(validationImages,idx(i));
    %         label = predictedLabels(idx(i));
    %         imshow(I)
    %         title(char(label))
    %     end
    
    try
        msg=sprintf('%s loop %d from %d MiniBatchSize %d with the results %d', Name, qq, nLoops, miniBatchSize, cp);
        sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
    catch
    end
    cd(resultsFolder);
    if  cp.Se > 0.79
        
        saveas(PrettyCM, ['BalCl_MB10_PrettyCM_' num2str(qq) '.jpg']);
        
        saveas(ROC_1, ['BalCl_MB10_ROC_1_' num2str(qq) '.jpg']);
        saveas(ROC_2, ['BalCl_MB10_ROC_2_' num2str(qq) '.jpg']);
        
        TrainingNeuNetw = findall(groot, 'Type', 'Figure');
        
        save(['RN_BalCl_MinBat10_' num2str(qq) '.mat'],'accuracy', 'AUC',...
            'c', 'cm', 'Confusion_Matrix', 'cp', ...
            'ind', 'miniBatchSize', 'netTransfer',...
            'numClasses', 'OPTROCPT', 'outputs', 'per', 'predictedLabels',...
            'scores','SUBY', 'SUBYNAMES', 'T', 'targets',...
            'TrainingNeuNetw','TheTableNice', 'validationImages', 'X', 'Y');
        %saveas(Architecture_1, ['UnCl_MB10_Architecture_1_' num2str(qq) '.jpg']);
        %saveas(Architecture_2, ['UnCl_MB10_Architecture_2_' num2str(qq) '.jpg']);
        
        %saveas(Second_Figure, ['UnCl_MB10_2nd_Figure_' num2str(qq) '.jpg']);
        
        
        
        try
            msg=sprintf('%s loop %d from %d MiniBatchSize %d , Sensitivity = %d , Balanced Accuracy = %d',...
                Name, qq, nLoops, miniBatchSize, cp.Se, cp.BA);
            sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
        catch
        end
        
%         close all
        % Save the figure of Training Progress in PNG format
        % saveas(TrainingNeuNetw,['RN_UnCl_MinBat10_' num2str(qq) '.png']);
        
        
%     else
%         save(['RN_BalCl_MinBat10_' num2str(qq) '.mat'],'accuracy', 'AUC',...
%             'cm', 'Confusion_Matrix', 'cp',...
%             'outputs', 'per', 'scores','SUBY', 'SUBYNAMES', 'T', 'targets',...
%             'TheTableNice', 'validationImages', 'X', 'Y');
%         saveas(PrettyCM, ['BalCl_MB10_PrettyCM_' num2str(qq) '.jpg']);
%         
%         try
%             msg=sprintf('%s loop %d from %d MiniBatchSize %d , Another one bits the dust, Sensitivity = %d',...
%                 Name, qq, nLoops, miniBatchSize, cp.Se);
%             sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
%         catch
%         end
        % Close all the figures
        
    end
    
    close all
    delete(findall(0));
    cd(codeFolder);
    % clearvars -except codeFolder resultsFolder TrainingFolder ValidationFolder nLoops qq
    
end

% Close progress bar
ioi_text_waitbar('Clear');

%%
nLoops = 1230;

for qq = 1221:nLoops
    
      
    ioi_text_waitbar(qq/nLoops, sprintf('ResNet101 Balanced MiniBatch=10, Running loop %d from %d', qq, 1440))
    
    close all
    trainingImages = imageDatastore(TrainingFolder, ...
        'IncludeSubfolders',true,...
        'FileExtensions','.tif','LabelSource','foldernames');
    
    trainingImages.ReadFcn = @(loc)imresize(imread(loc),[224, 224]);
    
    validationImages = imageDatastore(ValidationFolder, ...
        'IncludeSubfolders',true,...
        'FileExtensions','.tif','LabelSource','foldernames');
    
    validationImages.ReadFcn = @(loc)imresize(imread(loc),[224, 224]);
    
    
    % Load the pretrained GoogleNet network
    
    net = resnet101;
    net.Layers;
    
    % Extract the layer graph from the trained network and plot the layer graph.
    
    lgraph = layerGraph(net);
    Architecture_1 = figure('Units','normalized','Position',[0.1 0.1 0.8 0.8]);
    plot(lgraph);
    
       
    lgraph = removeLayers(lgraph, {'fc1000', 'prob', 'ClassificationLayer_predictions'});
    
    numClasses = numel(categories(trainingImages.Labels));
    newLayers = [
        fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',10,'BiasLearnRateFactor', 10)
        softmaxLayer('Name','softmax')
        classificationLayer('Name','classoutput')];
    lgraph = addLayers(lgraph,newLayers);
    
    lgraph = connectLayers(lgraph,'pool5','fc');
    
    Architecture_2 = figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
    plot(lgraph);
    ylim([0,10]);
    
    miniBatchSize = 10;
    
    %numIterationsPerEpoch = floor(numel(trainingImages.Labels)/miniBatchSize);
    options = trainingOptions('sgdm',...
        'ExecutionEnvironment','gpu',...
        'MiniBatchSize',miniBatchSize,...
        'MaxEpochs',100,...
        'InitialLearnRate',1e-5,...
        'VerboseFrequency',1,...
        'Plots','training-progress',...
        'ValidationData',validationImages,...
        'ValidationFrequency',5);
    netTransfer = trainNetwork(trainingImages,lgraph,options);
    
    %% Classify the validation images using the fine-tuned network.
    
    [predictedLabels, scores] = classify(netTransfer,validationImages,'ExecutionEnvironment','cpu');
    accuracy = mean(predictedLabels == validationImages.Labels);
    targets = grp2idx(validationImages.Labels);
    targets(targets==2) = 0;
    targets = ~targets;
    outputs = grp2idx(predictedLabels);
    outputs(outputs==2) = 0;
    outputs = ~outputs;
    [c,cm,ind,per] = confusion(targets', outputs');
    
    Confusion_Matrix=figure; plotconfusion(targets', outputs');
    
    %%
    
    addpath(genpath(resultsFolder))
    cp = my_classperf(cm);
    ROC_1=figure; plotroc(targets', scores(:,2)');
    
    [X,Y,T,AUC,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(targets', scores(:,2)',1,...
        'NBoot',1000,'TVals',0:0.05:1);
    ROC_2=figure; errorbar(X(:,3),Y(:,1),Y(:,1)-Y(:,2),Y(:,3)-Y(:,1));
    TheTableNice = table(validationImages.Labels,predictedLabels,'VariableNames',{'Actual','Predicted'});
    Confusion_Matrix1 = figure;
    PrettyCM = heatmap(TheTableNice,'Predicted','Actual');
    
    %     idx = randperm(20,4);
    %     Second_Figure = figure
    %     for i = 1:numel(idx)
    %         subplot(2,2,i)
    %         I = readimage(validationImages,idx(i));
    %         label = predictedLabels(idx(i));
    %         imshow(I)
    %         title(char(label))
    %     end
    
    try
        msg=sprintf('%s loop %d from %d MiniBatchSize %d with the results %d', Name, qq, nLoops, miniBatchSize, cp);
        sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
    catch
    end
    cd(resultsFolder);
    if  cp.Se > 0.79
        
        saveas(PrettyCM, ['BalCl_MB10_PrettyCM_' num2str(qq) '.jpg']);
        
        saveas(ROC_1, ['BalCl_MB10_ROC_1_' num2str(qq) '.jpg']);
        saveas(ROC_2, ['BalCl_MB10_ROC_2_' num2str(qq) '.jpg']);
        
        TrainingNeuNetw = findall(groot, 'Type', 'Figure');
        
        save(['RN_BalCl_MinBat10_' num2str(qq) '.mat'],'accuracy', 'AUC',...
            'c', 'cm', 'Confusion_Matrix', 'cp', ...
            'ind', 'miniBatchSize', 'netTransfer',...
            'numClasses', 'OPTROCPT', 'outputs', 'per', 'predictedLabels',...
            'scores','SUBY', 'SUBYNAMES', 'T', 'targets',...
            'TrainingNeuNetw','TheTableNice', 'validationImages', 'X', 'Y');
        %saveas(Architecture_1, ['UnCl_MB10_Architecture_1_' num2str(qq) '.jpg']);
        %saveas(Architecture_2, ['UnCl_MB10_Architecture_2_' num2str(qq) '.jpg']);
        
        %saveas(Second_Figure, ['UnCl_MB10_2nd_Figure_' num2str(qq) '.jpg']);
        
        
        
        try
            msg=sprintf('%s loop %d from %d MiniBatchSize %d , Sensitivity = %d , Balanced Accuracy = %d',...
                Name, qq, nLoops, miniBatchSize, cp.Se, cp.BA);
            sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
        catch
        end
        
%         close all
        % Save the figure of Training Progress in PNG format
        % saveas(TrainingNeuNetw,['RN_UnCl_MinBat10_' num2str(qq) '.png']);
        
        
%     else
%         save(['RN_BalCl_MinBat10_' num2str(qq) '.mat'],'accuracy', 'AUC',...
%             'cm', 'Confusion_Matrix', 'cp',...
%             'outputs', 'per', 'scores','SUBY', 'SUBYNAMES', 'T', 'targets',...
%             'TheTableNice', 'validationImages', 'X', 'Y');
%         saveas(PrettyCM, ['BalCl_MB10_PrettyCM_' num2str(qq) '.jpg']);
%         
%         try
%             msg=sprintf('%s loop %d from %d MiniBatchSize %d , Another one bits the dust, Sensitivity = %d',...
%                 Name, qq, nLoops, miniBatchSize, cp.Se);
%             sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
%         catch
%         end
        % Close all the figures
        
    end
    
    close all
    delete(findall(0));
    cd(codeFolder);
    % clearvars -except codeFolder resultsFolder TrainingFolder ValidationFolder nLoops qq
    
end

% Close progress bar
ioi_text_waitbar('Clear');

%%
nLoops = 1240;

for qq = 1231:nLoops
    
      
    ioi_text_waitbar(qq/nLoops, sprintf('ResNet101 Balanced MiniBatch=10, Running loop %d from %d', qq, 1440))
    
    close all
    trainingImages = imageDatastore(TrainingFolder, ...
        'IncludeSubfolders',true,...
        'FileExtensions','.tif','LabelSource','foldernames');
    
    trainingImages.ReadFcn = @(loc)imresize(imread(loc),[224, 224]);
    
    validationImages = imageDatastore(ValidationFolder, ...
        'IncludeSubfolders',true,...
        'FileExtensions','.tif','LabelSource','foldernames');
    
    validationImages.ReadFcn = @(loc)imresize(imread(loc),[224, 224]);
    
    
    % Load the pretrained GoogleNet network
    
    net = resnet101;
    net.Layers;
    
    % Extract the layer graph from the trained network and plot the layer graph.
    
    lgraph = layerGraph(net);
    Architecture_1 = figure('Units','normalized','Position',[0.1 0.1 0.8 0.8]);
    plot(lgraph);
    
       
    lgraph = removeLayers(lgraph, {'fc1000', 'prob', 'ClassificationLayer_predictions'});
    
    numClasses = numel(categories(trainingImages.Labels));
    newLayers = [
        fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',10,'BiasLearnRateFactor', 20)
        softmaxLayer('Name','softmax')
        classificationLayer('Name','classoutput')];
    lgraph = addLayers(lgraph,newLayers);
    
    lgraph = connectLayers(lgraph,'pool5','fc');
    
    Architecture_2 = figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
    plot(lgraph);
    ylim([0,10]);
    
    miniBatchSize = 10;
    
    %numIterationsPerEpoch = floor(numel(trainingImages.Labels)/miniBatchSize);
    options = trainingOptions('sgdm',...
        'ExecutionEnvironment','gpu',...
        'MiniBatchSize',miniBatchSize,...
        'MaxEpochs',100,...
        'InitialLearnRate',1e-5,...
        'VerboseFrequency',1,...
        'Plots','training-progress',...
        'ValidationData',validationImages,...
        'ValidationFrequency',5);
    netTransfer = trainNetwork(trainingImages,lgraph,options);
    
    %% Classify the validation images using the fine-tuned network.
    
    [predictedLabels, scores] = classify(netTransfer,validationImages,'ExecutionEnvironment','cpu');
    accuracy = mean(predictedLabels == validationImages.Labels);
    targets = grp2idx(validationImages.Labels);
    targets(targets==2) = 0;
    targets = ~targets;
    outputs = grp2idx(predictedLabels);
    outputs(outputs==2) = 0;
    outputs = ~outputs;
    [c,cm,ind,per] = confusion(targets', outputs');
    
    Confusion_Matrix=figure; plotconfusion(targets', outputs');
    
    %%
    
    addpath(genpath(resultsFolder))
    cp = my_classperf(cm);
    ROC_1=figure; plotroc(targets', scores(:,2)');
    
    [X,Y,T,AUC,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(targets', scores(:,2)',1,...
        'NBoot',1000,'TVals',0:0.05:1);
    ROC_2=figure; errorbar(X(:,3),Y(:,1),Y(:,1)-Y(:,2),Y(:,3)-Y(:,1));
    TheTableNice = table(validationImages.Labels,predictedLabels,'VariableNames',{'Actual','Predicted'});
    Confusion_Matrix1 = figure;
    PrettyCM = heatmap(TheTableNice,'Predicted','Actual');
    
    %     idx = randperm(20,4);
    %     Second_Figure = figure
    %     for i = 1:numel(idx)
    %         subplot(2,2,i)
    %         I = readimage(validationImages,idx(i));
    %         label = predictedLabels(idx(i));
    %         imshow(I)
    %         title(char(label))
    %     end
    
    try
        msg=sprintf('%s loop %d from %d MiniBatchSize %d with the results %d', Name, qq, nLoops, miniBatchSize, cp);
        sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
    catch
    end
    cd(resultsFolder);
    if  cp.Se > 0.79
        
        saveas(PrettyCM, ['BalCl_MB10_PrettyCM_' num2str(qq) '.jpg']);
        
        saveas(ROC_1, ['BalCl_MB10_ROC_1_' num2str(qq) '.jpg']);
        saveas(ROC_2, ['BalCl_MB10_ROC_2_' num2str(qq) '.jpg']);
        
        TrainingNeuNetw = findall(groot, 'Type', 'Figure');
        
        save(['RN_BalCl_MinBat10_' num2str(qq) '.mat'],'accuracy', 'AUC',...
            'c', 'cm', 'Confusion_Matrix', 'cp', ...
            'ind', 'miniBatchSize', 'netTransfer',...
            'numClasses', 'OPTROCPT', 'outputs', 'per', 'predictedLabels',...
            'scores','SUBY', 'SUBYNAMES', 'T', 'targets',...
            'TrainingNeuNetw','TheTableNice', 'validationImages', 'X', 'Y');
        %saveas(Architecture_1, ['UnCl_MB10_Architecture_1_' num2str(qq) '.jpg']);
        %saveas(Architecture_2, ['UnCl_MB10_Architecture_2_' num2str(qq) '.jpg']);
        
        %saveas(Second_Figure, ['UnCl_MB10_2nd_Figure_' num2str(qq) '.jpg']);
        
        
        
        try
            msg=sprintf('%s loop %d from %d MiniBatchSize %d , Sensitivity = %d , Balanced Accuracy = %d',...
                Name, qq, nLoops, miniBatchSize, cp.Se, cp.BA);
            sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
        catch
        end
        
%         close all
        % Save the figure of Training Progress in PNG format
        % saveas(TrainingNeuNetw,['RN_UnCl_MinBat10_' num2str(qq) '.png']);
        
        
%     else
%         save(['RN_BalCl_MinBat10_' num2str(qq) '.mat'],'accuracy', 'AUC',...
%             'cm', 'Confusion_Matrix', 'cp',...
%             'outputs', 'per', 'scores','SUBY', 'SUBYNAMES', 'T', 'targets',...
%             'TheTableNice', 'validationImages', 'X', 'Y');
%         saveas(PrettyCM, ['BalCl_MB10_PrettyCM_' num2str(qq) '.jpg']);
%         
%         try
%             msg=sprintf('%s loop %d from %d MiniBatchSize %d , Another one bits the dust, Sensitivity = %d',...
%                 Name, qq, nLoops, miniBatchSize, cp.Se);
%             sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
%         catch
%         end
        % Close all the figures
        
    end
    
    close all
    delete(findall(0));
    cd(codeFolder);
    % clearvars -except codeFolder resultsFolder TrainingFolder ValidationFolder nLoops qq
    
end

% Close progress bar
ioi_text_waitbar('Clear');

%%
nLoops = 1250;

for qq = 1241:nLoops
    
      
    ioi_text_waitbar(qq/nLoops, sprintf('ResNet101 Balanced MiniBatch=10, Running loop %d from %d', qq, 1440))
    
    close all
    trainingImages = imageDatastore(TrainingFolder, ...
        'IncludeSubfolders',true,...
        'FileExtensions','.tif','LabelSource','foldernames');
    
    trainingImages.ReadFcn = @(loc)imresize(imread(loc),[224, 224]);
    
    validationImages = imageDatastore(ValidationFolder, ...
        'IncludeSubfolders',true,...
        'FileExtensions','.tif','LabelSource','foldernames');
    
    validationImages.ReadFcn = @(loc)imresize(imread(loc),[224, 224]);
    
    
    % Load the pretrained GoogleNet network
    
    net = resnet101;
    net.Layers;
    
    % Extract the layer graph from the trained network and plot the layer graph.
    
    lgraph = layerGraph(net);
    Architecture_1 = figure('Units','normalized','Position',[0.1 0.1 0.8 0.8]);
    plot(lgraph);
    
       
    lgraph = removeLayers(lgraph, {'fc1000', 'prob', 'ClassificationLayer_predictions'});
    
    numClasses = numel(categories(trainingImages.Labels));
    newLayers = [
        fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',10,'BiasLearnRateFactor', 40)
        softmaxLayer('Name','softmax')
        classificationLayer('Name','classoutput')];
    lgraph = addLayers(lgraph,newLayers);
    
    lgraph = connectLayers(lgraph,'pool5','fc');
    
    Architecture_2 = figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
    plot(lgraph);
    ylim([0,10]);
    
    miniBatchSize = 10;
    
    %numIterationsPerEpoch = floor(numel(trainingImages.Labels)/miniBatchSize);
    options = trainingOptions('sgdm',...
        'ExecutionEnvironment','gpu',...
        'MiniBatchSize',miniBatchSize,...
        'MaxEpochs',100,...
        'InitialLearnRate',1e-5,...
        'VerboseFrequency',1,...
        'Plots','training-progress',...
        'ValidationData',validationImages,...
        'ValidationFrequency',5);
    netTransfer = trainNetwork(trainingImages,lgraph,options);
    
    %% Classify the validation images using the fine-tuned network.
    
    [predictedLabels, scores] = classify(netTransfer,validationImages,'ExecutionEnvironment','cpu');
    accuracy = mean(predictedLabels == validationImages.Labels);
    targets = grp2idx(validationImages.Labels);
    targets(targets==2) = 0;
    targets = ~targets;
    outputs = grp2idx(predictedLabels);
    outputs(outputs==2) = 0;
    outputs = ~outputs;
    [c,cm,ind,per] = confusion(targets', outputs');
    
    Confusion_Matrix=figure; plotconfusion(targets', outputs');
    
    %%
    
    addpath(genpath(resultsFolder))
    cp = my_classperf(cm);
    ROC_1=figure; plotroc(targets', scores(:,2)');
    
    [X,Y,T,AUC,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(targets', scores(:,2)',1,...
        'NBoot',1000,'TVals',0:0.05:1);
    ROC_2=figure; errorbar(X(:,3),Y(:,1),Y(:,1)-Y(:,2),Y(:,3)-Y(:,1));
    TheTableNice = table(validationImages.Labels,predictedLabels,'VariableNames',{'Actual','Predicted'});
    Confusion_Matrix1 = figure;
    PrettyCM = heatmap(TheTableNice,'Predicted','Actual');
    
    %     idx = randperm(20,4);
    %     Second_Figure = figure
    %     for i = 1:numel(idx)
    %         subplot(2,2,i)
    %         I = readimage(validationImages,idx(i));
    %         label = predictedLabels(idx(i));
    %         imshow(I)
    %         title(char(label))
    %     end
    
    try
        msg=sprintf('%s loop %d from %d MiniBatchSize %d with the results %d', Name, qq, nLoops, miniBatchSize, cp);
        sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
    catch
    end
    cd(resultsFolder);
    if  cp.Se > 0.79
        
        saveas(PrettyCM, ['BalCl_MB10_PrettyCM_' num2str(qq) '.jpg']);
        
        saveas(ROC_1, ['BalCl_MB10_ROC_1_' num2str(qq) '.jpg']);
        saveas(ROC_2, ['BalCl_MB10_ROC_2_' num2str(qq) '.jpg']);
        
        TrainingNeuNetw = findall(groot, 'Type', 'Figure');
        
        save(['RN_BalCl_MinBat10_' num2str(qq) '.mat'],'accuracy', 'AUC',...
            'c', 'cm', 'Confusion_Matrix', 'cp', ...
            'ind', 'miniBatchSize', 'netTransfer',...
            'numClasses', 'OPTROCPT', 'outputs', 'per', 'predictedLabels',...
            'scores','SUBY', 'SUBYNAMES', 'T', 'targets',...
            'TrainingNeuNetw','TheTableNice', 'validationImages', 'X', 'Y');
        %saveas(Architecture_1, ['UnCl_MB10_Architecture_1_' num2str(qq) '.jpg']);
        %saveas(Architecture_2, ['UnCl_MB10_Architecture_2_' num2str(qq) '.jpg']);
        
        %saveas(Second_Figure, ['UnCl_MB10_2nd_Figure_' num2str(qq) '.jpg']);
        
        
        
        try
            msg=sprintf('%s loop %d from %d MiniBatchSize %d , Sensitivity = %d , Balanced Accuracy = %d',...
                Name, qq, nLoops, miniBatchSize, cp.Se, cp.BA);
            sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
        catch
        end
        
%         close all
        % Save the figure of Training Progress in PNG format
        % saveas(TrainingNeuNetw,['RN_UnCl_MinBat10_' num2str(qq) '.png']);
        
        
%     else
%         save(['RN_BalCl_MinBat10_' num2str(qq) '.mat'],'accuracy', 'AUC',...
%             'cm', 'Confusion_Matrix', 'cp',...
%             'outputs', 'per', 'scores','SUBY', 'SUBYNAMES', 'T', 'targets',...
%             'TheTableNice', 'validationImages', 'X', 'Y');
%         saveas(PrettyCM, ['BalCl_MB10_PrettyCM_' num2str(qq) '.jpg']);
%         
%         try
%             msg=sprintf('%s loop %d from %d MiniBatchSize %d , Another one bits the dust, Sensitivity = %d',...
%                 Name, qq, nLoops, miniBatchSize, cp.Se);
%             sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
%         catch
%         end
        % Close all the figures
        
    end
    
    close all
    delete(findall(0));
    cd(codeFolder);
    % clearvars -except codeFolder resultsFolder TrainingFolder ValidationFolder nLoops qq
    
end

% Close progress bar
ioi_text_waitbar('Clear');


nLoops = 1260;

for qq = 1251:nLoops
    
      
    ioi_text_waitbar(qq/nLoops, sprintf('ResNet101 Balanced MiniBatch=10, Running loop %d from %d', qq, 1440))
    
    close all
    trainingImages = imageDatastore(TrainingFolder, ...
        'IncludeSubfolders',true,...
        'FileExtensions','.tif','LabelSource','foldernames');
    
    trainingImages.ReadFcn = @(loc)imresize(imread(loc),[224, 224]);
    
    validationImages = imageDatastore(ValidationFolder, ...
        'IncludeSubfolders',true,...
        'FileExtensions','.tif','LabelSource','foldernames');
    
    validationImages.ReadFcn = @(loc)imresize(imread(loc),[224, 224]);
    
    
    % Load the pretrained GoogleNet network
    
    net = resnet101;
    net.Layers;
    
    % Extract the layer graph from the trained network and plot the layer graph.
    
    lgraph = layerGraph(net);
    Architecture_1 = figure('Units','normalized','Position',[0.1 0.1 0.8 0.8]);
    plot(lgraph);
    
       
    lgraph = removeLayers(lgraph, {'fc1000', 'prob', 'ClassificationLayer_predictions'});
    
    numClasses = numel(categories(trainingImages.Labels));
    newLayers = [
        fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',10,'BiasLearnRateFactor', 60)
        softmaxLayer('Name','softmax')
        classificationLayer('Name','classoutput')];
    lgraph = addLayers(lgraph,newLayers);
    
    lgraph = connectLayers(lgraph,'pool5','fc');
    
    Architecture_2 = figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
    plot(lgraph);
    ylim([0,10]);
    
    miniBatchSize = 10;
    
    %numIterationsPerEpoch = floor(numel(trainingImages.Labels)/miniBatchSize);
    options = trainingOptions('sgdm',...
        'ExecutionEnvironment','gpu',...
        'MiniBatchSize',miniBatchSize,...
        'MaxEpochs',100,...
        'InitialLearnRate',1e-5,...
        'VerboseFrequency',1,...
        'Plots','training-progress',...
        'ValidationData',validationImages,...
        'ValidationFrequency',5);
    netTransfer = trainNetwork(trainingImages,lgraph,options);
    
    %% Classify the validation images using the fine-tuned network.
    
    [predictedLabels, scores] = classify(netTransfer,validationImages,'ExecutionEnvironment','cpu');
    accuracy = mean(predictedLabels == validationImages.Labels);
    targets = grp2idx(validationImages.Labels);
    targets(targets==2) = 0;
    targets = ~targets;
    outputs = grp2idx(predictedLabels);
    outputs(outputs==2) = 0;
    outputs = ~outputs;
    [c,cm,ind,per] = confusion(targets', outputs');
    
    Confusion_Matrix=figure; plotconfusion(targets', outputs');
    
    %%
    
    addpath(genpath(resultsFolder))
    cp = my_classperf(cm);
    ROC_1=figure; plotroc(targets', scores(:,2)');
    
    [X,Y,T,AUC,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(targets', scores(:,2)',1,...
        'NBoot',1000,'TVals',0:0.05:1);
    ROC_2=figure; errorbar(X(:,3),Y(:,1),Y(:,1)-Y(:,2),Y(:,3)-Y(:,1));
    TheTableNice = table(validationImages.Labels,predictedLabels,'VariableNames',{'Actual','Predicted'});
    Confusion_Matrix1 = figure;
    PrettyCM = heatmap(TheTableNice,'Predicted','Actual');
    
    %     idx = randperm(20,4);
    %     Second_Figure = figure
    %     for i = 1:numel(idx)
    %         subplot(2,2,i)
    %         I = readimage(validationImages,idx(i));
    %         label = predictedLabels(idx(i));
    %         imshow(I)
    %         title(char(label))
    %     end
    
    try
        msg=sprintf('%s loop %d from %d MiniBatchSize %d with the results %d', Name, qq, nLoops, miniBatchSize, cp);
        sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
    catch
    end
    cd(resultsFolder);
    if  cp.Se > 0.79
        
        saveas(PrettyCM, ['BalCl_MB10_PrettyCM_' num2str(qq) '.jpg']);
        
        saveas(ROC_1, ['BalCl_MB10_ROC_1_' num2str(qq) '.jpg']);
        saveas(ROC_2, ['BalCl_MB10_ROC_2_' num2str(qq) '.jpg']);
        
        TrainingNeuNetw = findall(groot, 'Type', 'Figure');
        
        save(['RN_BalCl_MinBat10_' num2str(qq) '.mat'],'accuracy', 'AUC',...
            'c', 'cm', 'Confusion_Matrix', 'cp', ...
            'ind', 'miniBatchSize', 'netTransfer',...
            'numClasses', 'OPTROCPT', 'outputs', 'per', 'predictedLabels',...
            'scores','SUBY', 'SUBYNAMES', 'T', 'targets',...
            'TrainingNeuNetw','TheTableNice', 'validationImages', 'X', 'Y');
        %saveas(Architecture_1, ['UnCl_MB10_Architecture_1_' num2str(qq) '.jpg']);
        %saveas(Architecture_2, ['UnCl_MB10_Architecture_2_' num2str(qq) '.jpg']);
        
        %saveas(Second_Figure, ['UnCl_MB10_2nd_Figure_' num2str(qq) '.jpg']);
        
        
        
        try
            msg=sprintf('%s loop %d from %d MiniBatchSize %d , Sensitivity = %d , Balanced Accuracy = %d',...
                Name, qq, nLoops, miniBatchSize, cp.Se, cp.BA);
            sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
        catch
        end
        
%         close all
        % Save the figure of Training Progress in PNG format
        % saveas(TrainingNeuNetw,['RN_UnCl_MinBat10_' num2str(qq) '.png']);
        
        
%     else
%         save(['RN_BalCl_MinBat10_' num2str(qq) '.mat'],'accuracy', 'AUC',...
%             'cm', 'Confusion_Matrix', 'cp',...
%             'outputs', 'per', 'scores','SUBY', 'SUBYNAMES', 'T', 'targets',...
%             'TheTableNice', 'validationImages', 'X', 'Y');
%         saveas(PrettyCM, ['BalCl_MB10_PrettyCM_' num2str(qq) '.jpg']);
%         
%         try
%             msg=sprintf('%s loop %d from %d MiniBatchSize %d , Another one bits the dust, Sensitivity = %d',...
%                 Name, qq, nLoops, miniBatchSize, cp.Se);
%             sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
%         catch
%         end
        % Close all the figures
        
    end
    
    close all
    delete(findall(0));
    cd(codeFolder);
    % clearvars -except codeFolder resultsFolder TrainingFolder ValidationFolder nLoops qq
    
end

% Close progress bar
ioi_text_waitbar('Clear');

% *ResNet 101 for Database of Brasil, IJC & ISSSTE*

% Unbalanced Class
% MiniBatch = 10

% Analysis of three databases is realized for compare the results, only the
% results that has a Sensitivity bigger than 80% is saved

clear; close all; clc;

% Folders Location
codeFolder = 'C:\Users\LANCYTT\Documents\Juan Carlos Torres\Thermograms Analysis';
% addpath('I:\Archivos Juan Carlos Torres\MATLAB\MATLAB Codigos');

resultsFolder = 'C:\Users\LANCYTT\Documents\Juan Carlos Torres\Results\ResNet101\Iteraciones\Balanced';
addpath('C:\Users\LANCYTT\Documents\Juan Carlos Torres\Thermograms Analysis');

TrainingFolder = 'C:\Users\LANCYTT\Documents\Juan Carlos Torres\Data Augmented Thermography\Data Augmented Bra+IJC+ISSSTE\Balanced Augmented Modify';
ValidationFolder = 'C:\Users\LANCYTT\Documents\Juan Carlos Torres\Data Augmented Thermography\Train - Valid Bra+IJC+ISSSTE\Validation Balanced Class';

pat_sendmail

ioi_text_waitbar(0, 'Running ResNet 101, Balanced MiniBatch 10... Please wait...');

Name=sprintf('ResNet101 Balanced');
nLoops = 1270;

for qq = 1261:nLoops
    
      
    ioi_text_waitbar(qq/nLoops, sprintf('ResNet101 Balanced MiniBatch=10, Running loop %d from %d', qq, 1440))
    
    close all
    trainingImages = imageDatastore(TrainingFolder, ...
        'IncludeSubfolders',true,...
        'FileExtensions','.tif','LabelSource','foldernames');
    
    trainingImages.ReadFcn = @(loc)imresize(imread(loc),[224, 224]);
    
    validationImages = imageDatastore(ValidationFolder, ...
        'IncludeSubfolders',true,...
        'FileExtensions','.tif','LabelSource','foldernames');
    
    validationImages.ReadFcn = @(loc)imresize(imread(loc),[224, 224]);
    
    
    % Load the pretrained GoogleNet network
    
    net = resnet101;
    net.Layers;
    
    % Extract the layer graph from the trained network and plot the layer graph.
    
    lgraph = layerGraph(net);
    Architecture_1 = figure('Units','normalized','Position',[0.1 0.1 0.8 0.8]);
    plot(lgraph);
    
       
    lgraph = removeLayers(lgraph, {'fc1000', 'prob', 'ClassificationLayer_predictions'});
    
    numClasses = numel(categories(trainingImages.Labels));
    newLayers = [
        fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',20,'BiasLearnRateFactor', 1)
        softmaxLayer('Name','softmax')
        classificationLayer('Name','classoutput')];
    lgraph = addLayers(lgraph,newLayers);
    
    lgraph = connectLayers(lgraph,'pool5','fc');
    
    Architecture_2 = figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
    plot(lgraph);
    ylim([0,10]);
    
    miniBatchSize = 10;
    
    %numIterationsPerEpoch = floor(numel(trainingImages.Labels)/miniBatchSize);
    options = trainingOptions('sgdm',...
        'ExecutionEnvironment','gpu',...
        'MiniBatchSize',miniBatchSize,...
        'MaxEpochs',100,...
        'InitialLearnRate',1e-5,...
        'VerboseFrequency',1,...
        'Plots','training-progress',...
        'ValidationData',validationImages,...
        'ValidationFrequency',5);
    netTransfer = trainNetwork(trainingImages,lgraph,options);
    
    %% Classify the validation images using the fine-tuned network.
    
    [predictedLabels, scores] = classify(netTransfer,validationImages,'ExecutionEnvironment','cpu');
    accuracy = mean(predictedLabels == validationImages.Labels);
    targets = grp2idx(validationImages.Labels);
    targets(targets==2) = 0;
    targets = ~targets;
    outputs = grp2idx(predictedLabels);
    outputs(outputs==2) = 0;
    outputs = ~outputs;
    [c,cm,ind,per] = confusion(targets', outputs');
    
    Confusion_Matrix=figure; plotconfusion(targets', outputs');
    
    %%
    
    addpath(genpath(resultsFolder))
    cp = my_classperf(cm);
    ROC_1=figure; plotroc(targets', scores(:,2)');
    
    [X,Y,T,AUC,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(targets', scores(:,2)',1,...
        'NBoot',1000,'TVals',0:0.05:1);
    ROC_2=figure; errorbar(X(:,3),Y(:,1),Y(:,1)-Y(:,2),Y(:,3)-Y(:,1));
    TheTableNice = table(validationImages.Labels,predictedLabels,'VariableNames',{'Actual','Predicted'});
    Confusion_Matrix1 = figure;
    PrettyCM = heatmap(TheTableNice,'Predicted','Actual');
    
    %     idx = randperm(20,4);
    %     Second_Figure = figure
    %     for i = 1:numel(idx)
    %         subplot(2,2,i)
    %         I = readimage(validationImages,idx(i));
    %         label = predictedLabels(idx(i));
    %         imshow(I)
    %         title(char(label))
    %     end
    
    try
        msg=sprintf('%s loop %d from %d MiniBatchSize %d with the results %d', Name, qq, nLoops, miniBatchSize, cp);
        sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
    catch
    end
    cd(resultsFolder);
    if  cp.Se > 0.79
        
        saveas(PrettyCM, ['BalCl_MB10_PrettyCM_' num2str(qq) '.jpg']);
        
        saveas(ROC_1, ['BalCl_MB10_ROC_1_' num2str(qq) '.jpg']);
        saveas(ROC_2, ['BalCl_MB10_ROC_2_' num2str(qq) '.jpg']);
        
        TrainingNeuNetw = findall(groot, 'Type', 'Figure');
        
        save(['RN_BalCl_MinBat10_' num2str(qq) '.mat'],'accuracy', 'AUC',...
            'c', 'cm', 'Confusion_Matrix', 'cp', ...
            'ind', 'miniBatchSize', 'netTransfer',...
            'numClasses', 'OPTROCPT', 'outputs', 'per', 'predictedLabels',...
            'scores','SUBY', 'SUBYNAMES', 'T', 'targets',...
            'TrainingNeuNetw','TheTableNice', 'validationImages', 'X', 'Y');
        %saveas(Architecture_1, ['UnCl_MB10_Architecture_1_' num2str(qq) '.jpg']);
        %saveas(Architecture_2, ['UnCl_MB10_Architecture_2_' num2str(qq) '.jpg']);
        
        %saveas(Second_Figure, ['UnCl_MB10_2nd_Figure_' num2str(qq) '.jpg']);
        
        
        
        try
            msg=sprintf('%s loop %d from %d MiniBatchSize %d , Sensitivity = %d , Balanced Accuracy = %d',...
                Name, qq, nLoops, miniBatchSize, cp.Se, cp.BA);
            sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
        catch
        end
        
%         close all
        % Save the figure of Training Progress in PNG format
        % saveas(TrainingNeuNetw,['RN_UnCl_MinBat10_' num2str(qq) '.png']);
        
        
%     else
%         save(['RN_BalCl_MinBat10_' num2str(qq) '.mat'],'accuracy', 'AUC',...
%             'cm', 'Confusion_Matrix', 'cp',...
%             'outputs', 'per', 'scores','SUBY', 'SUBYNAMES', 'T', 'targets',...
%             'TheTableNice', 'validationImages', 'X', 'Y');
%         saveas(PrettyCM, ['BalCl_MB10_PrettyCM_' num2str(qq) '.jpg']);
%         
%         try
%             msg=sprintf('%s loop %d from %d MiniBatchSize %d , Another one bits the dust, Sensitivity = %d',...
%                 Name, qq, nLoops, miniBatchSize, cp.Se);
%             sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
%         catch
%         end
        % Close all the figures
        
    end
    
    close all
    delete(findall(0));
    cd(codeFolder);
    % clearvars -except codeFolder resultsFolder TrainingFolder ValidationFolder nLoops qq
    
end

% Close progress bar
ioi_text_waitbar('Clear');

nLoops = 1280;

for qq = 1271:nLoops
    
      
    ioi_text_waitbar(qq/nLoops, sprintf('ResNet101 Balanced MiniBatch=10, Running loop %d from %d', qq, 1440))
    
    close all
    trainingImages = imageDatastore(TrainingFolder, ...
        'IncludeSubfolders',true,...
        'FileExtensions','.tif','LabelSource','foldernames');
    
    trainingImages.ReadFcn = @(loc)imresize(imread(loc),[224, 224]);
    
    validationImages = imageDatastore(ValidationFolder, ...
        'IncludeSubfolders',true,...
        'FileExtensions','.tif','LabelSource','foldernames');
    
    validationImages.ReadFcn = @(loc)imresize(imread(loc),[224, 224]);
    
    
    % Load the pretrained GoogleNet network
    
    net = resnet101;
    net.Layers;
    
    % Extract the layer graph from the trained network and plot the layer graph.
    
    lgraph = layerGraph(net);
    Architecture_1 = figure('Units','normalized','Position',[0.1 0.1 0.8 0.8]);
    plot(lgraph);
    
       
    lgraph = removeLayers(lgraph, {'fc1000', 'prob', 'ClassificationLayer_predictions'});
    
    numClasses = numel(categories(trainingImages.Labels));
    newLayers = [
        fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',20,'BiasLearnRateFactor', 5)
        softmaxLayer('Name','softmax')
        classificationLayer('Name','classoutput')];
    lgraph = addLayers(lgraph,newLayers);
    
    lgraph = connectLayers(lgraph,'pool5','fc');
    
    Architecture_2 = figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
    plot(lgraph);
    ylim([0,10]);
    
    miniBatchSize = 10;
    
    %numIterationsPerEpoch = floor(numel(trainingImages.Labels)/miniBatchSize);
    options = trainingOptions('sgdm',...
        'ExecutionEnvironment','gpu',...
        'MiniBatchSize',miniBatchSize,...
        'MaxEpochs',100,...
        'InitialLearnRate',1e-5,...
        'VerboseFrequency',1,...
        'Plots','training-progress',...
        'ValidationData',validationImages,...
        'ValidationFrequency',5);
    netTransfer = trainNetwork(trainingImages,lgraph,options);
    
    %% Classify the validation images using the fine-tuned network.
    
    [predictedLabels, scores] = classify(netTransfer,validationImages,'ExecutionEnvironment','cpu');
    accuracy = mean(predictedLabels == validationImages.Labels);
    targets = grp2idx(validationImages.Labels);
    targets(targets==2) = 0;
    targets = ~targets;
    outputs = grp2idx(predictedLabels);
    outputs(outputs==2) = 0;
    outputs = ~outputs;
    [c,cm,ind,per] = confusion(targets', outputs');
    
    Confusion_Matrix=figure; plotconfusion(targets', outputs');
    
    %%
    
    addpath(genpath(resultsFolder))
    cp = my_classperf(cm);
    ROC_1=figure; plotroc(targets', scores(:,2)');
    
    [X,Y,T,AUC,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(targets', scores(:,2)',1,...
        'NBoot',1000,'TVals',0:0.05:1);
    ROC_2=figure; errorbar(X(:,3),Y(:,1),Y(:,1)-Y(:,2),Y(:,3)-Y(:,1));
    TheTableNice = table(validationImages.Labels,predictedLabels,'VariableNames',{'Actual','Predicted'});
    Confusion_Matrix1 = figure;
    PrettyCM = heatmap(TheTableNice,'Predicted','Actual');
    
    %     idx = randperm(20,4);
    %     Second_Figure = figure
    %     for i = 1:numel(idx)
    %         subplot(2,2,i)
    %         I = readimage(validationImages,idx(i));
    %         label = predictedLabels(idx(i));
    %         imshow(I)
    %         title(char(label))
    %     end
    
    try
        msg=sprintf('%s loop %d from %d MiniBatchSize %d with the results %d', Name, qq, nLoops, miniBatchSize, cp);
        sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
    catch
    end
    cd(resultsFolder);
    if  cp.Se > 0.79
        
        saveas(PrettyCM, ['BalCl_MB10_PrettyCM_' num2str(qq) '.jpg']);
        
        saveas(ROC_1, ['BalCl_MB10_ROC_1_' num2str(qq) '.jpg']);
        saveas(ROC_2, ['BalCl_MB10_ROC_2_' num2str(qq) '.jpg']);
        
        TrainingNeuNetw = findall(groot, 'Type', 'Figure');
        
        save(['RN_BalCl_MinBat10_' num2str(qq) '.mat'],'accuracy', 'AUC',...
            'c', 'cm', 'Confusion_Matrix', 'cp', ...
            'ind', 'miniBatchSize', 'netTransfer',...
            'numClasses', 'OPTROCPT', 'outputs', 'per', 'predictedLabels',...
            'scores','SUBY', 'SUBYNAMES', 'T', 'targets',...
            'TrainingNeuNetw','TheTableNice', 'validationImages', 'X', 'Y');
        %saveas(Architecture_1, ['UnCl_MB10_Architecture_1_' num2str(qq) '.jpg']);
        %saveas(Architecture_2, ['UnCl_MB10_Architecture_2_' num2str(qq) '.jpg']);
        
        %saveas(Second_Figure, ['UnCl_MB10_2nd_Figure_' num2str(qq) '.jpg']);
        
        
        
        try
            msg=sprintf('%s loop %d from %d MiniBatchSize %d , Sensitivity = %d , Balanced Accuracy = %d',...
                Name, qq, nLoops, miniBatchSize, cp.Se, cp.BA);
            sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
        catch
        end
        
%         close all
        % Save the figure of Training Progress in PNG format
        % saveas(TrainingNeuNetw,['RN_UnCl_MinBat10_' num2str(qq) '.png']);
        
        
%     else
%         save(['RN_BalCl_MinBat10_' num2str(qq) '.mat'],'accuracy', 'AUC',...
%             'cm', 'Confusion_Matrix', 'cp',...
%             'outputs', 'per', 'scores','SUBY', 'SUBYNAMES', 'T', 'targets',...
%             'TheTableNice', 'validationImages', 'X', 'Y');
%         saveas(PrettyCM, ['BalCl_MB10_PrettyCM_' num2str(qq) '.jpg']);
%         
%         try
%             msg=sprintf('%s loop %d from %d MiniBatchSize %d , Another one bits the dust, Sensitivity = %d',...
%                 Name, qq, nLoops, miniBatchSize, cp.Se);
%             sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
%         catch
%         end
        % Close all the figures
        
    end
    
    close all
    delete(findall(0));
    cd(codeFolder);
    % clearvars -except codeFolder resultsFolder TrainingFolder ValidationFolder nLoops qq
    
end

% Close progress bar
ioi_text_waitbar('Clear');

%%
nLoops = 1290;

for qq = 1281:nLoops
    
      
    ioi_text_waitbar(qq/nLoops, sprintf('ResNet101 Balanced MiniBatch=10, Running loop %d from %d', qq, 1440))
    
    close all
    trainingImages = imageDatastore(TrainingFolder, ...
        'IncludeSubfolders',true,...
        'FileExtensions','.tif','LabelSource','foldernames');
    
    trainingImages.ReadFcn = @(loc)imresize(imread(loc),[224, 224]);
    
    validationImages = imageDatastore(ValidationFolder, ...
        'IncludeSubfolders',true,...
        'FileExtensions','.tif','LabelSource','foldernames');
    
    validationImages.ReadFcn = @(loc)imresize(imread(loc),[224, 224]);
    
    
    % Load the pretrained GoogleNet network
    
    net = resnet101;
    net.Layers;
    
    % Extract the layer graph from the trained network and plot the layer graph.
    
    lgraph = layerGraph(net);
    Architecture_1 = figure('Units','normalized','Position',[0.1 0.1 0.8 0.8]);
    plot(lgraph);
    
       
    lgraph = removeLayers(lgraph, {'fc1000', 'prob', 'ClassificationLayer_predictions'});
    
    numClasses = numel(categories(trainingImages.Labels));
    newLayers = [
        fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',20,'BiasLearnRateFactor', 10)
        softmaxLayer('Name','softmax')
        classificationLayer('Name','classoutput')];
    lgraph = addLayers(lgraph,newLayers);
    
    lgraph = connectLayers(lgraph,'pool5','fc');
    
    Architecture_2 = figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
    plot(lgraph);
    ylim([0,10]);
    
    miniBatchSize = 10;
    
    %numIterationsPerEpoch = floor(numel(trainingImages.Labels)/miniBatchSize);
    options = trainingOptions('sgdm',...
        'ExecutionEnvironment','gpu',...
        'MiniBatchSize',miniBatchSize,...
        'MaxEpochs',100,...
        'InitialLearnRate',1e-5,...
        'VerboseFrequency',1,...
        'Plots','training-progress',...
        'ValidationData',validationImages,...
        'ValidationFrequency',5);
    netTransfer = trainNetwork(trainingImages,lgraph,options);
    
    %% Classify the validation images using the fine-tuned network.
    
    [predictedLabels, scores] = classify(netTransfer,validationImages,'ExecutionEnvironment','cpu');
    accuracy = mean(predictedLabels == validationImages.Labels);
    targets = grp2idx(validationImages.Labels);
    targets(targets==2) = 0;
    targets = ~targets;
    outputs = grp2idx(predictedLabels);
    outputs(outputs==2) = 0;
    outputs = ~outputs;
    [c,cm,ind,per] = confusion(targets', outputs');
    
    Confusion_Matrix=figure; plotconfusion(targets', outputs');
    
    %%
    
    addpath(genpath(resultsFolder))
    cp = my_classperf(cm);
    ROC_1=figure; plotroc(targets', scores(:,2)');
    
    [X,Y,T,AUC,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(targets', scores(:,2)',1,...
        'NBoot',1000,'TVals',0:0.05:1);
    ROC_2=figure; errorbar(X(:,3),Y(:,1),Y(:,1)-Y(:,2),Y(:,3)-Y(:,1));
    TheTableNice = table(validationImages.Labels,predictedLabels,'VariableNames',{'Actual','Predicted'});
    Confusion_Matrix1 = figure;
    PrettyCM = heatmap(TheTableNice,'Predicted','Actual');
    
    %     idx = randperm(20,4);
    %     Second_Figure = figure
    %     for i = 1:numel(idx)
    %         subplot(2,2,i)
    %         I = readimage(validationImages,idx(i));
    %         label = predictedLabels(idx(i));
    %         imshow(I)
    %         title(char(label))
    %     end
    
    try
        msg=sprintf('%s loop %d from %d MiniBatchSize %d with the results %d', Name, qq, nLoops, miniBatchSize, cp);
        sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
    catch
    end
    cd(resultsFolder);
    if  cp.Se > 0.79
        
        saveas(PrettyCM, ['BalCl_MB10_PrettyCM_' num2str(qq) '.jpg']);
        
        saveas(ROC_1, ['BalCl_MB10_ROC_1_' num2str(qq) '.jpg']);
        saveas(ROC_2, ['BalCl_MB10_ROC_2_' num2str(qq) '.jpg']);
        
        TrainingNeuNetw = findall(groot, 'Type', 'Figure');
        
        save(['RN_BalCl_MinBat10_' num2str(qq) '.mat'],'accuracy', 'AUC',...
            'c', 'cm', 'Confusion_Matrix', 'cp', ...
            'ind', 'miniBatchSize', 'netTransfer',...
            'numClasses', 'OPTROCPT', 'outputs', 'per', 'predictedLabels',...
            'scores','SUBY', 'SUBYNAMES', 'T', 'targets',...
            'TrainingNeuNetw','TheTableNice', 'validationImages', 'X', 'Y');
        %saveas(Architecture_1, ['UnCl_MB10_Architecture_1_' num2str(qq) '.jpg']);
        %saveas(Architecture_2, ['UnCl_MB10_Architecture_2_' num2str(qq) '.jpg']);
        
        %saveas(Second_Figure, ['UnCl_MB10_2nd_Figure_' num2str(qq) '.jpg']);
        
        
        
        try
            msg=sprintf('%s loop %d from %d MiniBatchSize %d , Sensitivity = %d , Balanced Accuracy = %d',...
                Name, qq, nLoops, miniBatchSize, cp.Se, cp.BA);
            sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
        catch
        end
        
%         close all
        % Save the figure of Training Progress in PNG format
        % saveas(TrainingNeuNetw,['RN_UnCl_MinBat10_' num2str(qq) '.png']);
        
        
%     else
%         save(['RN_BalCl_MinBat10_' num2str(qq) '.mat'],'accuracy', 'AUC',...
%             'cm', 'Confusion_Matrix', 'cp',...
%             'outputs', 'per', 'scores','SUBY', 'SUBYNAMES', 'T', 'targets',...
%             'TheTableNice', 'validationImages', 'X', 'Y');
%         saveas(PrettyCM, ['BalCl_MB10_PrettyCM_' num2str(qq) '.jpg']);
%         
%         try
%             msg=sprintf('%s loop %d from %d MiniBatchSize %d , Another one bits the dust, Sensitivity = %d',...
%                 Name, qq, nLoops, miniBatchSize, cp.Se);
%             sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
%         catch
%         end
        % Close all the figures
        
    end
    
    close all
    delete(findall(0));
    cd(codeFolder);
    % clearvars -except codeFolder resultsFolder TrainingFolder ValidationFolder nLoops qq
    
end

% Close progress bar
ioi_text_waitbar('Clear');

%%
nLoops = 1300;

for qq = 1291:nLoops
    
      
    ioi_text_waitbar(qq/nLoops, sprintf('ResNet101 Balanced MiniBatch=10, Running loop %d from %d', qq, 1440))
    
    close all
    trainingImages = imageDatastore(TrainingFolder, ...
        'IncludeSubfolders',true,...
        'FileExtensions','.tif','LabelSource','foldernames');
    
    trainingImages.ReadFcn = @(loc)imresize(imread(loc),[224, 224]);
    
    validationImages = imageDatastore(ValidationFolder, ...
        'IncludeSubfolders',true,...
        'FileExtensions','.tif','LabelSource','foldernames');
    
    validationImages.ReadFcn = @(loc)imresize(imread(loc),[224, 224]);
    
    
    % Load the pretrained GoogleNet network
    
    net = resnet101;
    net.Layers;
    
    % Extract the layer graph from the trained network and plot the layer graph.
    
    lgraph = layerGraph(net);
    Architecture_1 = figure('Units','normalized','Position',[0.1 0.1 0.8 0.8]);
    plot(lgraph);
    
       
    lgraph = removeLayers(lgraph, {'fc1000', 'prob', 'ClassificationLayer_predictions'});
    
    numClasses = numel(categories(trainingImages.Labels));
    newLayers = [
        fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',20,'BiasLearnRateFactor', 20)
        softmaxLayer('Name','softmax')
        classificationLayer('Name','classoutput')];
    lgraph = addLayers(lgraph,newLayers);
    
    lgraph = connectLayers(lgraph,'pool5','fc');
    
    Architecture_2 = figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
    plot(lgraph);
    ylim([0,10]);
    
    miniBatchSize = 10;
    
    %numIterationsPerEpoch = floor(numel(trainingImages.Labels)/miniBatchSize);
    options = trainingOptions('sgdm',...
        'ExecutionEnvironment','gpu',...
        'MiniBatchSize',miniBatchSize,...
        'MaxEpochs',100,...
        'InitialLearnRate',1e-5,...
        'VerboseFrequency',1,...
        'Plots','training-progress',...
        'ValidationData',validationImages,...
        'ValidationFrequency',5);
    netTransfer = trainNetwork(trainingImages,lgraph,options);
    
    %% Classify the validation images using the fine-tuned network.
    
    [predictedLabels, scores] = classify(netTransfer,validationImages,'ExecutionEnvironment','cpu');
    accuracy = mean(predictedLabels == validationImages.Labels);
    targets = grp2idx(validationImages.Labels);
    targets(targets==2) = 0;
    targets = ~targets;
    outputs = grp2idx(predictedLabels);
    outputs(outputs==2) = 0;
    outputs = ~outputs;
    [c,cm,ind,per] = confusion(targets', outputs');
    
    Confusion_Matrix=figure; plotconfusion(targets', outputs');
    
    %%
    
    addpath(genpath(resultsFolder))
    cp = my_classperf(cm);
    ROC_1=figure; plotroc(targets', scores(:,2)');
    
    [X,Y,T,AUC,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(targets', scores(:,2)',1,...
        'NBoot',1000,'TVals',0:0.05:1);
    ROC_2=figure; errorbar(X(:,3),Y(:,1),Y(:,1)-Y(:,2),Y(:,3)-Y(:,1));
    TheTableNice = table(validationImages.Labels,predictedLabels,'VariableNames',{'Actual','Predicted'});
    Confusion_Matrix1 = figure;
    PrettyCM = heatmap(TheTableNice,'Predicted','Actual');
    
    %     idx = randperm(20,4);
    %     Second_Figure = figure
    %     for i = 1:numel(idx)
    %         subplot(2,2,i)
    %         I = readimage(validationImages,idx(i));
    %         label = predictedLabels(idx(i));
    %         imshow(I)
    %         title(char(label))
    %     end
    
    try
        msg=sprintf('%s loop %d from %d MiniBatchSize %d with the results %d', Name, qq, nLoops, miniBatchSize, cp);
        sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
    catch
    end
    cd(resultsFolder);
    if  cp.Se > 0.79
        
        saveas(PrettyCM, ['BalCl_MB10_PrettyCM_' num2str(qq) '.jpg']);
        
        saveas(ROC_1, ['BalCl_MB10_ROC_1_' num2str(qq) '.jpg']);
        saveas(ROC_2, ['BalCl_MB10_ROC_2_' num2str(qq) '.jpg']);
        
        TrainingNeuNetw = findall(groot, 'Type', 'Figure');
        
        save(['RN_BalCl_MinBat10_' num2str(qq) '.mat'],'accuracy', 'AUC',...
            'c', 'cm', 'Confusion_Matrix', 'cp', ...
            'ind', 'miniBatchSize', 'netTransfer',...
            'numClasses', 'OPTROCPT', 'outputs', 'per', 'predictedLabels',...
            'scores','SUBY', 'SUBYNAMES', 'T', 'targets',...
            'TrainingNeuNetw','TheTableNice', 'validationImages', 'X', 'Y');
        %saveas(Architecture_1, ['UnCl_MB10_Architecture_1_' num2str(qq) '.jpg']);
        %saveas(Architecture_2, ['UnCl_MB10_Architecture_2_' num2str(qq) '.jpg']);
        
        %saveas(Second_Figure, ['UnCl_MB10_2nd_Figure_' num2str(qq) '.jpg']);
        
        
        
        try
            msg=sprintf('%s loop %d from %d MiniBatchSize %d , Sensitivity = %d , Balanced Accuracy = %d',...
                Name, qq, nLoops, miniBatchSize, cp.Se, cp.BA);
            sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
        catch
        end
        
%         close all
        % Save the figure of Training Progress in PNG format
        % saveas(TrainingNeuNetw,['RN_UnCl_MinBat10_' num2str(qq) '.png']);
        
        
%     else
%         save(['RN_BalCl_MinBat10_' num2str(qq) '.mat'],'accuracy', 'AUC',...
%             'cm', 'Confusion_Matrix', 'cp',...
%             'outputs', 'per', 'scores','SUBY', 'SUBYNAMES', 'T', 'targets',...
%             'TheTableNice', 'validationImages', 'X', 'Y');
%         saveas(PrettyCM, ['BalCl_MB10_PrettyCM_' num2str(qq) '.jpg']);
%         
%         try
%             msg=sprintf('%s loop %d from %d MiniBatchSize %d , Another one bits the dust, Sensitivity = %d',...
%                 Name, qq, nLoops, miniBatchSize, cp.Se);
%             sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
%         catch
%         end
        % Close all the figures
        
    end
    
    close all
    delete(findall(0));
    cd(codeFolder);
    % clearvars -except codeFolder resultsFolder TrainingFolder ValidationFolder nLoops qq
    
end

% Close progress bar
ioi_text_waitbar('Clear');

%%
nLoops = 1310;

for qq = 1301:nLoops
    
      
    ioi_text_waitbar(qq/nLoops, sprintf('ResNet101 Balanced MiniBatch=10, Running loop %d from %d', qq, 1440))
    
    close all
    trainingImages = imageDatastore(TrainingFolder, ...
        'IncludeSubfolders',true,...
        'FileExtensions','.tif','LabelSource','foldernames');
    
    trainingImages.ReadFcn = @(loc)imresize(imread(loc),[224, 224]);
    
    validationImages = imageDatastore(ValidationFolder, ...
        'IncludeSubfolders',true,...
        'FileExtensions','.tif','LabelSource','foldernames');
    
    validationImages.ReadFcn = @(loc)imresize(imread(loc),[224, 224]);
    
    
    % Load the pretrained GoogleNet network
    
    net = resnet101;
    net.Layers;
    
    % Extract the layer graph from the trained network and plot the layer graph.
    
    lgraph = layerGraph(net);
    Architecture_1 = figure('Units','normalized','Position',[0.1 0.1 0.8 0.8]);
    plot(lgraph);
    
       
    lgraph = removeLayers(lgraph, {'fc1000', 'prob', 'ClassificationLayer_predictions'});
    
    numClasses = numel(categories(trainingImages.Labels));
    newLayers = [
        fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',20,'BiasLearnRateFactor', 40)
        softmaxLayer('Name','softmax')
        classificationLayer('Name','classoutput')];
    lgraph = addLayers(lgraph,newLayers);
    
    lgraph = connectLayers(lgraph,'pool5','fc');
    
    Architecture_2 = figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
    plot(lgraph);
    ylim([0,10]);
    
    miniBatchSize = 10;
    
    %numIterationsPerEpoch = floor(numel(trainingImages.Labels)/miniBatchSize);
    options = trainingOptions('sgdm',...
        'ExecutionEnvironment','gpu',...
        'MiniBatchSize',miniBatchSize,...
        'MaxEpochs',100,...
        'InitialLearnRate',1e-5,...
        'VerboseFrequency',1,...
        'Plots','training-progress',...
        'ValidationData',validationImages,...
        'ValidationFrequency',5);
    netTransfer = trainNetwork(trainingImages,lgraph,options);
    
    %% Classify the validation images using the fine-tuned network.
    
    [predictedLabels, scores] = classify(netTransfer,validationImages,'ExecutionEnvironment','cpu');
    accuracy = mean(predictedLabels == validationImages.Labels);
    targets = grp2idx(validationImages.Labels);
    targets(targets==2) = 0;
    targets = ~targets;
    outputs = grp2idx(predictedLabels);
    outputs(outputs==2) = 0;
    outputs = ~outputs;
    [c,cm,ind,per] = confusion(targets', outputs');
    
    Confusion_Matrix=figure; plotconfusion(targets', outputs');
    
    %%
    
    addpath(genpath(resultsFolder))
    cp = my_classperf(cm);
    ROC_1=figure; plotroc(targets', scores(:,2)');
    
    [X,Y,T,AUC,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(targets', scores(:,2)',1,...
        'NBoot',1000,'TVals',0:0.05:1);
    ROC_2=figure; errorbar(X(:,3),Y(:,1),Y(:,1)-Y(:,2),Y(:,3)-Y(:,1));
    TheTableNice = table(validationImages.Labels,predictedLabels,'VariableNames',{'Actual','Predicted'});
    Confusion_Matrix1 = figure;
    PrettyCM = heatmap(TheTableNice,'Predicted','Actual');
    
    %     idx = randperm(20,4);
    %     Second_Figure = figure
    %     for i = 1:numel(idx)
    %         subplot(2,2,i)
    %         I = readimage(validationImages,idx(i));
    %         label = predictedLabels(idx(i));
    %         imshow(I)
    %         title(char(label))
    %     end
    
    try
        msg=sprintf('%s loop %d from %d MiniBatchSize %d with the results %d', Name, qq, nLoops, miniBatchSize, cp);
        sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
    catch
    end
    cd(resultsFolder);
    if  cp.Se > 0.79
        
        saveas(PrettyCM, ['BalCl_MB10_PrettyCM_' num2str(qq) '.jpg']);
        
        saveas(ROC_1, ['BalCl_MB10_ROC_1_' num2str(qq) '.jpg']);
        saveas(ROC_2, ['BalCl_MB10_ROC_2_' num2str(qq) '.jpg']);
        
        TrainingNeuNetw = findall(groot, 'Type', 'Figure');
        
        save(['RN_BalCl_MinBat10_' num2str(qq) '.mat'],'accuracy', 'AUC',...
            'c', 'cm', 'Confusion_Matrix', 'cp', ...
            'ind', 'miniBatchSize', 'netTransfer',...
            'numClasses', 'OPTROCPT', 'outputs', 'per', 'predictedLabels',...
            'scores','SUBY', 'SUBYNAMES', 'T', 'targets',...
            'TrainingNeuNetw','TheTableNice', 'validationImages', 'X', 'Y');
        %saveas(Architecture_1, ['UnCl_MB10_Architecture_1_' num2str(qq) '.jpg']);
        %saveas(Architecture_2, ['UnCl_MB10_Architecture_2_' num2str(qq) '.jpg']);
        
        %saveas(Second_Figure, ['UnCl_MB10_2nd_Figure_' num2str(qq) '.jpg']);
        
        
        
        try
            msg=sprintf('%s loop %d from %d MiniBatchSize %d , Sensitivity = %d , Balanced Accuracy = %d',...
                Name, qq, nLoops, miniBatchSize, cp.Se, cp.BA);
            sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
        catch
        end
        
%         close all
        % Save the figure of Training Progress in PNG format
        % saveas(TrainingNeuNetw,['RN_UnCl_MinBat10_' num2str(qq) '.png']);
        
        
%     else
%         save(['RN_BalCl_MinBat10_' num2str(qq) '.mat'],'accuracy', 'AUC',...
%             'cm', 'Confusion_Matrix', 'cp',...
%             'outputs', 'per', 'scores','SUBY', 'SUBYNAMES', 'T', 'targets',...
%             'TheTableNice', 'validationImages', 'X', 'Y');
%         saveas(PrettyCM, ['BalCl_MB10_PrettyCM_' num2str(qq) '.jpg']);
%         
%         try
%             msg=sprintf('%s loop %d from %d MiniBatchSize %d , Another one bits the dust, Sensitivity = %d',...
%                 Name, qq, nLoops, miniBatchSize, cp.Se);
%             sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
%         catch
%         end
        % Close all the figures
        
    end
    
    close all
    delete(findall(0));
    cd(codeFolder);
    % clearvars -except codeFolder resultsFolder TrainingFolder ValidationFolder nLoops qq
    
end

% Close progress bar
ioi_text_waitbar('Clear');


nLoops = 1320;

for qq = 1311:nLoops
    
      
    ioi_text_waitbar(qq/nLoops, sprintf('ResNet101 Balanced MiniBatch=10, Running loop %d from %d', qq, 1440))
    
    close all
    trainingImages = imageDatastore(TrainingFolder, ...
        'IncludeSubfolders',true,...
        'FileExtensions','.tif','LabelSource','foldernames');
    
    trainingImages.ReadFcn = @(loc)imresize(imread(loc),[224, 224]);
    
    validationImages = imageDatastore(ValidationFolder, ...
        'IncludeSubfolders',true,...
        'FileExtensions','.tif','LabelSource','foldernames');
    
    validationImages.ReadFcn = @(loc)imresize(imread(loc),[224, 224]);
    
    
    % Load the pretrained GoogleNet network
    
    net = resnet101;
    net.Layers;
    
    % Extract the layer graph from the trained network and plot the layer graph.
    
    lgraph = layerGraph(net);
    Architecture_1 = figure('Units','normalized','Position',[0.1 0.1 0.8 0.8]);
    plot(lgraph);
    
       
    lgraph = removeLayers(lgraph, {'fc1000', 'prob', 'ClassificationLayer_predictions'});
    
    numClasses = numel(categories(trainingImages.Labels));
    newLayers = [
        fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',20,'BiasLearnRateFactor', 60)
        softmaxLayer('Name','softmax')
        classificationLayer('Name','classoutput')];
    lgraph = addLayers(lgraph,newLayers);
    
    lgraph = connectLayers(lgraph,'pool5','fc');
    
    Architecture_2 = figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
    plot(lgraph);
    ylim([0,10]);
    
    miniBatchSize = 10;
    
    %numIterationsPerEpoch = floor(numel(trainingImages.Labels)/miniBatchSize);
    options = trainingOptions('sgdm',...
        'ExecutionEnvironment','gpu',...
        'MiniBatchSize',miniBatchSize,...
        'MaxEpochs',100,...
        'InitialLearnRate',1e-5,...
        'VerboseFrequency',1,...
        'Plots','training-progress',...
        'ValidationData',validationImages,...
        'ValidationFrequency',5);
    netTransfer = trainNetwork(trainingImages,lgraph,options);
    
    %% Classify the validation images using the fine-tuned network.
    
    [predictedLabels, scores] = classify(netTransfer,validationImages,'ExecutionEnvironment','cpu');
    accuracy = mean(predictedLabels == validationImages.Labels);
    targets = grp2idx(validationImages.Labels);
    targets(targets==2) = 0;
    targets = ~targets;
    outputs = grp2idx(predictedLabels);
    outputs(outputs==2) = 0;
    outputs = ~outputs;
    [c,cm,ind,per] = confusion(targets', outputs');
    
    Confusion_Matrix=figure; plotconfusion(targets', outputs');
    
    %%
    
    addpath(genpath(resultsFolder))
    cp = my_classperf(cm);
    ROC_1=figure; plotroc(targets', scores(:,2)');
    
    [X,Y,T,AUC,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(targets', scores(:,2)',1,...
        'NBoot',1000,'TVals',0:0.05:1);
    ROC_2=figure; errorbar(X(:,3),Y(:,1),Y(:,1)-Y(:,2),Y(:,3)-Y(:,1));
    TheTableNice = table(validationImages.Labels,predictedLabels,'VariableNames',{'Actual','Predicted'});
    Confusion_Matrix1 = figure;
    PrettyCM = heatmap(TheTableNice,'Predicted','Actual');
    
    %     idx = randperm(20,4);
    %     Second_Figure = figure
    %     for i = 1:numel(idx)
    %         subplot(2,2,i)
    %         I = readimage(validationImages,idx(i));
    %         label = predictedLabels(idx(i));
    %         imshow(I)
    %         title(char(label))
    %     end
    
    try
        msg=sprintf('%s loop %d from %d MiniBatchSize %d with the results %d', Name, qq, nLoops, miniBatchSize, cp);
        sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
    catch
    end
    cd(resultsFolder);
    if  cp.Se > 0.79
        
        saveas(PrettyCM, ['BalCl_MB10_PrettyCM_' num2str(qq) '.jpg']);
        
        saveas(ROC_1, ['BalCl_MB10_ROC_1_' num2str(qq) '.jpg']);
        saveas(ROC_2, ['BalCl_MB10_ROC_2_' num2str(qq) '.jpg']);
        
        TrainingNeuNetw = findall(groot, 'Type', 'Figure');
        
        save(['RN_BalCl_MinBat10_' num2str(qq) '.mat'],'accuracy', 'AUC',...
            'c', 'cm', 'Confusion_Matrix', 'cp', ...
            'ind', 'miniBatchSize', 'netTransfer',...
            'numClasses', 'OPTROCPT', 'outputs', 'per', 'predictedLabels',...
            'scores','SUBY', 'SUBYNAMES', 'T', 'targets',...
            'TrainingNeuNetw','TheTableNice', 'validationImages', 'X', 'Y');
        %saveas(Architecture_1, ['UnCl_MB10_Architecture_1_' num2str(qq) '.jpg']);
        %saveas(Architecture_2, ['UnCl_MB10_Architecture_2_' num2str(qq) '.jpg']);
        
        %saveas(Second_Figure, ['UnCl_MB10_2nd_Figure_' num2str(qq) '.jpg']);
        
        
        
        try
            msg=sprintf('%s loop %d from %d MiniBatchSize %d , Sensitivity = %d , Balanced Accuracy = %d',...
                Name, qq, nLoops, miniBatchSize, cp.Se, cp.BA);
            sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
        catch
        end
        
%         close all
        % Save the figure of Training Progress in PNG format
        % saveas(TrainingNeuNetw,['RN_UnCl_MinBat10_' num2str(qq) '.png']);
        
        
%     else
%         save(['RN_BalCl_MinBat10_' num2str(qq) '.mat'],'accuracy', 'AUC',...
%             'cm', 'Confusion_Matrix', 'cp',...
%             'outputs', 'per', 'scores','SUBY', 'SUBYNAMES', 'T', 'targets',...
%             'TheTableNice', 'validationImages', 'X', 'Y');
%         saveas(PrettyCM, ['BalCl_MB10_PrettyCM_' num2str(qq) '.jpg']);
%         
%         try
%             msg=sprintf('%s loop %d from %d MiniBatchSize %d , Another one bits the dust, Sensitivity = %d',...
%                 Name, qq, nLoops, miniBatchSize, cp.Se);
%             sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
%         catch
%         end
        % Close all the figures
        
    end
    
    close all
    delete(findall(0));
    cd(codeFolder);
    % clearvars -except codeFolder resultsFolder TrainingFolder ValidationFolder nLoops qq
    
end

% Close progress bar
ioi_text_waitbar('Clear');

% *ResNet 101 for Database of Brasil, IJC & ISSSTE*

% Unbalanced Class
% MiniBatch = 10

% Analysis of three databases is realized for compare the results, only the
% results that has a Sensitivity bigger than 80% is saved

clear; close all; clc;

% Folders Location
codeFolder = 'C:\Users\LANCYTT\Documents\Juan Carlos Torres\Thermograms Analysis';
% addpath('I:\Archivos Juan Carlos Torres\MATLAB\MATLAB Codigos');

resultsFolder = 'C:\Users\LANCYTT\Documents\Juan Carlos Torres\Results\ResNet101\Iteraciones\Balanced';
addpath('C:\Users\LANCYTT\Documents\Juan Carlos Torres\Thermograms Analysis');

TrainingFolder = 'C:\Users\LANCYTT\Documents\Juan Carlos Torres\Data Augmented Thermography\Data Augmented Bra+IJC+ISSSTE\Balanced Augmented Modify';
ValidationFolder = 'C:\Users\LANCYTT\Documents\Juan Carlos Torres\Data Augmented Thermography\Train - Valid Bra+IJC+ISSSTE\Validation Balanced Class';

pat_sendmail

ioi_text_waitbar(0, 'Running ResNet 101, Balanced MiniBatch 10... Please wait...');

Name=sprintf('ResNet101 Balanced');
nLoops = 1330;

for qq = 1321:nLoops
    
      
    ioi_text_waitbar(qq/nLoops, sprintf('ResNet101 Balanced MiniBatch=10, Running loop %d from %d', qq, 1440))
    
    close all
    trainingImages = imageDatastore(TrainingFolder, ...
        'IncludeSubfolders',true,...
        'FileExtensions','.tif','LabelSource','foldernames');
    
    trainingImages.ReadFcn = @(loc)imresize(imread(loc),[224, 224]);
    
    validationImages = imageDatastore(ValidationFolder, ...
        'IncludeSubfolders',true,...
        'FileExtensions','.tif','LabelSource','foldernames');
    
    validationImages.ReadFcn = @(loc)imresize(imread(loc),[224, 224]);
    
    
    % Load the pretrained GoogleNet network
    
    net = resnet101;
    net.Layers;
    
    % Extract the layer graph from the trained network and plot the layer graph.
    
    lgraph = layerGraph(net);
    Architecture_1 = figure('Units','normalized','Position',[0.1 0.1 0.8 0.8]);
    plot(lgraph);
    
       
    lgraph = removeLayers(lgraph, {'fc1000', 'prob', 'ClassificationLayer_predictions'});
    
    numClasses = numel(categories(trainingImages.Labels));
    newLayers = [
        fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',40,'BiasLearnRateFactor', 1)
        softmaxLayer('Name','softmax')
        classificationLayer('Name','classoutput')];
    lgraph = addLayers(lgraph,newLayers);
    
    lgraph = connectLayers(lgraph,'pool5','fc');
    
    Architecture_2 = figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
    plot(lgraph);
    ylim([0,10]);
    
    miniBatchSize = 10;
    
    %numIterationsPerEpoch = floor(numel(trainingImages.Labels)/miniBatchSize);
    options = trainingOptions('sgdm',...
        'ExecutionEnvironment','gpu',...
        'MiniBatchSize',miniBatchSize,...
        'MaxEpochs',100,...
        'InitialLearnRate',1e-5,...
        'VerboseFrequency',1,...
        'Plots','training-progress',...
        'ValidationData',validationImages,...
        'ValidationFrequency',5);
    netTransfer = trainNetwork(trainingImages,lgraph,options);
    
    %% Classify the validation images using the fine-tuned network.
    
    [predictedLabels, scores] = classify(netTransfer,validationImages,'ExecutionEnvironment','cpu');
    accuracy = mean(predictedLabels == validationImages.Labels);
    targets = grp2idx(validationImages.Labels);
    targets(targets==2) = 0;
    targets = ~targets;
    outputs = grp2idx(predictedLabels);
    outputs(outputs==2) = 0;
    outputs = ~outputs;
    [c,cm,ind,per] = confusion(targets', outputs');
    
    Confusion_Matrix=figure; plotconfusion(targets', outputs');
    
    %%
    
    addpath(genpath(resultsFolder))
    cp = my_classperf(cm);
    ROC_1=figure; plotroc(targets', scores(:,2)');
    
    [X,Y,T,AUC,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(targets', scores(:,2)',1,...
        'NBoot',1000,'TVals',0:0.05:1);
    ROC_2=figure; errorbar(X(:,3),Y(:,1),Y(:,1)-Y(:,2),Y(:,3)-Y(:,1));
    TheTableNice = table(validationImages.Labels,predictedLabels,'VariableNames',{'Actual','Predicted'});
    Confusion_Matrix1 = figure;
    PrettyCM = heatmap(TheTableNice,'Predicted','Actual');
    
    %     idx = randperm(20,4);
    %     Second_Figure = figure
    %     for i = 1:numel(idx)
    %         subplot(2,2,i)
    %         I = readimage(validationImages,idx(i));
    %         label = predictedLabels(idx(i));
    %         imshow(I)
    %         title(char(label))
    %     end
    
    try
        msg=sprintf('%s loop %d from %d MiniBatchSize %d with the results %d', Name, qq, nLoops, miniBatchSize, cp);
        sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
    catch
    end
    cd(resultsFolder);
    if  cp.Se > 0.79
        
        saveas(PrettyCM, ['BalCl_MB10_PrettyCM_' num2str(qq) '.jpg']);
        
        saveas(ROC_1, ['BalCl_MB10_ROC_1_' num2str(qq) '.jpg']);
        saveas(ROC_2, ['BalCl_MB10_ROC_2_' num2str(qq) '.jpg']);
        
        TrainingNeuNetw = findall(groot, 'Type', 'Figure');
        
        save(['RN_BalCl_MinBat10_' num2str(qq) '.mat'],'accuracy', 'AUC',...
            'c', 'cm', 'Confusion_Matrix', 'cp', ...
            'ind', 'miniBatchSize', 'netTransfer',...
            'numClasses', 'OPTROCPT', 'outputs', 'per', 'predictedLabels',...
            'scores','SUBY', 'SUBYNAMES', 'T', 'targets',...
            'TrainingNeuNetw','TheTableNice', 'validationImages', 'X', 'Y');
        %saveas(Architecture_1, ['UnCl_MB10_Architecture_1_' num2str(qq) '.jpg']);
        %saveas(Architecture_2, ['UnCl_MB10_Architecture_2_' num2str(qq) '.jpg']);
        
        %saveas(Second_Figure, ['UnCl_MB10_2nd_Figure_' num2str(qq) '.jpg']);
        
        
        
        try
            msg=sprintf('%s loop %d from %d MiniBatchSize %d , Sensitivity = %d , Balanced Accuracy = %d',...
                Name, qq, nLoops, miniBatchSize, cp.Se, cp.BA);
            sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
        catch
        end
        
%         close all
        % Save the figure of Training Progress in PNG format
        % saveas(TrainingNeuNetw,['RN_UnCl_MinBat10_' num2str(qq) '.png']);
        
        
%     else
%         save(['RN_BalCl_MinBat10_' num2str(qq) '.mat'],'accuracy', 'AUC',...
%             'cm', 'Confusion_Matrix', 'cp',...
%             'outputs', 'per', 'scores','SUBY', 'SUBYNAMES', 'T', 'targets',...
%             'TheTableNice', 'validationImages', 'X', 'Y');
%         saveas(PrettyCM, ['BalCl_MB10_PrettyCM_' num2str(qq) '.jpg']);
%         
%         try
%             msg=sprintf('%s loop %d from %d MiniBatchSize %d , Another one bits the dust, Sensitivity = %d',...
%                 Name, qq, nLoops, miniBatchSize, cp.Se);
%             sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
%         catch
%         end
        % Close all the figures
        
    end
    
    close all
    delete(findall(0));
    cd(codeFolder);
    % clearvars -except codeFolder resultsFolder TrainingFolder ValidationFolder nLoops qq
    
end

% Close progress bar
ioi_text_waitbar('Clear');

nLoops = 1340;

for qq = 1331:nLoops
    
      
    ioi_text_waitbar(qq/nLoops, sprintf('ResNet101 Balanced MiniBatch=10, Running loop %d from %d', qq, 1440))
    
    close all
    trainingImages = imageDatastore(TrainingFolder, ...
        'IncludeSubfolders',true,...
        'FileExtensions','.tif','LabelSource','foldernames');
    
    trainingImages.ReadFcn = @(loc)imresize(imread(loc),[224, 224]);
    
    validationImages = imageDatastore(ValidationFolder, ...
        'IncludeSubfolders',true,...
        'FileExtensions','.tif','LabelSource','foldernames');
    
    validationImages.ReadFcn = @(loc)imresize(imread(loc),[224, 224]);
    
    
    % Load the pretrained GoogleNet network
    
    net = resnet101;
    net.Layers;
    
    % Extract the layer graph from the trained network and plot the layer graph.
    
    lgraph = layerGraph(net);
    Architecture_1 = figure('Units','normalized','Position',[0.1 0.1 0.8 0.8]);
    plot(lgraph);
    
       
    lgraph = removeLayers(lgraph, {'fc1000', 'prob', 'ClassificationLayer_predictions'});
    
    numClasses = numel(categories(trainingImages.Labels));
    newLayers = [
        fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',40,'BiasLearnRateFactor', 5)
        softmaxLayer('Name','softmax')
        classificationLayer('Name','classoutput')];
    lgraph = addLayers(lgraph,newLayers);
    
    lgraph = connectLayers(lgraph,'pool5','fc');
    
    Architecture_2 = figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
    plot(lgraph);
    ylim([0,10]);
    
    miniBatchSize = 10;
    
    %numIterationsPerEpoch = floor(numel(trainingImages.Labels)/miniBatchSize);
    options = trainingOptions('sgdm',...
        'ExecutionEnvironment','gpu',...
        'MiniBatchSize',miniBatchSize,...
        'MaxEpochs',100,...
        'InitialLearnRate',1e-5,...
        'VerboseFrequency',1,...
        'Plots','training-progress',...
        'ValidationData',validationImages,...
        'ValidationFrequency',5);
    netTransfer = trainNetwork(trainingImages,lgraph,options);
    
    %% Classify the validation images using the fine-tuned network.
    
    [predictedLabels, scores] = classify(netTransfer,validationImages,'ExecutionEnvironment','cpu');
    accuracy = mean(predictedLabels == validationImages.Labels);
    targets = grp2idx(validationImages.Labels);
    targets(targets==2) = 0;
    targets = ~targets;
    outputs = grp2idx(predictedLabels);
    outputs(outputs==2) = 0;
    outputs = ~outputs;
    [c,cm,ind,per] = confusion(targets', outputs');
    
    Confusion_Matrix=figure; plotconfusion(targets', outputs');
    
    %%
    
    addpath(genpath(resultsFolder))
    cp = my_classperf(cm);
    ROC_1=figure; plotroc(targets', scores(:,2)');
    
    [X,Y,T,AUC,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(targets', scores(:,2)',1,...
        'NBoot',1000,'TVals',0:0.05:1);
    ROC_2=figure; errorbar(X(:,3),Y(:,1),Y(:,1)-Y(:,2),Y(:,3)-Y(:,1));
    TheTableNice = table(validationImages.Labels,predictedLabels,'VariableNames',{'Actual','Predicted'});
    Confusion_Matrix1 = figure;
    PrettyCM = heatmap(TheTableNice,'Predicted','Actual');
    
    %     idx = randperm(20,4);
    %     Second_Figure = figure
    %     for i = 1:numel(idx)
    %         subplot(2,2,i)
    %         I = readimage(validationImages,idx(i));
    %         label = predictedLabels(idx(i));
    %         imshow(I)
    %         title(char(label))
    %     end
    
    try
        msg=sprintf('%s loop %d from %d MiniBatchSize %d with the results %d', Name, qq, nLoops, miniBatchSize, cp);
        sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
    catch
    end
    cd(resultsFolder);
    if  cp.Se > 0.79
        
        saveas(PrettyCM, ['BalCl_MB10_PrettyCM_' num2str(qq) '.jpg']);
        
        saveas(ROC_1, ['BalCl_MB10_ROC_1_' num2str(qq) '.jpg']);
        saveas(ROC_2, ['BalCl_MB10_ROC_2_' num2str(qq) '.jpg']);
        
        TrainingNeuNetw = findall(groot, 'Type', 'Figure');
        
        save(['RN_BalCl_MinBat10_' num2str(qq) '.mat'],'accuracy', 'AUC',...
            'c', 'cm', 'Confusion_Matrix', 'cp', ...
            'ind', 'miniBatchSize', 'netTransfer',...
            'numClasses', 'OPTROCPT', 'outputs', 'per', 'predictedLabels',...
            'scores','SUBY', 'SUBYNAMES', 'T', 'targets',...
            'TrainingNeuNetw','TheTableNice', 'validationImages', 'X', 'Y');
        %saveas(Architecture_1, ['UnCl_MB10_Architecture_1_' num2str(qq) '.jpg']);
        %saveas(Architecture_2, ['UnCl_MB10_Architecture_2_' num2str(qq) '.jpg']);
        
        %saveas(Second_Figure, ['UnCl_MB10_2nd_Figure_' num2str(qq) '.jpg']);
        
        
        
        try
            msg=sprintf('%s loop %d from %d MiniBatchSize %d , Sensitivity = %d , Balanced Accuracy = %d',...
                Name, qq, nLoops, miniBatchSize, cp.Se, cp.BA);
            sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
        catch
        end
        
%         close all
        % Save the figure of Training Progress in PNG format
        % saveas(TrainingNeuNetw,['RN_UnCl_MinBat10_' num2str(qq) '.png']);
        
        
%     else
%         save(['RN_BalCl_MinBat10_' num2str(qq) '.mat'],'accuracy', 'AUC',...
%             'cm', 'Confusion_Matrix', 'cp',...
%             'outputs', 'per', 'scores','SUBY', 'SUBYNAMES', 'T', 'targets',...
%             'TheTableNice', 'validationImages', 'X', 'Y');
%         saveas(PrettyCM, ['BalCl_MB10_PrettyCM_' num2str(qq) '.jpg']);
%         
%         try
%             msg=sprintf('%s loop %d from %d MiniBatchSize %d , Another one bits the dust, Sensitivity = %d',...
%                 Name, qq, nLoops, miniBatchSize, cp.Se);
%             sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
%         catch
%         end
        % Close all the figures
        
    end
    
    close all
    delete(findall(0));
    cd(codeFolder);
    % clearvars -except codeFolder resultsFolder TrainingFolder ValidationFolder nLoops qq
    
end

% Close progress bar
ioi_text_waitbar('Clear');

%%
nLoops = 1350;

for qq = 1341:nLoops
    
      
    ioi_text_waitbar(qq/nLoops, sprintf('ResNet101 Balanced MiniBatch=10, Running loop %d from %d', qq, 1440))
    
    close all
    trainingImages = imageDatastore(TrainingFolder, ...
        'IncludeSubfolders',true,...
        'FileExtensions','.tif','LabelSource','foldernames');
    
    trainingImages.ReadFcn = @(loc)imresize(imread(loc),[224, 224]);
    
    validationImages = imageDatastore(ValidationFolder, ...
        'IncludeSubfolders',true,...
        'FileExtensions','.tif','LabelSource','foldernames');
    
    validationImages.ReadFcn = @(loc)imresize(imread(loc),[224, 224]);
    
    
    % Load the pretrained GoogleNet network
    
    net = resnet101;
    net.Layers;
    
    % Extract the layer graph from the trained network and plot the layer graph.
    
    lgraph = layerGraph(net);
    Architecture_1 = figure('Units','normalized','Position',[0.1 0.1 0.8 0.8]);
    plot(lgraph);
    
       
    lgraph = removeLayers(lgraph, {'fc1000', 'prob', 'ClassificationLayer_predictions'});
    
    numClasses = numel(categories(trainingImages.Labels));
    newLayers = [
        fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',40,'BiasLearnRateFactor', 10)
        softmaxLayer('Name','softmax')
        classificationLayer('Name','classoutput')];
    lgraph = addLayers(lgraph,newLayers);
    
    lgraph = connectLayers(lgraph,'pool5','fc');
    
    Architecture_2 = figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
    plot(lgraph);
    ylim([0,10]);
    
    miniBatchSize = 10;
    
    %numIterationsPerEpoch = floor(numel(trainingImages.Labels)/miniBatchSize);
    options = trainingOptions('sgdm',...
        'ExecutionEnvironment','gpu',...
        'MiniBatchSize',miniBatchSize,...
        'MaxEpochs',100,...
        'InitialLearnRate',1e-5,...
        'VerboseFrequency',1,...
        'Plots','training-progress',...
        'ValidationData',validationImages,...
        'ValidationFrequency',5);
    netTransfer = trainNetwork(trainingImages,lgraph,options);
    
    %% Classify the validation images using the fine-tuned network.
    
    [predictedLabels, scores] = classify(netTransfer,validationImages,'ExecutionEnvironment','cpu');
    accuracy = mean(predictedLabels == validationImages.Labels);
    targets = grp2idx(validationImages.Labels);
    targets(targets==2) = 0;
    targets = ~targets;
    outputs = grp2idx(predictedLabels);
    outputs(outputs==2) = 0;
    outputs = ~outputs;
    [c,cm,ind,per] = confusion(targets', outputs');
    
    Confusion_Matrix=figure; plotconfusion(targets', outputs');
    
    %%
    
    addpath(genpath(resultsFolder))
    cp = my_classperf(cm);
    ROC_1=figure; plotroc(targets', scores(:,2)');
    
    [X,Y,T,AUC,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(targets', scores(:,2)',1,...
        'NBoot',1000,'TVals',0:0.05:1);
    ROC_2=figure; errorbar(X(:,3),Y(:,1),Y(:,1)-Y(:,2),Y(:,3)-Y(:,1));
    TheTableNice = table(validationImages.Labels,predictedLabels,'VariableNames',{'Actual','Predicted'});
    Confusion_Matrix1 = figure;
    PrettyCM = heatmap(TheTableNice,'Predicted','Actual');
    
    %     idx = randperm(20,4);
    %     Second_Figure = figure
    %     for i = 1:numel(idx)
    %         subplot(2,2,i)
    %         I = readimage(validationImages,idx(i));
    %         label = predictedLabels(idx(i));
    %         imshow(I)
    %         title(char(label))
    %     end
    
    try
        msg=sprintf('%s loop %d from %d MiniBatchSize %d with the results %d', Name, qq, nLoops, miniBatchSize, cp);
        sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
    catch
    end
    cd(resultsFolder);
    if  cp.Se > 0.79
        
        saveas(PrettyCM, ['BalCl_MB10_PrettyCM_' num2str(qq) '.jpg']);
        
        saveas(ROC_1, ['BalCl_MB10_ROC_1_' num2str(qq) '.jpg']);
        saveas(ROC_2, ['BalCl_MB10_ROC_2_' num2str(qq) '.jpg']);
        
        TrainingNeuNetw = findall(groot, 'Type', 'Figure');
        
        save(['RN_BalCl_MinBat10_' num2str(qq) '.mat'],'accuracy', 'AUC',...
            'c', 'cm', 'Confusion_Matrix', 'cp', ...
            'ind', 'miniBatchSize', 'netTransfer',...
            'numClasses', 'OPTROCPT', 'outputs', 'per', 'predictedLabels',...
            'scores','SUBY', 'SUBYNAMES', 'T', 'targets',...
            'TrainingNeuNetw','TheTableNice', 'validationImages', 'X', 'Y');
        %saveas(Architecture_1, ['UnCl_MB10_Architecture_1_' num2str(qq) '.jpg']);
        %saveas(Architecture_2, ['UnCl_MB10_Architecture_2_' num2str(qq) '.jpg']);
        
        %saveas(Second_Figure, ['UnCl_MB10_2nd_Figure_' num2str(qq) '.jpg']);
        
        
        
        try
            msg=sprintf('%s loop %d from %d MiniBatchSize %d , Sensitivity = %d , Balanced Accuracy = %d',...
                Name, qq, nLoops, miniBatchSize, cp.Se, cp.BA);
            sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
        catch
        end
        
%         close all
        % Save the figure of Training Progress in PNG format
        % saveas(TrainingNeuNetw,['RN_UnCl_MinBat10_' num2str(qq) '.png']);
        
        
%     else
%         save(['RN_BalCl_MinBat10_' num2str(qq) '.mat'],'accuracy', 'AUC',...
%             'cm', 'Confusion_Matrix', 'cp',...
%             'outputs', 'per', 'scores','SUBY', 'SUBYNAMES', 'T', 'targets',...
%             'TheTableNice', 'validationImages', 'X', 'Y');
%         saveas(PrettyCM, ['BalCl_MB10_PrettyCM_' num2str(qq) '.jpg']);
%         
%         try
%             msg=sprintf('%s loop %d from %d MiniBatchSize %d , Another one bits the dust, Sensitivity = %d',...
%                 Name, qq, nLoops, miniBatchSize, cp.Se);
%             sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
%         catch
%         end
        % Close all the figures
        
    end
    
    close all
    delete(findall(0));
    cd(codeFolder);
    % clearvars -except codeFolder resultsFolder TrainingFolder ValidationFolder nLoops qq
    
end

% Close progress bar
ioi_text_waitbar('Clear');

%%
nLoops = 1360;

for qq = 1351:nLoops
    
      
    ioi_text_waitbar(qq/nLoops, sprintf('ResNet101 Balanced MiniBatch=10, Running loop %d from %d', qq, 1440))
    
    close all
    trainingImages = imageDatastore(TrainingFolder, ...
        'IncludeSubfolders',true,...
        'FileExtensions','.tif','LabelSource','foldernames');
    
    trainingImages.ReadFcn = @(loc)imresize(imread(loc),[224, 224]);
    
    validationImages = imageDatastore(ValidationFolder, ...
        'IncludeSubfolders',true,...
        'FileExtensions','.tif','LabelSource','foldernames');
    
    validationImages.ReadFcn = @(loc)imresize(imread(loc),[224, 224]);
    
    
    % Load the pretrained GoogleNet network
    
    net = resnet101;
    net.Layers;
    
    % Extract the layer graph from the trained network and plot the layer graph.
    
    lgraph = layerGraph(net);
    Architecture_1 = figure('Units','normalized','Position',[0.1 0.1 0.8 0.8]);
    plot(lgraph);
    
       
    lgraph = removeLayers(lgraph, {'fc1000', 'prob', 'ClassificationLayer_predictions'});
    
    numClasses = numel(categories(trainingImages.Labels));
    newLayers = [
        fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',40,'BiasLearnRateFactor', 20)
        softmaxLayer('Name','softmax')
        classificationLayer('Name','classoutput')];
    lgraph = addLayers(lgraph,newLayers);
    
    lgraph = connectLayers(lgraph,'pool5','fc');
    
    Architecture_2 = figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
    plot(lgraph);
    ylim([0,10]);
    
    miniBatchSize = 10;
    
    %numIterationsPerEpoch = floor(numel(trainingImages.Labels)/miniBatchSize);
    options = trainingOptions('sgdm',...
        'ExecutionEnvironment','gpu',...
        'MiniBatchSize',miniBatchSize,...
        'MaxEpochs',100,...
        'InitialLearnRate',1e-5,...
        'VerboseFrequency',1,...
        'Plots','training-progress',...
        'ValidationData',validationImages,...
        'ValidationFrequency',5);
    netTransfer = trainNetwork(trainingImages,lgraph,options);
    
    %% Classify the validation images using the fine-tuned network.
    
    [predictedLabels, scores] = classify(netTransfer,validationImages,'ExecutionEnvironment','cpu');
    accuracy = mean(predictedLabels == validationImages.Labels);
    targets = grp2idx(validationImages.Labels);
    targets(targets==2) = 0;
    targets = ~targets;
    outputs = grp2idx(predictedLabels);
    outputs(outputs==2) = 0;
    outputs = ~outputs;
    [c,cm,ind,per] = confusion(targets', outputs');
    
    Confusion_Matrix=figure; plotconfusion(targets', outputs');
    
    %%
    
    addpath(genpath(resultsFolder))
    cp = my_classperf(cm);
    ROC_1=figure; plotroc(targets', scores(:,2)');
    
    [X,Y,T,AUC,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(targets', scores(:,2)',1,...
        'NBoot',1000,'TVals',0:0.05:1);
    ROC_2=figure; errorbar(X(:,3),Y(:,1),Y(:,1)-Y(:,2),Y(:,3)-Y(:,1));
    TheTableNice = table(validationImages.Labels,predictedLabels,'VariableNames',{'Actual','Predicted'});
    Confusion_Matrix1 = figure;
    PrettyCM = heatmap(TheTableNice,'Predicted','Actual');
    
    %     idx = randperm(20,4);
    %     Second_Figure = figure
    %     for i = 1:numel(idx)
    %         subplot(2,2,i)
    %         I = readimage(validationImages,idx(i));
    %         label = predictedLabels(idx(i));
    %         imshow(I)
    %         title(char(label))
    %     end
    
    try
        msg=sprintf('%s loop %d from %d MiniBatchSize %d with the results %d', Name, qq, nLoops, miniBatchSize, cp);
        sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
    catch
    end
    cd(resultsFolder);
    if  cp.Se > 0.79
        
        saveas(PrettyCM, ['BalCl_MB10_PrettyCM_' num2str(qq) '.jpg']);
        
        saveas(ROC_1, ['BalCl_MB10_ROC_1_' num2str(qq) '.jpg']);
        saveas(ROC_2, ['BalCl_MB10_ROC_2_' num2str(qq) '.jpg']);
        
        TrainingNeuNetw = findall(groot, 'Type', 'Figure');
        
        save(['RN_BalCl_MinBat10_' num2str(qq) '.mat'],'accuracy', 'AUC',...
            'c', 'cm', 'Confusion_Matrix', 'cp', ...
            'ind', 'miniBatchSize', 'netTransfer',...
            'numClasses', 'OPTROCPT', 'outputs', 'per', 'predictedLabels',...
            'scores','SUBY', 'SUBYNAMES', 'T', 'targets',...
            'TrainingNeuNetw','TheTableNice', 'validationImages', 'X', 'Y');
        %saveas(Architecture_1, ['UnCl_MB10_Architecture_1_' num2str(qq) '.jpg']);
        %saveas(Architecture_2, ['UnCl_MB10_Architecture_2_' num2str(qq) '.jpg']);
        
        %saveas(Second_Figure, ['UnCl_MB10_2nd_Figure_' num2str(qq) '.jpg']);
        
        
        
        try
            msg=sprintf('%s loop %d from %d MiniBatchSize %d , Sensitivity = %d , Balanced Accuracy = %d',...
                Name, qq, nLoops, miniBatchSize, cp.Se, cp.BA);
            sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
        catch
        end
        
%         close all
        % Save the figure of Training Progress in PNG format
        % saveas(TrainingNeuNetw,['RN_UnCl_MinBat10_' num2str(qq) '.png']);
        
        
%     else
%         save(['RN_BalCl_MinBat10_' num2str(qq) '.mat'],'accuracy', 'AUC',...
%             'cm', 'Confusion_Matrix', 'cp',...
%             'outputs', 'per', 'scores','SUBY', 'SUBYNAMES', 'T', 'targets',...
%             'TheTableNice', 'validationImages', 'X', 'Y');
%         saveas(PrettyCM, ['BalCl_MB10_PrettyCM_' num2str(qq) '.jpg']);
%         
%         try
%             msg=sprintf('%s loop %d from %d MiniBatchSize %d , Another one bits the dust, Sensitivity = %d',...
%                 Name, qq, nLoops, miniBatchSize, cp.Se);
%             sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
%         catch
%         end
        % Close all the figures
        
    end
    
    close all
    delete(findall(0));
    cd(codeFolder);
    % clearvars -except codeFolder resultsFolder TrainingFolder ValidationFolder nLoops qq
    
end

% Close progress bar
ioi_text_waitbar('Clear');

%%
nLoops = 1370;

for qq = 1361:nLoops
    
      
    ioi_text_waitbar(qq/nLoops, sprintf('ResNet101 Balanced MiniBatch=10, Running loop %d from %d', qq, 1440))
    
    close all
    trainingImages = imageDatastore(TrainingFolder, ...
        'IncludeSubfolders',true,...
        'FileExtensions','.tif','LabelSource','foldernames');
    
    trainingImages.ReadFcn = @(loc)imresize(imread(loc),[224, 224]);
    
    validationImages = imageDatastore(ValidationFolder, ...
        'IncludeSubfolders',true,...
        'FileExtensions','.tif','LabelSource','foldernames');
    
    validationImages.ReadFcn = @(loc)imresize(imread(loc),[224, 224]);
    
    
    % Load the pretrained GoogleNet network
    
    net = resnet101;
    net.Layers;
    
    % Extract the layer graph from the trained network and plot the layer graph.
    
    lgraph = layerGraph(net);
    Architecture_1 = figure('Units','normalized','Position',[0.1 0.1 0.8 0.8]);
    plot(lgraph);
    
       
    lgraph = removeLayers(lgraph, {'fc1000', 'prob', 'ClassificationLayer_predictions'});
    
    numClasses = numel(categories(trainingImages.Labels));
    newLayers = [
        fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',40,'BiasLearnRateFactor', 40)
        softmaxLayer('Name','softmax')
        classificationLayer('Name','classoutput')];
    lgraph = addLayers(lgraph,newLayers);
    
    lgraph = connectLayers(lgraph,'pool5','fc');
    
    Architecture_2 = figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
    plot(lgraph);
    ylim([0,10]);
    
    miniBatchSize = 10;
    
    %numIterationsPerEpoch = floor(numel(trainingImages.Labels)/miniBatchSize);
    options = trainingOptions('sgdm',...
        'ExecutionEnvironment','gpu',...
        'MiniBatchSize',miniBatchSize,...
        'MaxEpochs',100,...
        'InitialLearnRate',1e-5,...
        'VerboseFrequency',1,...
        'Plots','training-progress',...
        'ValidationData',validationImages,...
        'ValidationFrequency',5);
    netTransfer = trainNetwork(trainingImages,lgraph,options);
    
    %% Classify the validation images using the fine-tuned network.
    
    [predictedLabels, scores] = classify(netTransfer,validationImages,'ExecutionEnvironment','cpu');
    accuracy = mean(predictedLabels == validationImages.Labels);
    targets = grp2idx(validationImages.Labels);
    targets(targets==2) = 0;
    targets = ~targets;
    outputs = grp2idx(predictedLabels);
    outputs(outputs==2) = 0;
    outputs = ~outputs;
    [c,cm,ind,per] = confusion(targets', outputs');
    
    Confusion_Matrix=figure; plotconfusion(targets', outputs');
    
    %%
    
    addpath(genpath(resultsFolder))
    cp = my_classperf(cm);
    ROC_1=figure; plotroc(targets', scores(:,2)');
    
    [X,Y,T,AUC,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(targets', scores(:,2)',1,...
        'NBoot',1000,'TVals',0:0.05:1);
    ROC_2=figure; errorbar(X(:,3),Y(:,1),Y(:,1)-Y(:,2),Y(:,3)-Y(:,1));
    TheTableNice = table(validationImages.Labels,predictedLabels,'VariableNames',{'Actual','Predicted'});
    Confusion_Matrix1 = figure;
    PrettyCM = heatmap(TheTableNice,'Predicted','Actual');
    
    %     idx = randperm(20,4);
    %     Second_Figure = figure
    %     for i = 1:numel(idx)
    %         subplot(2,2,i)
    %         I = readimage(validationImages,idx(i));
    %         label = predictedLabels(idx(i));
    %         imshow(I)
    %         title(char(label))
    %     end
    
    try
        msg=sprintf('%s loop %d from %d MiniBatchSize %d with the results %d', Name, qq, nLoops, miniBatchSize, cp);
        sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
    catch
    end
    cd(resultsFolder);
    if  cp.Se > 0.79
        
        saveas(PrettyCM, ['BalCl_MB10_PrettyCM_' num2str(qq) '.jpg']);
        
        saveas(ROC_1, ['BalCl_MB10_ROC_1_' num2str(qq) '.jpg']);
        saveas(ROC_2, ['BalCl_MB10_ROC_2_' num2str(qq) '.jpg']);
        
        TrainingNeuNetw = findall(groot, 'Type', 'Figure');
        
        save(['RN_BalCl_MinBat10_' num2str(qq) '.mat'],'accuracy', 'AUC',...
            'c', 'cm', 'Confusion_Matrix', 'cp', ...
            'ind', 'miniBatchSize', 'netTransfer',...
            'numClasses', 'OPTROCPT', 'outputs', 'per', 'predictedLabels',...
            'scores','SUBY', 'SUBYNAMES', 'T', 'targets',...
            'TrainingNeuNetw','TheTableNice', 'validationImages', 'X', 'Y');
        %saveas(Architecture_1, ['UnCl_MB10_Architecture_1_' num2str(qq) '.jpg']);
        %saveas(Architecture_2, ['UnCl_MB10_Architecture_2_' num2str(qq) '.jpg']);
        
        %saveas(Second_Figure, ['UnCl_MB10_2nd_Figure_' num2str(qq) '.jpg']);
        
        
        
        try
            msg=sprintf('%s loop %d from %d MiniBatchSize %d , Sensitivity = %d , Balanced Accuracy = %d',...
                Name, qq, nLoops, miniBatchSize, cp.Se, cp.BA);
            sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
        catch
        end
        
%         close all
        % Save the figure of Training Progress in PNG format
        % saveas(TrainingNeuNetw,['RN_UnCl_MinBat10_' num2str(qq) '.png']);
        
        
%     else
%         save(['RN_BalCl_MinBat10_' num2str(qq) '.mat'],'accuracy', 'AUC',...
%             'cm', 'Confusion_Matrix', 'cp',...
%             'outputs', 'per', 'scores','SUBY', 'SUBYNAMES', 'T', 'targets',...
%             'TheTableNice', 'validationImages', 'X', 'Y');
%         saveas(PrettyCM, ['BalCl_MB10_PrettyCM_' num2str(qq) '.jpg']);
%         
%         try
%             msg=sprintf('%s loop %d from %d MiniBatchSize %d , Another one bits the dust, Sensitivity = %d',...
%                 Name, qq, nLoops, miniBatchSize, cp.Se);
%             sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
%         catch
%         end
        % Close all the figures
        
    end
    
    close all
    delete(findall(0));
    cd(codeFolder);
    % clearvars -except codeFolder resultsFolder TrainingFolder ValidationFolder nLoops qq
    
end

% Close progress bar
ioi_text_waitbar('Clear');


nLoops = 1380;

for qq = 1371:nLoops
    
      
    ioi_text_waitbar(qq/nLoops, sprintf('ResNet101 Balanced MiniBatch=10, Running loop %d from %d', qq, 1440))
    
    close all
    trainingImages = imageDatastore(TrainingFolder, ...
        'IncludeSubfolders',true,...
        'FileExtensions','.tif','LabelSource','foldernames');
    
    trainingImages.ReadFcn = @(loc)imresize(imread(loc),[224, 224]);
    
    validationImages = imageDatastore(ValidationFolder, ...
        'IncludeSubfolders',true,...
        'FileExtensions','.tif','LabelSource','foldernames');
    
    validationImages.ReadFcn = @(loc)imresize(imread(loc),[224, 224]);
    
    
    % Load the pretrained GoogleNet network
    
    net = resnet101;
    net.Layers;
    
    % Extract the layer graph from the trained network and plot the layer graph.
    
    lgraph = layerGraph(net);
    Architecture_1 = figure('Units','normalized','Position',[0.1 0.1 0.8 0.8]);
    plot(lgraph);
    
       
    lgraph = removeLayers(lgraph, {'fc1000', 'prob', 'ClassificationLayer_predictions'});
    
    numClasses = numel(categories(trainingImages.Labels));
    newLayers = [
        fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',40,'BiasLearnRateFactor', 60)
        softmaxLayer('Name','softmax')
        classificationLayer('Name','classoutput')];
    lgraph = addLayers(lgraph,newLayers);
    
    lgraph = connectLayers(lgraph,'pool5','fc');
    
    Architecture_2 = figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
    plot(lgraph);
    ylim([0,10]);
    
    miniBatchSize = 10;
    
    %numIterationsPerEpoch = floor(numel(trainingImages.Labels)/miniBatchSize);
    options = trainingOptions('sgdm',...
        'ExecutionEnvironment','gpu',...
        'MiniBatchSize',miniBatchSize,...
        'MaxEpochs',100,...
        'InitialLearnRate',1e-5,...
        'VerboseFrequency',1,...
        'Plots','training-progress',...
        'ValidationData',validationImages,...
        'ValidationFrequency',5);
    netTransfer = trainNetwork(trainingImages,lgraph,options);
    
    %% Classify the validation images using the fine-tuned network.
    
    [predictedLabels, scores] = classify(netTransfer,validationImages,'ExecutionEnvironment','cpu');
    accuracy = mean(predictedLabels == validationImages.Labels);
    targets = grp2idx(validationImages.Labels);
    targets(targets==2) = 0;
    targets = ~targets;
    outputs = grp2idx(predictedLabels);
    outputs(outputs==2) = 0;
    outputs = ~outputs;
    [c,cm,ind,per] = confusion(targets', outputs');
    
    Confusion_Matrix=figure; plotconfusion(targets', outputs');
    
    %%
    
    addpath(genpath(resultsFolder))
    cp = my_classperf(cm);
    ROC_1=figure; plotroc(targets', scores(:,2)');
    
    [X,Y,T,AUC,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(targets', scores(:,2)',1,...
        'NBoot',1000,'TVals',0:0.05:1);
    ROC_2=figure; errorbar(X(:,3),Y(:,1),Y(:,1)-Y(:,2),Y(:,3)-Y(:,1));
    TheTableNice = table(validationImages.Labels,predictedLabels,'VariableNames',{'Actual','Predicted'});
    Confusion_Matrix1 = figure;
    PrettyCM = heatmap(TheTableNice,'Predicted','Actual');
    
    %     idx = randperm(20,4);
    %     Second_Figure = figure
    %     for i = 1:numel(idx)
    %         subplot(2,2,i)
    %         I = readimage(validationImages,idx(i));
    %         label = predictedLabels(idx(i));
    %         imshow(I)
    %         title(char(label))
    %     end
    
    try
        msg=sprintf('%s loop %d from %d MiniBatchSize %d with the results %d', Name, qq, nLoops, miniBatchSize, cp);
        sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
    catch
    end
    cd(resultsFolder);
    if  cp.Se > 0.79
        
        saveas(PrettyCM, ['BalCl_MB10_PrettyCM_' num2str(qq) '.jpg']);
        
        saveas(ROC_1, ['BalCl_MB10_ROC_1_' num2str(qq) '.jpg']);
        saveas(ROC_2, ['BalCl_MB10_ROC_2_' num2str(qq) '.jpg']);
        
        TrainingNeuNetw = findall(groot, 'Type', 'Figure');
        
        save(['RN_BalCl_MinBat10_' num2str(qq) '.mat'],'accuracy', 'AUC',...
            'c', 'cm', 'Confusion_Matrix', 'cp', ...
            'ind', 'miniBatchSize', 'netTransfer',...
            'numClasses', 'OPTROCPT', 'outputs', 'per', 'predictedLabels',...
            'scores','SUBY', 'SUBYNAMES', 'T', 'targets',...
            'TrainingNeuNetw','TheTableNice', 'validationImages', 'X', 'Y');
        %saveas(Architecture_1, ['UnCl_MB10_Architecture_1_' num2str(qq) '.jpg']);
        %saveas(Architecture_2, ['UnCl_MB10_Architecture_2_' num2str(qq) '.jpg']);
        
        %saveas(Second_Figure, ['UnCl_MB10_2nd_Figure_' num2str(qq) '.jpg']);
        
        
        
        try
            msg=sprintf('%s loop %d from %d MiniBatchSize %d , Sensitivity = %d , Balanced Accuracy = %d',...
                Name, qq, nLoops, miniBatchSize, cp.Se, cp.BA);
            sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
        catch
        end
        
%         close all
        % Save the figure of Training Progress in PNG format
        % saveas(TrainingNeuNetw,['RN_UnCl_MinBat10_' num2str(qq) '.png']);
        
        
%     else
%         save(['RN_BalCl_MinBat10_' num2str(qq) '.mat'],'accuracy', 'AUC',...
%             'cm', 'Confusion_Matrix', 'cp',...
%             'outputs', 'per', 'scores','SUBY', 'SUBYNAMES', 'T', 'targets',...
%             'TheTableNice', 'validationImages', 'X', 'Y');
%         saveas(PrettyCM, ['BalCl_MB10_PrettyCM_' num2str(qq) '.jpg']);
%         
%         try
%             msg=sprintf('%s loop %d from %d MiniBatchSize %d , Another one bits the dust, Sensitivity = %d',...
%                 Name, qq, nLoops, miniBatchSize, cp.Se);
%             sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
%         catch
%         end
        % Close all the figures
        
    end
    
    close all
    delete(findall(0));
    cd(codeFolder);
    % clearvars -except codeFolder resultsFolder TrainingFolder ValidationFolder nLoops qq
    
end

% Close progress bar
ioi_text_waitbar('Clear');

% *ResNet 101 for Database of Brasil, IJC & ISSSTE*

% Unbalanced Class
% MiniBatch = 10

% Analysis of three databases is realized for compare the results, only the
% results that has a Sensitivity bigger than 80% is saved

clear; close all; clc;

% Folders Location
codeFolder = 'C:\Users\LANCYTT\Documents\Juan Carlos Torres\Thermograms Analysis';
% addpath('I:\Archivos Juan Carlos Torres\MATLAB\MATLAB Codigos');

resultsFolder = 'C:\Users\LANCYTT\Documents\Juan Carlos Torres\Results\ResNet101\Iteraciones\Balanced';
addpath('C:\Users\LANCYTT\Documents\Juan Carlos Torres\Thermograms Analysis');

TrainingFolder = 'C:\Users\LANCYTT\Documents\Juan Carlos Torres\Data Augmented Thermography\Data Augmented Bra+IJC+ISSSTE\Balanced Augmented Modify';
ValidationFolder = 'C:\Users\LANCYTT\Documents\Juan Carlos Torres\Data Augmented Thermography\Train - Valid Bra+IJC+ISSSTE\Validation Balanced Class';

pat_sendmail

ioi_text_waitbar(0, 'Running ResNet 101, Balanced MiniBatch 10... Please wait...');

Name=sprintf('ResNet101 Balanced');
nLoops = 1390;

for qq = 1381:nLoops
    
      
    ioi_text_waitbar(qq/nLoops, sprintf('ResNet101 Balanced MiniBatch=10, Running loop %d from %d', qq, 1440))
    
    close all
    trainingImages = imageDatastore(TrainingFolder, ...
        'IncludeSubfolders',true,...
        'FileExtensions','.tif','LabelSource','foldernames');
    
    trainingImages.ReadFcn = @(loc)imresize(imread(loc),[224, 224]);
    
    validationImages = imageDatastore(ValidationFolder, ...
        'IncludeSubfolders',true,...
        'FileExtensions','.tif','LabelSource','foldernames');
    
    validationImages.ReadFcn = @(loc)imresize(imread(loc),[224, 224]);
    
    
    % Load the pretrained GoogleNet network
    
    net = resnet101;
    net.Layers;
    
    % Extract the layer graph from the trained network and plot the layer graph.
    
    lgraph = layerGraph(net);
    Architecture_1 = figure('Units','normalized','Position',[0.1 0.1 0.8 0.8]);
    plot(lgraph);
    
       
    lgraph = removeLayers(lgraph, {'fc1000', 'prob', 'ClassificationLayer_predictions'});
    
    numClasses = numel(categories(trainingImages.Labels));
    newLayers = [
        fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',60,'BiasLearnRateFactor', 1)
        softmaxLayer('Name','softmax')
        classificationLayer('Name','classoutput')];
    lgraph = addLayers(lgraph,newLayers);
    
    lgraph = connectLayers(lgraph,'pool5','fc');
    
    Architecture_2 = figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
    plot(lgraph);
    ylim([0,10]);
    
    miniBatchSize = 10;
    
    %numIterationsPerEpoch = floor(numel(trainingImages.Labels)/miniBatchSize);
    options = trainingOptions('sgdm',...
        'ExecutionEnvironment','gpu',...
        'MiniBatchSize',miniBatchSize,...
        'MaxEpochs',100,...
        'InitialLearnRate',1e-5,...
        'VerboseFrequency',1,...
        'Plots','training-progress',...
        'ValidationData',validationImages,...
        'ValidationFrequency',5);
    netTransfer = trainNetwork(trainingImages,lgraph,options);
    
    %% Classify the validation images using the fine-tuned network.
    
    [predictedLabels, scores] = classify(netTransfer,validationImages,'ExecutionEnvironment','cpu');
    accuracy = mean(predictedLabels == validationImages.Labels);
    targets = grp2idx(validationImages.Labels);
    targets(targets==2) = 0;
    targets = ~targets;
    outputs = grp2idx(predictedLabels);
    outputs(outputs==2) = 0;
    outputs = ~outputs;
    [c,cm,ind,per] = confusion(targets', outputs');
    
    Confusion_Matrix=figure; plotconfusion(targets', outputs');
    
    %%
    
    addpath(genpath(resultsFolder))
    cp = my_classperf(cm);
    ROC_1=figure; plotroc(targets', scores(:,2)');
    
    [X,Y,T,AUC,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(targets', scores(:,2)',1,...
        'NBoot',1000,'TVals',0:0.05:1);
    ROC_2=figure; errorbar(X(:,3),Y(:,1),Y(:,1)-Y(:,2),Y(:,3)-Y(:,1));
    TheTableNice = table(validationImages.Labels,predictedLabels,'VariableNames',{'Actual','Predicted'});
    Confusion_Matrix1 = figure;
    PrettyCM = heatmap(TheTableNice,'Predicted','Actual');
    
    %     idx = randperm(20,4);
    %     Second_Figure = figure
    %     for i = 1:numel(idx)
    %         subplot(2,2,i)
    %         I = readimage(validationImages,idx(i));
    %         label = predictedLabels(idx(i));
    %         imshow(I)
    %         title(char(label))
    %     end
    
    try
        msg=sprintf('%s loop %d from %d MiniBatchSize %d with the results %d', Name, qq, nLoops, miniBatchSize, cp);
        sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
    catch
    end
    cd(resultsFolder);
    if  cp.Se > 0.79
        
        saveas(PrettyCM, ['BalCl_MB10_PrettyCM_' num2str(qq) '.jpg']);
        
        saveas(ROC_1, ['BalCl_MB10_ROC_1_' num2str(qq) '.jpg']);
        saveas(ROC_2, ['BalCl_MB10_ROC_2_' num2str(qq) '.jpg']);
        
        TrainingNeuNetw = findall(groot, 'Type', 'Figure');
        
        save(['RN_BalCl_MinBat10_' num2str(qq) '.mat'],'accuracy', 'AUC',...
            'c', 'cm', 'Confusion_Matrix', 'cp', ...
            'ind', 'miniBatchSize', 'netTransfer',...
            'numClasses', 'OPTROCPT', 'outputs', 'per', 'predictedLabels',...
            'scores','SUBY', 'SUBYNAMES', 'T', 'targets',...
            'TrainingNeuNetw','TheTableNice', 'validationImages', 'X', 'Y');
        %saveas(Architecture_1, ['UnCl_MB10_Architecture_1_' num2str(qq) '.jpg']);
        %saveas(Architecture_2, ['UnCl_MB10_Architecture_2_' num2str(qq) '.jpg']);
        
        %saveas(Second_Figure, ['UnCl_MB10_2nd_Figure_' num2str(qq) '.jpg']);
        
        
        
        try
            msg=sprintf('%s loop %d from %d MiniBatchSize %d , Sensitivity = %d , Balanced Accuracy = %d',...
                Name, qq, nLoops, miniBatchSize, cp.Se, cp.BA);
            sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
        catch
        end
        
%         close all
        % Save the figure of Training Progress in PNG format
        % saveas(TrainingNeuNetw,['RN_UnCl_MinBat10_' num2str(qq) '.png']);
        
        
%     else
%         save(['RN_BalCl_MinBat10_' num2str(qq) '.mat'],'accuracy', 'AUC',...
%             'cm', 'Confusion_Matrix', 'cp',...
%             'outputs', 'per', 'scores','SUBY', 'SUBYNAMES', 'T', 'targets',...
%             'TheTableNice', 'validationImages', 'X', 'Y');
%         saveas(PrettyCM, ['BalCl_MB10_PrettyCM_' num2str(qq) '.jpg']);
%         
%         try
%             msg=sprintf('%s loop %d from %d MiniBatchSize %d , Another one bits the dust, Sensitivity = %d',...
%                 Name, qq, nLoops, miniBatchSize, cp.Se);
%             sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
%         catch
%         end
        % Close all the figures
        
    end
    
    close all
    delete(findall(0));
    cd(codeFolder);
    % clearvars -except codeFolder resultsFolder TrainingFolder ValidationFolder nLoops qq
    
end

% Close progress bar
ioi_text_waitbar('Clear');

nLoops = 1400;

for qq = 1391:nLoops
    
      
    ioi_text_waitbar(qq/nLoops, sprintf('ResNet101 Balanced MiniBatch=10, Running loop %d from %d', qq, 1440))
    
    close all
    trainingImages = imageDatastore(TrainingFolder, ...
        'IncludeSubfolders',true,...
        'FileExtensions','.tif','LabelSource','foldernames');
    
    trainingImages.ReadFcn = @(loc)imresize(imread(loc),[224, 224]);
    
    validationImages = imageDatastore(ValidationFolder, ...
        'IncludeSubfolders',true,...
        'FileExtensions','.tif','LabelSource','foldernames');
    
    validationImages.ReadFcn = @(loc)imresize(imread(loc),[224, 224]);
    
    
    % Load the pretrained GoogleNet network
    
    net = resnet101;
    net.Layers;
    
    % Extract the layer graph from the trained network and plot the layer graph.
    
    lgraph = layerGraph(net);
    Architecture_1 = figure('Units','normalized','Position',[0.1 0.1 0.8 0.8]);
    plot(lgraph);
    
       
    lgraph = removeLayers(lgraph, {'fc1000', 'prob', 'ClassificationLayer_predictions'});
    
    numClasses = numel(categories(trainingImages.Labels));
    newLayers = [
        fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',60,'BiasLearnRateFactor', 5)
        softmaxLayer('Name','softmax')
        classificationLayer('Name','classoutput')];
    lgraph = addLayers(lgraph,newLayers);
    
    lgraph = connectLayers(lgraph,'pool5','fc');
    
    Architecture_2 = figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
    plot(lgraph);
    ylim([0,10]);
    
    miniBatchSize = 10;
    
    %numIterationsPerEpoch = floor(numel(trainingImages.Labels)/miniBatchSize);
    options = trainingOptions('sgdm',...
        'ExecutionEnvironment','gpu',...
        'MiniBatchSize',miniBatchSize,...
        'MaxEpochs',100,...
        'InitialLearnRate',1e-5,...
        'VerboseFrequency',1,...
        'Plots','training-progress',...
        'ValidationData',validationImages,...
        'ValidationFrequency',5);
    netTransfer = trainNetwork(trainingImages,lgraph,options);
    
    %% Classify the validation images using the fine-tuned network.
    
    [predictedLabels, scores] = classify(netTransfer,validationImages,'ExecutionEnvironment','cpu');
    accuracy = mean(predictedLabels == validationImages.Labels);
    targets = grp2idx(validationImages.Labels);
    targets(targets==2) = 0;
    targets = ~targets;
    outputs = grp2idx(predictedLabels);
    outputs(outputs==2) = 0;
    outputs = ~outputs;
    [c,cm,ind,per] = confusion(targets', outputs');
    
    Confusion_Matrix=figure; plotconfusion(targets', outputs');
    
    %%
    
    addpath(genpath(resultsFolder))
    cp = my_classperf(cm);
    ROC_1=figure; plotroc(targets', scores(:,2)');
    
    [X,Y,T,AUC,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(targets', scores(:,2)',1,...
        'NBoot',1000,'TVals',0:0.05:1);
    ROC_2=figure; errorbar(X(:,3),Y(:,1),Y(:,1)-Y(:,2),Y(:,3)-Y(:,1));
    TheTableNice = table(validationImages.Labels,predictedLabels,'VariableNames',{'Actual','Predicted'});
    Confusion_Matrix1 = figure;
    PrettyCM = heatmap(TheTableNice,'Predicted','Actual');
    
    %     idx = randperm(20,4);
    %     Second_Figure = figure
    %     for i = 1:numel(idx)
    %         subplot(2,2,i)
    %         I = readimage(validationImages,idx(i));
    %         label = predictedLabels(idx(i));
    %         imshow(I)
    %         title(char(label))
    %     end
    
    try
        msg=sprintf('%s loop %d from %d MiniBatchSize %d with the results %d', Name, qq, nLoops, miniBatchSize, cp);
        sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
    catch
    end
    cd(resultsFolder);
    if  cp.Se > 0.79
        
        saveas(PrettyCM, ['BalCl_MB10_PrettyCM_' num2str(qq) '.jpg']);
        
        saveas(ROC_1, ['BalCl_MB10_ROC_1_' num2str(qq) '.jpg']);
        saveas(ROC_2, ['BalCl_MB10_ROC_2_' num2str(qq) '.jpg']);
        
        TrainingNeuNetw = findall(groot, 'Type', 'Figure');
        
        save(['RN_BalCl_MinBat10_' num2str(qq) '.mat'],'accuracy', 'AUC',...
            'c', 'cm', 'Confusion_Matrix', 'cp', ...
            'ind', 'miniBatchSize', 'netTransfer',...
            'numClasses', 'OPTROCPT', 'outputs', 'per', 'predictedLabels',...
            'scores','SUBY', 'SUBYNAMES', 'T', 'targets',...
            'TrainingNeuNetw','TheTableNice', 'validationImages', 'X', 'Y');
        %saveas(Architecture_1, ['UnCl_MB10_Architecture_1_' num2str(qq) '.jpg']);
        %saveas(Architecture_2, ['UnCl_MB10_Architecture_2_' num2str(qq) '.jpg']);
        
        %saveas(Second_Figure, ['UnCl_MB10_2nd_Figure_' num2str(qq) '.jpg']);
        
        
        
        try
            msg=sprintf('%s loop %d from %d MiniBatchSize %d , Sensitivity = %d , Balanced Accuracy = %d',...
                Name, qq, nLoops, miniBatchSize, cp.Se, cp.BA);
            sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
        catch
        end
        
%         close all
        % Save the figure of Training Progress in PNG format
        % saveas(TrainingNeuNetw,['RN_UnCl_MinBat10_' num2str(qq) '.png']);
        
        
%     else
%         save(['RN_BalCl_MinBat10_' num2str(qq) '.mat'],'accuracy', 'AUC',...
%             'cm', 'Confusion_Matrix', 'cp',...
%             'outputs', 'per', 'scores','SUBY', 'SUBYNAMES', 'T', 'targets',...
%             'TheTableNice', 'validationImages', 'X', 'Y');
%         saveas(PrettyCM, ['BalCl_MB10_PrettyCM_' num2str(qq) '.jpg']);
%         
%         try
%             msg=sprintf('%s loop %d from %d MiniBatchSize %d , Another one bits the dust, Sensitivity = %d',...
%                 Name, qq, nLoops, miniBatchSize, cp.Se);
%             sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
%         catch
%         end
        % Close all the figures
        
    end
    
    close all
    delete(findall(0));
    cd(codeFolder);
    % clearvars -except codeFolder resultsFolder TrainingFolder ValidationFolder nLoops qq
    
end

% Close progress bar
ioi_text_waitbar('Clear');

%%
nLoops = 1410;

for qq = 1401:nLoops
    
      
    ioi_text_waitbar(qq/nLoops, sprintf('ResNet101 Balanced MiniBatch=10, Running loop %d from %d', qq, 1440))
    
    close all
    trainingImages = imageDatastore(TrainingFolder, ...
        'IncludeSubfolders',true,...
        'FileExtensions','.tif','LabelSource','foldernames');
    
    trainingImages.ReadFcn = @(loc)imresize(imread(loc),[224, 224]);
    
    validationImages = imageDatastore(ValidationFolder, ...
        'IncludeSubfolders',true,...
        'FileExtensions','.tif','LabelSource','foldernames');
    
    validationImages.ReadFcn = @(loc)imresize(imread(loc),[224, 224]);
    
    
    % Load the pretrained GoogleNet network
    
    net = resnet101;
    net.Layers;
    
    % Extract the layer graph from the trained network and plot the layer graph.
    
    lgraph = layerGraph(net);
    Architecture_1 = figure('Units','normalized','Position',[0.1 0.1 0.8 0.8]);
    plot(lgraph);
    
       
    lgraph = removeLayers(lgraph, {'fc1000', 'prob', 'ClassificationLayer_predictions'});
    
    numClasses = numel(categories(trainingImages.Labels));
    newLayers = [
        fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',60,'BiasLearnRateFactor', 10)
        softmaxLayer('Name','softmax')
        classificationLayer('Name','classoutput')];
    lgraph = addLayers(lgraph,newLayers);
    
    lgraph = connectLayers(lgraph,'pool5','fc');
    
    Architecture_2 = figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
    plot(lgraph);
    ylim([0,10]);
    
    miniBatchSize = 10;
    
    %numIterationsPerEpoch = floor(numel(trainingImages.Labels)/miniBatchSize);
    options = trainingOptions('sgdm',...
        'ExecutionEnvironment','gpu',...
        'MiniBatchSize',miniBatchSize,...
        'MaxEpochs',100,...
        'InitialLearnRate',1e-5,...
        'VerboseFrequency',1,...
        'Plots','training-progress',...
        'ValidationData',validationImages,...
        'ValidationFrequency',5);
    netTransfer = trainNetwork(trainingImages,lgraph,options);
    
    %% Classify the validation images using the fine-tuned network.
    
    [predictedLabels, scores] = classify(netTransfer,validationImages,'ExecutionEnvironment','cpu');
    accuracy = mean(predictedLabels == validationImages.Labels);
    targets = grp2idx(validationImages.Labels);
    targets(targets==2) = 0;
    targets = ~targets;
    outputs = grp2idx(predictedLabels);
    outputs(outputs==2) = 0;
    outputs = ~outputs;
    [c,cm,ind,per] = confusion(targets', outputs');
    
    Confusion_Matrix=figure; plotconfusion(targets', outputs');
    
    %%
    
    addpath(genpath(resultsFolder))
    cp = my_classperf(cm);
    ROC_1=figure; plotroc(targets', scores(:,2)');
    
    [X,Y,T,AUC,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(targets', scores(:,2)',1,...
        'NBoot',1000,'TVals',0:0.05:1);
    ROC_2=figure; errorbar(X(:,3),Y(:,1),Y(:,1)-Y(:,2),Y(:,3)-Y(:,1));
    TheTableNice = table(validationImages.Labels,predictedLabels,'VariableNames',{'Actual','Predicted'});
    Confusion_Matrix1 = figure;
    PrettyCM = heatmap(TheTableNice,'Predicted','Actual');
    
    %     idx = randperm(20,4);
    %     Second_Figure = figure
    %     for i = 1:numel(idx)
    %         subplot(2,2,i)
    %         I = readimage(validationImages,idx(i));
    %         label = predictedLabels(idx(i));
    %         imshow(I)
    %         title(char(label))
    %     end
    
    try
        msg=sprintf('%s loop %d from %d MiniBatchSize %d with the results %d', Name, qq, nLoops, miniBatchSize, cp);
        sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
    catch
    end
    cd(resultsFolder);
    if  cp.Se > 0.79
        
        saveas(PrettyCM, ['BalCl_MB10_PrettyCM_' num2str(qq) '.jpg']);
        
        saveas(ROC_1, ['BalCl_MB10_ROC_1_' num2str(qq) '.jpg']);
        saveas(ROC_2, ['BalCl_MB10_ROC_2_' num2str(qq) '.jpg']);
        
        TrainingNeuNetw = findall(groot, 'Type', 'Figure');
        
        save(['RN_BalCl_MinBat10_' num2str(qq) '.mat'],'accuracy', 'AUC',...
            'c', 'cm', 'Confusion_Matrix', 'cp', ...
            'ind', 'miniBatchSize', 'netTransfer',...
            'numClasses', 'OPTROCPT', 'outputs', 'per', 'predictedLabels',...
            'scores','SUBY', 'SUBYNAMES', 'T', 'targets',...
            'TrainingNeuNetw','TheTableNice', 'validationImages', 'X', 'Y');
        %saveas(Architecture_1, ['UnCl_MB10_Architecture_1_' num2str(qq) '.jpg']);
        %saveas(Architecture_2, ['UnCl_MB10_Architecture_2_' num2str(qq) '.jpg']);
        
        %saveas(Second_Figure, ['UnCl_MB10_2nd_Figure_' num2str(qq) '.jpg']);
        
        
        
        try
            msg=sprintf('%s loop %d from %d MiniBatchSize %d , Sensitivity = %d , Balanced Accuracy = %d',...
                Name, qq, nLoops, miniBatchSize, cp.Se, cp.BA);
            sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
        catch
        end
        
%         close all
        % Save the figure of Training Progress in PNG format
        % saveas(TrainingNeuNetw,['RN_UnCl_MinBat10_' num2str(qq) '.png']);
        
        
%     else
%         save(['RN_BalCl_MinBat10_' num2str(qq) '.mat'],'accuracy', 'AUC',...
%             'cm', 'Confusion_Matrix', 'cp',...
%             'outputs', 'per', 'scores','SUBY', 'SUBYNAMES', 'T', 'targets',...
%             'TheTableNice', 'validationImages', 'X', 'Y');
%         saveas(PrettyCM, ['BalCl_MB10_PrettyCM_' num2str(qq) '.jpg']);
%         
%         try
%             msg=sprintf('%s loop %d from %d MiniBatchSize %d , Another one bits the dust, Sensitivity = %d',...
%                 Name, qq, nLoops, miniBatchSize, cp.Se);
%             sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
%         catch
%         end
        % Close all the figures
        
    end
    
    close all
    delete(findall(0));
    cd(codeFolder);
    % clearvars -except codeFolder resultsFolder TrainingFolder ValidationFolder nLoops qq
    
end

% Close progress bar
ioi_text_waitbar('Clear');

%%
nLoops = 1420;

for qq = 1411:nLoops
    
      
    ioi_text_waitbar(qq/nLoops, sprintf('ResNet101 Balanced MiniBatch=10, Running loop %d from %d', qq, 1440))
    
    close all
    trainingImages = imageDatastore(TrainingFolder, ...
        'IncludeSubfolders',true,...
        'FileExtensions','.tif','LabelSource','foldernames');
    
    trainingImages.ReadFcn = @(loc)imresize(imread(loc),[224, 224]);
    
    validationImages = imageDatastore(ValidationFolder, ...
        'IncludeSubfolders',true,...
        'FileExtensions','.tif','LabelSource','foldernames');
    
    validationImages.ReadFcn = @(loc)imresize(imread(loc),[224, 224]);
    
    
    % Load the pretrained GoogleNet network
    
    net = resnet101;
    net.Layers;
    
    % Extract the layer graph from the trained network and plot the layer graph.
    
    lgraph = layerGraph(net);
    Architecture_1 = figure('Units','normalized','Position',[0.1 0.1 0.8 0.8]);
    plot(lgraph);
    
       
    lgraph = removeLayers(lgraph, {'fc1000', 'prob', 'ClassificationLayer_predictions'});
    
    numClasses = numel(categories(trainingImages.Labels));
    newLayers = [
        fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',60,'BiasLearnRateFactor', 20)
        softmaxLayer('Name','softmax')
        classificationLayer('Name','classoutput')];
    lgraph = addLayers(lgraph,newLayers);
    
    lgraph = connectLayers(lgraph,'pool5','fc');
    
    Architecture_2 = figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
    plot(lgraph);
    ylim([0,10]);
    
    miniBatchSize = 10;
    
    %numIterationsPerEpoch = floor(numel(trainingImages.Labels)/miniBatchSize);
    options = trainingOptions('sgdm',...
        'ExecutionEnvironment','gpu',...
        'MiniBatchSize',miniBatchSize,...
        'MaxEpochs',100,...
        'InitialLearnRate',1e-5,...
        'VerboseFrequency',1,...
        'Plots','training-progress',...
        'ValidationData',validationImages,...
        'ValidationFrequency',5);
    netTransfer = trainNetwork(trainingImages,lgraph,options);
    
    %% Classify the validation images using the fine-tuned network.
    
    [predictedLabels, scores] = classify(netTransfer,validationImages,'ExecutionEnvironment','cpu');
    accuracy = mean(predictedLabels == validationImages.Labels);
    targets = grp2idx(validationImages.Labels);
    targets(targets==2) = 0;
    targets = ~targets;
    outputs = grp2idx(predictedLabels);
    outputs(outputs==2) = 0;
    outputs = ~outputs;
    [c,cm,ind,per] = confusion(targets', outputs');
    
    Confusion_Matrix=figure; plotconfusion(targets', outputs');
    
    %%
    
    addpath(genpath(resultsFolder))
    cp = my_classperf(cm);
    ROC_1=figure; plotroc(targets', scores(:,2)');
    
    [X,Y,T,AUC,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(targets', scores(:,2)',1,...
        'NBoot',1000,'TVals',0:0.05:1);
    ROC_2=figure; errorbar(X(:,3),Y(:,1),Y(:,1)-Y(:,2),Y(:,3)-Y(:,1));
    TheTableNice = table(validationImages.Labels,predictedLabels,'VariableNames',{'Actual','Predicted'});
    Confusion_Matrix1 = figure;
    PrettyCM = heatmap(TheTableNice,'Predicted','Actual');
    
    %     idx = randperm(20,4);
    %     Second_Figure = figure
    %     for i = 1:numel(idx)
    %         subplot(2,2,i)
    %         I = readimage(validationImages,idx(i));
    %         label = predictedLabels(idx(i));
    %         imshow(I)
    %         title(char(label))
    %     end
    
    try
        msg=sprintf('%s loop %d from %d MiniBatchSize %d with the results %d', Name, qq, nLoops, miniBatchSize, cp);
        sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
    catch
    end
    cd(resultsFolder);
    if  cp.Se > 0.79
        
        saveas(PrettyCM, ['BalCl_MB10_PrettyCM_' num2str(qq) '.jpg']);
        
        saveas(ROC_1, ['BalCl_MB10_ROC_1_' num2str(qq) '.jpg']);
        saveas(ROC_2, ['BalCl_MB10_ROC_2_' num2str(qq) '.jpg']);
        
        TrainingNeuNetw = findall(groot, 'Type', 'Figure');
        
        save(['RN_BalCl_MinBat10_' num2str(qq) '.mat'],'accuracy', 'AUC',...
            'c', 'cm', 'Confusion_Matrix', 'cp', ...
            'ind', 'miniBatchSize', 'netTransfer',...
            'numClasses', 'OPTROCPT', 'outputs', 'per', 'predictedLabels',...
            'scores','SUBY', 'SUBYNAMES', 'T', 'targets',...
            'TrainingNeuNetw','TheTableNice', 'validationImages', 'X', 'Y');
        %saveas(Architecture_1, ['UnCl_MB10_Architecture_1_' num2str(qq) '.jpg']);
        %saveas(Architecture_2, ['UnCl_MB10_Architecture_2_' num2str(qq) '.jpg']);
        
        %saveas(Second_Figure, ['UnCl_MB10_2nd_Figure_' num2str(qq) '.jpg']);
        
        
        
        try
            msg=sprintf('%s loop %d from %d MiniBatchSize %d , Sensitivity = %d , Balanced Accuracy = %d',...
                Name, qq, nLoops, miniBatchSize, cp.Se, cp.BA);
            sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
        catch
        end
        
%         close all
        % Save the figure of Training Progress in PNG format
        % saveas(TrainingNeuNetw,['RN_UnCl_MinBat10_' num2str(qq) '.png']);
        
        
%     else
%         save(['RN_BalCl_MinBat10_' num2str(qq) '.mat'],'accuracy', 'AUC',...
%             'cm', 'Confusion_Matrix', 'cp',...
%             'outputs', 'per', 'scores','SUBY', 'SUBYNAMES', 'T', 'targets',...
%             'TheTableNice', 'validationImages', 'X', 'Y');
%         saveas(PrettyCM, ['BalCl_MB10_PrettyCM_' num2str(qq) '.jpg']);
%         
%         try
%             msg=sprintf('%s loop %d from %d MiniBatchSize %d , Another one bits the dust, Sensitivity = %d',...
%                 Name, qq, nLoops, miniBatchSize, cp.Se);
%             sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
%         catch
%         end
        % Close all the figures
        
    end
    
    close all
    delete(findall(0));
    cd(codeFolder);
    % clearvars -except codeFolder resultsFolder TrainingFolder ValidationFolder nLoops qq
    
end

% Close progress bar
ioi_text_waitbar('Clear');

%%
nLoops = 1430;

for qq = 1421:nLoops
    
      
    ioi_text_waitbar(qq/nLoops, sprintf('ResNet101 Balanced MiniBatch=10, Running loop %d from %d', qq, 1440))
    
    close all
    trainingImages = imageDatastore(TrainingFolder, ...
        'IncludeSubfolders',true,...
        'FileExtensions','.tif','LabelSource','foldernames');
    
    trainingImages.ReadFcn = @(loc)imresize(imread(loc),[224, 224]);
    
    validationImages = imageDatastore(ValidationFolder, ...
        'IncludeSubfolders',true,...
        'FileExtensions','.tif','LabelSource','foldernames');
    
    validationImages.ReadFcn = @(loc)imresize(imread(loc),[224, 224]);
    
    
    % Load the pretrained GoogleNet network
    
    net = resnet101;
    net.Layers;
    
    % Extract the layer graph from the trained network and plot the layer graph.
    
    lgraph = layerGraph(net);
    Architecture_1 = figure('Units','normalized','Position',[0.1 0.1 0.8 0.8]);
    plot(lgraph);
    
       
    lgraph = removeLayers(lgraph, {'fc1000', 'prob', 'ClassificationLayer_predictions'});
    
    numClasses = numel(categories(trainingImages.Labels));
    newLayers = [
        fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',60,'BiasLearnRateFactor', 40)
        softmaxLayer('Name','softmax')
        classificationLayer('Name','classoutput')];
    lgraph = addLayers(lgraph,newLayers);
    
    lgraph = connectLayers(lgraph,'pool5','fc');
    
    Architecture_2 = figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
    plot(lgraph);
    ylim([0,10]);
    
    miniBatchSize = 10;
    
    %numIterationsPerEpoch = floor(numel(trainingImages.Labels)/miniBatchSize);
    options = trainingOptions('sgdm',...
        'ExecutionEnvironment','gpu',...
        'MiniBatchSize',miniBatchSize,...
        'MaxEpochs',100,...
        'InitialLearnRate',1e-5,...
        'VerboseFrequency',1,...
        'Plots','training-progress',...
        'ValidationData',validationImages,...
        'ValidationFrequency',5);
    netTransfer = trainNetwork(trainingImages,lgraph,options);
    
    %% Classify the validation images using the fine-tuned network.
    
    [predictedLabels, scores] = classify(netTransfer,validationImages,'ExecutionEnvironment','cpu');
    accuracy = mean(predictedLabels == validationImages.Labels);
    targets = grp2idx(validationImages.Labels);
    targets(targets==2) = 0;
    targets = ~targets;
    outputs = grp2idx(predictedLabels);
    outputs(outputs==2) = 0;
    outputs = ~outputs;
    [c,cm,ind,per] = confusion(targets', outputs');
    
    Confusion_Matrix=figure; plotconfusion(targets', outputs');
    
    %%
    
    addpath(genpath(resultsFolder))
    cp = my_classperf(cm);
    ROC_1=figure; plotroc(targets', scores(:,2)');
    
    [X,Y,T,AUC,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(targets', scores(:,2)',1,...
        'NBoot',1000,'TVals',0:0.05:1);
    ROC_2=figure; errorbar(X(:,3),Y(:,1),Y(:,1)-Y(:,2),Y(:,3)-Y(:,1));
    TheTableNice = table(validationImages.Labels,predictedLabels,'VariableNames',{'Actual','Predicted'});
    Confusion_Matrix1 = figure;
    PrettyCM = heatmap(TheTableNice,'Predicted','Actual');
    
    %     idx = randperm(20,4);
    %     Second_Figure = figure
    %     for i = 1:numel(idx)
    %         subplot(2,2,i)
    %         I = readimage(validationImages,idx(i));
    %         label = predictedLabels(idx(i));
    %         imshow(I)
    %         title(char(label))
    %     end
    
    try
        msg=sprintf('%s loop %d from %d MiniBatchSize %d with the results %d', Name, qq, nLoops, miniBatchSize, cp);
        sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
    catch
    end
    cd(resultsFolder);
    if  cp.Se > 0.79
        
        saveas(PrettyCM, ['BalCl_MB10_PrettyCM_' num2str(qq) '.jpg']);
        
        saveas(ROC_1, ['BalCl_MB10_ROC_1_' num2str(qq) '.jpg']);
        saveas(ROC_2, ['BalCl_MB10_ROC_2_' num2str(qq) '.jpg']);
        
        TrainingNeuNetw = findall(groot, 'Type', 'Figure');
        
        save(['RN_BalCl_MinBat10_' num2str(qq) '.mat'],'accuracy', 'AUC',...
            'c', 'cm', 'Confusion_Matrix', 'cp', ...
            'ind', 'miniBatchSize', 'netTransfer',...
            'numClasses', 'OPTROCPT', 'outputs', 'per', 'predictedLabels',...
            'scores','SUBY', 'SUBYNAMES', 'T', 'targets',...
            'TrainingNeuNetw','TheTableNice', 'validationImages', 'X', 'Y');
        %saveas(Architecture_1, ['UnCl_MB10_Architecture_1_' num2str(qq) '.jpg']);
        %saveas(Architecture_2, ['UnCl_MB10_Architecture_2_' num2str(qq) '.jpg']);
        
        %saveas(Second_Figure, ['UnCl_MB10_2nd_Figure_' num2str(qq) '.jpg']);
        
        
        
        try
            msg=sprintf('%s loop %d from %d MiniBatchSize %d , Sensitivity = %d , Balanced Accuracy = %d',...
                Name, qq, nLoops, miniBatchSize, cp.Se, cp.BA);
            sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
        catch
        end
        
%         close all
        % Save the figure of Training Progress in PNG format
        % saveas(TrainingNeuNetw,['RN_UnCl_MinBat10_' num2str(qq) '.png']);
        
        
%     else
%         save(['RN_BalCl_MinBat10_' num2str(qq) '.mat'],'accuracy', 'AUC',...
%             'cm', 'Confusion_Matrix', 'cp',...
%             'outputs', 'per', 'scores','SUBY', 'SUBYNAMES', 'T', 'targets',...
%             'TheTableNice', 'validationImages', 'X', 'Y');
%         saveas(PrettyCM, ['BalCl_MB10_PrettyCM_' num2str(qq) '.jpg']);
%         
%         try
%             msg=sprintf('%s loop %d from %d MiniBatchSize %d , Another one bits the dust, Sensitivity = %d',...
%                 Name, qq, nLoops, miniBatchSize, cp.Se);
%             sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
%         catch
%         end
        % Close all the figures
        
    end
    
    close all
    delete(findall(0));
    cd(codeFolder);
    % clearvars -except codeFolder resultsFolder TrainingFolder ValidationFolder nLoops qq
    
end

% Close progress bar
ioi_text_waitbar('Clear');


nLoops = 1440;

for qq = 1431:nLoops
    
      
    ioi_text_waitbar(qq/nLoops, sprintf('ResNet101 Balanced MiniBatch=10, Running loop %d from %d', qq, 1440))
    
    close all
    trainingImages = imageDatastore(TrainingFolder, ...
        'IncludeSubfolders',true,...
        'FileExtensions','.tif','LabelSource','foldernames');
    
    trainingImages.ReadFcn = @(loc)imresize(imread(loc),[224, 224]);
    
    validationImages = imageDatastore(ValidationFolder, ...
        'IncludeSubfolders',true,...
        'FileExtensions','.tif','LabelSource','foldernames');
    
    validationImages.ReadFcn = @(loc)imresize(imread(loc),[224, 224]);
    
    
    % Load the pretrained GoogleNet network
    
    net = resnet101;
    net.Layers;
    
    % Extract the layer graph from the trained network and plot the layer graph.
    
    lgraph = layerGraph(net);
    Architecture_1 = figure('Units','normalized','Position',[0.1 0.1 0.8 0.8]);
    plot(lgraph);
    
       
    lgraph = removeLayers(lgraph, {'fc1000', 'prob', 'ClassificationLayer_predictions'});
    
    numClasses = numel(categories(trainingImages.Labels));
    newLayers = [
        fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',60,'BiasLearnRateFactor', 60)
        softmaxLayer('Name','softmax')
        classificationLayer('Name','classoutput')];
    lgraph = addLayers(lgraph,newLayers);
    
    lgraph = connectLayers(lgraph,'pool5','fc');
    
    Architecture_2 = figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
    plot(lgraph);
    ylim([0,10]);
    
    miniBatchSize = 10;
    
    %numIterationsPerEpoch = floor(numel(trainingImages.Labels)/miniBatchSize);
    options = trainingOptions('sgdm',...
        'ExecutionEnvironment','gpu',...
        'MiniBatchSize',miniBatchSize,...
        'MaxEpochs',100,...
        'InitialLearnRate',1e-5,...
        'VerboseFrequency',1,...
        'Plots','training-progress',...
        'ValidationData',validationImages,...
        'ValidationFrequency',5);
    netTransfer = trainNetwork(trainingImages,lgraph,options);
    
    %% Classify the validation images using the fine-tuned network.
    
    [predictedLabels, scores] = classify(netTransfer,validationImages,'ExecutionEnvironment','cpu');
    accuracy = mean(predictedLabels == validationImages.Labels);
    targets = grp2idx(validationImages.Labels);
    targets(targets==2) = 0;
    targets = ~targets;
    outputs = grp2idx(predictedLabels);
    outputs(outputs==2) = 0;
    outputs = ~outputs;
    [c,cm,ind,per] = confusion(targets', outputs');
    
    Confusion_Matrix=figure; plotconfusion(targets', outputs');
    
    %%
    
    addpath(genpath(resultsFolder))
    cp = my_classperf(cm);
    ROC_1=figure; plotroc(targets', scores(:,2)');
    
    [X,Y,T,AUC,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(targets', scores(:,2)',1,...
        'NBoot',1000,'TVals',0:0.05:1);
    ROC_2=figure; errorbar(X(:,3),Y(:,1),Y(:,1)-Y(:,2),Y(:,3)-Y(:,1));
    TheTableNice = table(validationImages.Labels,predictedLabels,'VariableNames',{'Actual','Predicted'});
    Confusion_Matrix1 = figure;
    PrettyCM = heatmap(TheTableNice,'Predicted','Actual');
    
    %     idx = randperm(20,4);
    %     Second_Figure = figure
    %     for i = 1:numel(idx)
    %         subplot(2,2,i)
    %         I = readimage(validationImages,idx(i));
    %         label = predictedLabels(idx(i));
    %         imshow(I)
    %         title(char(label))
    %     end
    
    try
        msg=sprintf('%s loop %d from %d MiniBatchSize %d with the results %d', Name, qq, nLoops, miniBatchSize, cp);
        sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
    catch
    end
    cd(resultsFolder);
    if  cp.Se > 0.79
        
        saveas(PrettyCM, ['BalCl_MB10_PrettyCM_' num2str(qq) '.jpg']);
        
        saveas(ROC_1, ['BalCl_MB10_ROC_1_' num2str(qq) '.jpg']);
        saveas(ROC_2, ['BalCl_MB10_ROC_2_' num2str(qq) '.jpg']);
        
        TrainingNeuNetw = findall(groot, 'Type', 'Figure');
        
        save(['RN_BalCl_MinBat10_' num2str(qq) '.mat'],'accuracy', 'AUC',...
            'c', 'cm', 'Confusion_Matrix', 'cp', ...
            'ind', 'miniBatchSize', 'netTransfer',...
            'numClasses', 'OPTROCPT', 'outputs', 'per', 'predictedLabels',...
            'scores','SUBY', 'SUBYNAMES', 'T', 'targets',...
            'TrainingNeuNetw','TheTableNice', 'validationImages', 'X', 'Y');
        %saveas(Architecture_1, ['UnCl_MB10_Architecture_1_' num2str(qq) '.jpg']);
        %saveas(Architecture_2, ['UnCl_MB10_Architecture_2_' num2str(qq) '.jpg']);
        
        %saveas(Second_Figure, ['UnCl_MB10_2nd_Figure_' num2str(qq) '.jpg']);
        
        
        
        try
            msg=sprintf('%s loop %d from %d MiniBatchSize %d , Sensitivity = %d , Balanced Accuracy = %d',...
                Name, qq, nLoops, miniBatchSize, cp.Se, cp.BA);
            sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
        catch
        end
        
%         close all
        % Save the figure of Training Progress in PNG format
        % saveas(TrainingNeuNetw,['RN_UnCl_MinBat10_' num2str(qq) '.png']);
        
        
%     else
%         save(['RN_BalCl_MinBat10_' num2str(qq) '.mat'],'accuracy', 'AUC',...
%             'cm', 'Confusion_Matrix', 'cp',...
%             'outputs', 'per', 'scores','SUBY', 'SUBYNAMES', 'T', 'targets',...
%             'TheTableNice', 'validationImages', 'X', 'Y');
%         saveas(PrettyCM, ['BalCl_MB10_PrettyCM_' num2str(qq) '.jpg']);
%         
%         try
%             msg=sprintf('%s loop %d from %d MiniBatchSize %d , Another one bits the dust, Sensitivity = %d',...
%                 Name, qq, nLoops, miniBatchSize, cp.Se);
%             sendmail('juan.carlos.torres@alumnos.uaslp.edu.mx', 'My Subjsect', msg);
%         catch
%         end
        % Close all the figures
        
    end
    
    close all
    delete(findall(0));
    cd(codeFolder);
    % clearvars -except codeFolder resultsFolder TrainingFolder ValidationFolder nLoops qq
    
end

% Close progress bar
ioi_text_waitbar('Clear');