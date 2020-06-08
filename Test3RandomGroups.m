%% Test 3 Random Groups for compare the Sensibility in function of the number of thermograms evaluated

ResultsFolder = 'C:\Users\ADMIN\Documents\Universidad\Doctorado\Termogramas\Termogramas TIF\Random Test\Results';

addpath('C:\Users\ADMIN\Dropbox\Doctorado\MATLAB Codigos');

%% Unbalanced Test
% Load the best result of the net
load('E:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\Results\ResNet101\Iteraciones\Unbalanced\RN_UnCl_MinBat10_448.mat', 'netTransfer');
% delete(findall(0));

%% Group A

ValidationFolder = 'C:\Users\ADMIN\Documents\Universidad\Doctorado\Termogramas\Termogramas TIF\Random Test\Grupo A\Unbalanced';

validationImages = imageDatastore(ValidationFolder, ...
    'IncludeSubfolders',true,...
    'FileExtensions','.tif','LabelSource','foldernames');

validationImages.ReadFcn = @(loc)imresize(imread(loc),[224, 224]);

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

cp = my_classperf(cm);
ROC_1 = figure; plotroc(targets', scores(:,2)');
[X,Y,T,AUC,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(targets', scores(:,2)',1,...
    'NBoot',1000,'TVals',0:0.05:1);
ROC_2 = figure; errorbar(X(:,3),Y(:,1),Y(:,1)-Y(:,2),Y(:,3)-Y(:,1));

NiceTable_UnbalGpoA = table(validationImages.Labels,predictedLabels,'VariableNames',{'Actual','Predicted'});

Confusion_Matrix1 = figure;

PrettyCM = heatmap(NiceTable_UnbalGpoA,'Predicted','Actual');

cd(ResultsFolder)

saveas(PrettyCM, ['PrettyCM_Unbal_GpoA.jpg']);
saveas(ROC_1, ['ROC_1_Unbal_GpoA.jpg']);
saveas(ROC_2, ['ROC_2_Unbal_GpoA.jpg']);
save(['Unbal_GpoA.mat'],'accuracy', 'AUC', 'c', 'cm', 'Confusion_Matrix', 'cp', ...
            'ind', 'netTransfer', 'OPTROCPT', 'outputs', 'per', 'predictedLabels',...
            'scores','SUBY', 'SUBYNAMES', 'T', 'targets', 'NiceTable_UnbalGpoA', 'validationImages', 'X', 'Y');

clearvars -except netTransfer ResultsFolder

%% Group B

ValidationFolder = 'C:\Users\ADMIN\Documents\Universidad\Doctorado\Termogramas\Termogramas TIF\Random Test\Grupo B\Unbalanced';

validationImages = imageDatastore(ValidationFolder, ...
    'IncludeSubfolders',true,...
    'FileExtensions','.tif','LabelSource','foldernames');

validationImages.ReadFcn = @(loc)imresize(imread(loc),[224, 224]);

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

cp = my_classperf(cm);
ROC_1 = figure; plotroc(targets', scores(:,2)');
[X,Y,T,AUC,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(targets', scores(:,2)',1,...
    'NBoot',1000,'TVals',0:0.05:1);
ROC_2 = figure; errorbar(X(:,3),Y(:,1),Y(:,1)-Y(:,2),Y(:,3)-Y(:,1));

NiceTable_UnbalGpoB = table(validationImages.Labels,predictedLabels,'VariableNames',{'Actual','Predicted'});

Confusion_Matrix1 = figure;

PrettyCM = heatmap(NiceTable_UnbalGpoB,'Predicted','Actual');

cd(ResultsFolder)

saveas(PrettyCM, ['PrettyCM_Unbal_GpoB.jpg']);
saveas(ROC_1, ['ROC_1_Unbal_GpoB.jpg']);
saveas(ROC_2, ['ROC_2_Unbal_GpoB.jpg']);
save(['Unbal_GpoB.mat'],'accuracy', 'AUC', 'c', 'cm', 'Confusion_Matrix', 'cp', ...
            'ind', 'netTransfer', 'OPTROCPT', 'outputs', 'per', 'predictedLabels',...
            'scores','SUBY', 'SUBYNAMES', 'T', 'targets', 'NiceTable_UnbalGpoB', 'validationImages', 'X', 'Y');

clearvars -except netTransfer ResultsFolder

%% Group C

ValidationFolder = 'C:\Users\ADMIN\Documents\Universidad\Doctorado\Termogramas\Termogramas TIF\Random Test\Grupo C\Unbalanced';

validationImages = imageDatastore(ValidationFolder, ...
    'IncludeSubfolders',true,...
    'FileExtensions','.tif','LabelSource','foldernames');

validationImages.ReadFcn = @(loc)imresize(imread(loc),[224, 224]);

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

cp = my_classperf(cm);
ROC_1 = figure; plotroc(targets', scores(:,2)');
[X,Y,T,AUC,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(targets', scores(:,2)',1,...
    'NBoot',1000,'TVals',0:0.05:1);
ROC_2 = figure; errorbar(X(:,3),Y(:,1),Y(:,1)-Y(:,2),Y(:,3)-Y(:,1));

NiceTable_UnbalGpoC = table(validationImages.Labels,predictedLabels,'VariableNames',{'Actual','Predicted'});

Confusion_Matrix1 = figure;

PrettyCM = heatmap(NiceTable_UnbalGpoC,'Predicted','Actual');

cd(ResultsFolder)

saveas(PrettyCM, ['PrettyCM_Unbal_GpoC.jpg']);
saveas(ROC_1, ['ROC_1_Unbal_GpoC.jpg']);
saveas(ROC_2, ['ROC_2_Unbal_GpoC.jpg']);
save(['Unbal_GpoC.mat'],'accuracy', 'AUC', 'c', 'cm', 'Confusion_Matrix', 'cp', ...
            'ind', 'netTransfer','OPTROCPT', 'outputs', 'per', 'predictedLabels',...
            'scores','SUBY', 'SUBYNAMES', 'T', 'targets', 'NiceTable_UnbalGpoC', 'validationImages', 'X', 'Y');

clearvars -except netTransfer ResultsFolder


%% Balanced Test
% Load the best result of the net
load('E:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\Results\ResNet101\Iteraciones\Balanced\RN_BalCl_MinBat10_1239.mat', 'netTransfer');

%% Group A Balanced

ValidationFolder = 'C:\Users\ADMIN\Documents\Universidad\Doctorado\Termogramas\Termogramas TIF\Random Test\Grupo A\Balanced';

validationImages = imageDatastore(ValidationFolder, ...
    'IncludeSubfolders',true,...
    'FileExtensions','.tif','LabelSource','foldernames');

validationImages.ReadFcn = @(loc)imresize(imread(loc),[224, 224]);

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

cp = my_classperf(cm);
ROC_1 = figure; plotroc(targets', scores(:,2)');
[X,Y,T,AUC,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(targets', scores(:,2)',1,...
    'NBoot',1000,'TVals',0:0.05:1);
ROC_2 = figure; errorbar(X(:,3),Y(:,1),Y(:,1)-Y(:,2),Y(:,3)-Y(:,1));

NiceTable_BalGpoA = table(validationImages.Labels,predictedLabels,'VariableNames',{'Actual','Predicted'});

Confusion_Matrix1 = figure;

PrettyCM = heatmap(NiceTable_BalGpoA,'Predicted','Actual');

cd(ResultsFolder)

saveas(PrettyCM, ['PrettyCM_Bal_GpoA.jpg']);
saveas(ROC_1, ['ROC_1_Bal_GpoA.jpg']);
saveas(ROC_2, ['ROC_2_Bal_GpoA.jpg']);
save(['Bal_GpoA.mat'],'accuracy', 'AUC', 'c', 'cm', 'Confusion_Matrix', 'cp', ...
            'ind', 'netTransfer', 'OPTROCPT', 'outputs', 'per', 'predictedLabels',...
            'scores','SUBY', 'SUBYNAMES', 'T', 'targets', 'NiceTable_BalGpoA', 'validationImages', 'X', 'Y');

clearvars -except netTransfer ResultsFolder

%% Group B Balanced

ValidationFolder = 'C:\Users\ADMIN\Documents\Universidad\Doctorado\Termogramas\Termogramas TIF\Random Test\Grupo B\Balanced';

validationImages = imageDatastore(ValidationFolder, ...
    'IncludeSubfolders',true,...
    'FileExtensions','.tif','LabelSource','foldernames');

validationImages.ReadFcn = @(loc)imresize(imread(loc),[224, 224]);

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

cp = my_classperf(cm);
ROC_1 = figure; plotroc(targets', scores(:,2)');
[X,Y,T,AUC,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(targets', scores(:,2)',1,...
    'NBoot',1000,'TVals',0:0.05:1);
ROC_2 = figure; errorbar(X(:,3),Y(:,1),Y(:,1)-Y(:,2),Y(:,3)-Y(:,1));

NiceTable_BalGpoB = table(validationImages.Labels,predictedLabels,'VariableNames',{'Actual','Predicted'});

Confusion_Matrix1 = figure;

PrettyCM = heatmap(NiceTable_BalGpoB,'Predicted','Actual');

cd(ResultsFolder)

saveas(PrettyCM, ['PrettyCM_Bal_GpoB.jpg']);
saveas(ROC_1, ['ROC_1_Bal_GpoB.jpg']);
saveas(ROC_2, ['ROC_2_Bal_GpoB.jpg']);
save(['Bal_GpoB.mat'],'accuracy', 'AUC', 'c', 'cm', 'Confusion_Matrix', 'cp', ...
            'ind', 'netTransfer', 'OPTROCPT', 'outputs', 'per', 'predictedLabels',...
            'scores','SUBY', 'SUBYNAMES', 'T', 'targets', 'NiceTable_BalGpoB', 'validationImages', 'X', 'Y');

clearvars -except netTransfer ResultsFolder

%% Group C Balanced

ValidationFolder = 'C:\Users\ADMIN\Documents\Universidad\Doctorado\Termogramas\Termogramas TIF\Random Test\Grupo C\Balanced';

validationImages = imageDatastore(ValidationFolder, ...
    'IncludeSubfolders',true,...
    'FileExtensions','.tif','LabelSource','foldernames');

validationImages.ReadFcn = @(loc)imresize(imread(loc),[224, 224]);

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

cp = my_classperf(cm);
ROC_1 = figure; plotroc(targets', scores(:,2)');
[X,Y,T,AUC,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(targets', scores(:,2)',1,...
    'NBoot',1000,'TVals',0:0.05:1);
ROC_2 = figure; errorbar(X(:,3),Y(:,1),Y(:,1)-Y(:,2),Y(:,3)-Y(:,1));

NiceTable_BalGpoC = table(validationImages.Labels,predictedLabels,'VariableNames',{'Actual','Predicted'});

Confusion_Matrix1 = figure;

PrettyCM = heatmap(NiceTable_BalGpoC,'Predicted','Actual');

cd(ResultsFolder)

saveas(PrettyCM, ['PrettyCM_Bal_GpoC.jpg']);
saveas(ROC_1, ['ROC_1_Bal_GpoC.jpg']);
saveas(ROC_2, ['ROC_2_Bal_GpoC.jpg']);
save(['Bal_GpoC.mat'],'accuracy', 'AUC', 'c', 'cm', 'Confusion_Matrix', 'cp', ...
            'ind', 'netTransfer', 'OPTROCPT', 'outputs', 'per', 'predictedLabels',...
            'scores','SUBY', 'SUBYNAMES', 'T', 'targets', 'NiceTable_BalGpoC', 'validationImages', 'X', 'Y');

