%% Transfer Learning CAM Zhou VGG 16 Breast Cancer Thermography

% Application of Transfer Learning with the VGG 16 architecture modified
% for Zhou for the problem of breast cancer, it's used for the generation
% of the file with the retrain

clear, close all, clc;

% Folders Location

% codeFolder = 'K:\Archivos Juan Carlos Torres\MATLAB\MATLAB Codigos\Práctica\Prueba Deep Learning\Data Augmented\Res Net 101';

resultsFolder = 'K:\Archivos Juan Carlos Torres\MATLAB\matconvnet-1.0-beta25\CAM Zhou';

TrainingFolder = 'K:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\Data Augmented Thermography\Data Augmented Brasil + IJC\200 times each image';
ValidationFolder = 'K:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\Data Augmented Thermography\train-val brasil + ijc frontal\Validation';


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


%% * Importing the model

protofile = 'K:\Archivos Juan Carlos Torres\MATLAB\matconvnet-1.0-beta25\deploy_vgg16CAM.prototxt';
datafile = 'K:\Archivos Juan Carlos Torres\MATLAB\matconvnet-1.0-beta25\vgg16CAM_train_iter_90000.caffemodel';
net = importCaffeNetwork(protofile,datafile)
net.Layers

% Modifying last three layers 
layersTransfer = net.Layers(1:end-3)

numClasses = numel(categories(trainingImages.Labels))
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,...
    'WeightLearnRateFactor', 20,...
    'BiasLearnRateFactor', 20)
    softmaxLayer
    classificationLayer];

miniBatchSize = 10
numIterationsPerEpoch = floor(numel(trainingImages.Labels)/miniBatchSize);
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
accuracy = mean(predictedLabels == validationImages.Labels)
targets = grp2idx(validationImages.Labels);
targets(targets==2) = 0;
targets = ~targets;
outputs = grp2idx(predictedLabels);
outputs(outputs==2) = 0;
outputs = ~outputs;
[c,cm,ind,per] = confusion(targets', outputs');

cd('K:\Archivos Juan Carlos Torres\MATLAB\matconvnet-1.0-beta25\CAM Zhou')
save('VGG16_CAM_Modified')



%% Evaluation of the CAM with the original code


% visualizationPointer = 0;
% 
% topCategoryNum = 2;
% predictionResult_bbox1 = zeros(nImgs, topCategoryNum*2);
% predictionResult_bbox2 = zeros(nImgs, topCategoryNum*2);
% predictionResult_bboxCombine = zeros(nImgs, topCategoryNum*5);
% 
% if matlabpool('size')==0
%     try
%         matlabpool
%     catch e
%     end
% end
% 
% heatMapFolder = ['heatMap-' datasetName '-' netName];
% bbox_threshold = [20, 100, 110];
% curParaThreshold = [num2str(bbox_threshold(1)) ' ' num2str(bbox_threshold(2)) ' ' num2str(bbox_threshold(3))];
% parfor i=1:size(imageList,1)
%     curImgIDX = i;
% 
%     height_original = sizeFull_imageList(curImgIDX,1);%tmp.Height;
%     weight_original = sizeFull_imageList(curImgIDX,2);%tmp.Width;
%     
%     [a b c] = fileparts(imageList{curImgIDX,1});
%     curPath_fullSizeImg = ['/data/vision/torralba/deeplearning/imagenet_toolkit/ILSVRC2012_img_val/' b c];
%     curMatFile = [heatMapFolder '/' b '.mat'];
%     [heatMapSet, value_category, IDX_category] = loadHeatMap( curMatFile);
%     
%     curResult_bbox1 = [];
%     curResult_bbox2 = [];
%     curResult_bboxCombine = [];
%     for j=1:5
%         curHeatMapFile = [heatMapFolder '/top' num2str(j) '/' b '.jpg'];
% 
%         curBBoxFile = [heatMapFolder '/top' num2str(j) '/' b '_default.txt'];
%         %curBBoxFileGraphcut = [heatMapFolder '/top' num2str(j) '/' b '_graphcut.txt'];
%         curCategory = categories{IDX_category(j),1};
%         %imwrite(curHeatMap, ['result_bbox/heatmap_tmp' b randString '.jpg']);
%         if ~exist(curBBoxFile)
%             %system(['/data/vision/torralba/deeplearning/package/bbox_hui/final ' curHeatMapFile ' ' curBBoxFile]);
%             
%             system(['/data/vision/torralba/deeplearning/package/bbox_hui_new/./dt_box ' curHeatMapFile ' ' curParaThreshold ' ' curBBoxFile]);
%         end
%         curPredictCategory = categories{IDX_category(j),1};
%         curPredictCategoryID = categories{IDX_category(j),1}(1:9);
%         curPredictCategoryGTID = categoryIDMap(curPredictCategoryID);
%         
%         
%         boxData = dlmread(curBBoxFile);
%         boxData_formulate = [boxData(1:4:end)' boxData(2:4:end)' boxData(1:4:end)'+boxData(3:4:end)' boxData(2:4:end)'+boxData(4:4:end)'];
%         boxData_formulate = [min(boxData_formulate(:,1),boxData_formulate(:,3)),min(boxData_formulate(:,2),boxData_formulate(:,4)),max(boxData_formulate(:,1),boxData_formulate(:,3)),max(boxData_formulate(:,2),boxData_formulate(:,4))];
%            
% %         try
% %             boxDataGraphcut = dlmread(curBBoxFileGraphcut);
% %             boxData_formulateGraphcut = [boxDataGraphcut(1:4:end)' boxDataGraphcut(2:4:end)' boxDataGraphcut(1:4:end)'+boxDataGraphcut(3:4:end)' boxDataGraphcut(2:4:end)'+boxDataGraphcut(4:4:end)'];
% %         catch exception
% %             boxDataGraphcut = dlmread(curBBoxFile);
% %             boxData_formulateGraphcut = [boxDataGraphcut(1:4:end)' boxDataGraphcut(2:4:end)' boxDataGraphcut(1:4:end)'+boxDataGraphcut(3:4:end)' boxDataGraphcut(2:4:end)'+boxDataGraphcut(4:4:end)'];
% %             boxData_formulateGraphcut = boxData_formulateGraphcut(1,:);
% %         end
% 
%         bbox = boxData_formulate(1,:); 
%         curPredictTuple = [curPredictCategoryGTID bbox(1) bbox(2) bbox(3) bbox(4)];
%         curResult_bbox1 = [curResult_bbox1 curPredictTuple];
%         curResult_bboxCombine = [curResult_bboxCombine curPredictTuple];
%         
%         bbox = boxData_formulate(2,:); 
%         %bbox = boxData_formulateGraphcut(1,:);
%         curPredictTuple = [curPredictCategoryGTID bbox(1) bbox(2) bbox(3) bbox(4)];
%         curResult_bbox2 = [curResult_bbox2 curPredictTuple];      
%         
%         curResult_bboxCombine = [curResult_bboxCombine curPredictTuple];
%         if visualizationPointer == 1
%               
%             curHeatMap = imread(curHeatMapFile);
%             curHeatMap = imresize(curHeatMap,[height_original weight_original]);
%         
%             subplot(1,2,1),hold off, imshow(curPath_fullSizeImg);
%             hold on
%             curBox = boxData_formulate(1,:);
%             rectangle('Position',[curBox(1) curBox(2) curBox(3)-curBox(1) curBox(4)-curBox(2)],'EdgeColor',[1 0 0]);
%             subplot(1,2,2),imagesc(curHeatMap);
%             title(curCategory);
%             waitforbuttonpress
%         end
%     end
%     
%     predictionResult_bbox1(i, :) = curResult_bbox1;
%     predictionResult_bbox2(i, :) = curResult_bbox2;
%     predictionResult_bboxCombine(i,:) = curResult_bboxCombine(1:topCategoryNum*5);
%     disp([netName ' processing ' b])
% end
% 
% 
% addpath('evaluation');
% disp([netName '--------bbox1' ]);
% [cls_error, clsloc_error] = simpleEvaluation(predictionResult_bbox1);
% disp([(1:5)',clsloc_error,cls_error]);
% 
% disp([netName '--------bbox2' ]);
% [cls_error, clsloc_error] = simpleEvaluation(predictionResult_bbox2);
% disp([(1:5)',clsloc_error,cls_error]);
% 
% disp([netName '--------bboxCombine' ]);
% [cls_error, clsloc_error] = simpleEvaluation(predictionResult_bboxCombine);
% disp([(1:5)',clsloc_error,cls_error]);

