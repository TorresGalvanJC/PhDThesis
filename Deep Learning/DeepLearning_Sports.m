%% Scene classification - revisited
% We still have 4 scenes we would like to identify, This time we want to
% use a combination approach
% We can use a pretrained CNN as a feature extractor
% and we can use a machine Learning algorithm to classify those features

% BIG PC Folders to use
% 
% codeFolder = 'C:\Users\Ram�n\Desktop\Juan Carlos Torres\Pr�ctica\Prueba Deep Learning';
% cd(codeFolder);
% ImagesFolder = 'C:\Users\Ram�n\Desktop\Juan Carlos Torres\Reducidas';

% LAPTOP Folders to use
codeFolder = 'C:\Users\ADMIN\Dropbox\Doctorado\MATLAB Codigos\Pr�ctica\Prueba Deep Learning';
 cd(codeFolder);
ImagesFolder = 'C:\Users\ADMIN\Documents\Universidad\Doctorado\Pruebas Deep Learning\Reducidas';


categories = {'B�squetbol'  ,  'B�isbol'  ,  'Futbol'   , 'Tauromaquia'};

%% Load in input images

ImageSports = imageDatastore(fullfile(ImagesFolder, categories), ...
    'LabelSource', 'foldernames')
tbl = countEachLabel(ImageSports)

%% Split each category into the same number of images

minSetCount = min(tbl{:,2}); % determine the smallest amount of images in a category

% Use splitEachLabel method to trim the set.

ImageSports = splitEachLabel(ImageSports, minSetCount, 'randomize');

% Notice that each set now has exactly the same number of images.

countEachLabel(ImageSports)

% Find the first instance of an image for each category

Basquetbol = find(ImageSports.Labels == categories(1), 1);
Beisbol = find(ImageSports.Labels == categories(2), 1);
Futbol = find(ImageSports.Labels == categories(3), 1);
Tauromaquia = find(ImageSports.Labels == categories(4), 1);

%% Visualize image in each set

figure
subplot(2,2,1); % For determinate the place of the images 2 row, 2 col, quadrant1)

imshow(ImageSports.Files{Basquetbol})
title(categories(1));

subplot(2,2,2);

imshow(ImageSports.Files{Beisbol})
title(categories(2));

subplot(2,2,3);
title(categories(3));
imshow(ImageSports.Files{Futbol})
title(categories(3));

subplot(2,2,4);
imshow(ImageSports.Files{Tauromaquia})
title(categories(4));

%% Load Pre-trained CNN
% The CNN model is saved in MatConvNet's format [3]. Load the MatConvNet
% network data into |convnet|, a |SeriesNetwork| object from Neural Network
% Toolbox(TM), using the helper function |helperImportMatConvNet|. A
% SeriesNetwork object can be used to inspect the network architecture,
% classify new data, and extract network activations from specific layers.

% Location of pre-trained "AlexNet"
cnnURL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-caffe-alex.mat';
% Store CNN model in a temporary folder
cnnMatFile = fullfile(tempdir, 'imagenet-caffe-alex.mat');

if ~exist(cnnMatFile, 'file') % download only once
    disp('Downloading pre-trained CNN model...');
    websave(cnnMatFile, cnnURL);
end

convnet = helperImportMatConvNet(cnnMatFile);

%%
% |convnet.Layers| defines the architecture of the CNN. 

% View the CNN architecture
convnet.Layers

%%
% The first layer defines the input dimensions. Each CNN has a different
% input size requirements. The one used in this example requires image
% input that is 227-by-227-by-3.

% Inspect the first layer
convnet.Layers(1)

%%
% The intermediate layers make up the bulk of the CNN. These are a series
% of convolutional layers, interspersed with rectified linear units (ReLU)
% and max-pooling layers [2]. Following the these layers are 3
% fully-connected layers.
%
% The final layer is the classification layer and its properties depend on
% the classification task. In this example, the CNN model that was loaded
% was trained to solve a 1000-way classification problem. Thus the
% classification layer has 1000 classes from the ImageNet dataset. 

% Inspect the last layer
convnet.Layers(end)

% Number of class names for ImageNet classification task
numel(convnet.Layers(end).ClassNames)

%%
% Note that the CNN model is not going to be used for the original
% classification task. It is going to be re-purposed to solve a different
% classification task on the Caltech 101 dataset.

%% Pre-process Images For CNN
% As mentioned above, |convnet| can only process RGB images that are
% 227-by-227. To avoid re-saving all the images in Caltech 101 to this
% format, setup the |imds| read function, |imds.ReadFcn|, to pre-process
% images on-the-fly. The |imds.ReadFcn| is called every time an image is
% read from the |ImageDatastore|.

% Set the ImageDatastore ReadFcn
ImageSports.ReadFcn = @(filename)readAndPreprocessImage(filename);

%% Prepare Training and Test Image Sets
% Split the sets into training and validation data. Pick 30% of images
% from each set for the training data and the remainder, 70%, for the
% validation data. Randomize the split to avoid biasing the results. The
% training and test sets will be processed by the CNN model.

[trainingSet, testSet] = splitEachLabel(ImageSports, 0.3, 'randomize');

%% Extract Features from the CNN
% Notice how the first layer of the network has learned filters for
% capturing blob and edge features. These "primitive" features are then
% processed by deeper network layers, which combine the early features to
% form higher level image features. These higher level features are better
% suited for recognition tasks because they combine all the primitive
% features into a richer image representation [5].
%
% You can easily extract features from one of the deeper layers using the
% |activations| method. Selecting which of the deep layers to choose is a
% design choice, but typically starting with the layer right before the
% classification layer is a good place to start. In |convnet|, the this
% layer is named 'fc7'. Let's extract training features using that layer.
featureLayer = 'fc7';
trainingFeatures = activations(convnet, trainingSet, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');

%%
% Note that the activations are computed on the GPU and the 'MiniBatchSize'
% is set 32 to ensure that the CNN and image data fit into GPU memory.
% You may need to lower the 'MiniBatchSize' if your GPU runs out of memory.
%
% Also, the activations output is arranged as columns. This helps speed-up
% the multiclass linear SVM training that follows.

%% Train A Multiclass SVM Classifier Using CNN Features
% Next, use the CNN image features to train a multiclass SVM classifier. A
% fast Stochastic Gradient Descent solver is used for training by setting
% the |fitcecoc| function's 'Learners' parameter to 'Linear'. This helps
% speed-up the training when working with high-dimensional CNN feature
% vectors, which each have a length of 4096.

% Get training labels from the trainingSet
trainingLabels = trainingSet.Labels;

% Train multiclass SVM classifier using a fast linear solver, and set
% 'ObservationsIn' to 'columns' to match the arrangement used for training
% features.
classifier = fitcecoc(trainingFeatures, trainingLabels, ...
    'Learners', 'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'columns');

%% Evaluate Classifier
% Repeat the procedure used earlier to extract image features from
% |testSet|. The test features can then be passed to the classifier to
% measure the accuracy of the trained classifier.

% Extract test features using the CNN
testFeatures = activations(convnet, testSet, featureLayer, 'MiniBatchSize',32);

% Pass CNN image features to trained classifier
predictedLabels = predict(classifier, testFeatures);

% Get the known labels
testLabels = testSet.Labels;

% Tabulate the results using a confusion matrix.
confMat = confusionmat(testLabels, predictedLabels);

% Convert confusion matrix into percentage form
confMat = bsxfun(@rdivide,confMat,sum(confMat,2));
%

% Display the mean accuracy
mean(diag(confMat))

%% Try the Newly Trained Classifier on Test Images
% You can now apply the newly trained classifier to categorize new images.
maxCount = size(testSet.Files,1);
randNum = randi(maxCount);
newImage = testSet.Files{randNum};

% Pre-process the images as required for the CNN
img = readAndPreprocessImage(newImage);

% Extract image features using the CNN
tic
imageFeatures = activations(convnet, img, featureLayer);
toc
%

% Make a prediction using the classifier
label = predict(classifier, imageFeatures);
testSet.Labels(randNum)

imshow(newImage);
if strcmp(char(label),char(testSet.Labels(randNum)))
	titleColor = [0 0.8 0];
else
	titleColor = 'r';
end
title(sprintf('Best Guess: %s; Actual: %s',...
	char(label),testSet.Labels(randNum)),...
	'color',titleColor)

