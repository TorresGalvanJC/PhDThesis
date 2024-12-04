%% Visualize Activations of a Convolutional Neural Network

net = alexnet; 
net.Layers

% Read and show an image. Save its size for future use.

im = imread(fullfile(matlabroot,'examples','nnet','face.jpg'));
imshow(im)
imgSize = size(im);
imgSize = imgSize(1:2);

% Show Activations of First Convolutional Layer
act1 = activations(net,im,'conv1','OutputAs','channels');
sz = size(act1);
act1 = reshape(act1,[sz(1) sz(2) 1 sz(3)]);
% it's 1 because activation don't have color
montage(mat2gray(act1),'Size',[8 12])

% Investigate activation in specific channel
act1ch32 = act1(:,:,:,32);
act1ch32 = mat2gray(act1ch32);
act1ch32 = imresize(act1ch32,imgSize);
imshowpair(im,act1ch32,'montage')

% Find the strongest activation channel
figure
[maxValue,maxValueIndex] = max(max(max(act1)));
act1chMax = act1(:,:,:,maxValueIndex);
act1chMax = mat2gray(act1chMax);
act1chMax = imresize(act1chMax,imgSize);
imshowpair(im,act1chMax,'montage')

% Investigate a deeper layer (Left to Right)
act5 = activations(net,im,'conv5','OutputAs','channels');
sz = size(act5);
act5 = reshape(act5,[sz(1) sz(2) 1 sz(3)]);
montage(imresize(mat2gray(act5),[48 48]))

% Displaying the strongest activation
[maxValue5,maxValueIndex5] = max(max(max(act5)));
act5chMax = act5(:,:,:,maxValueIndex5);
imshow(imresize(mat2gray(act5chMax),imgSize))

% It's possible to select the most interesant channel 


% Internet web site for help
% https://la.mathworks.com/help/nnet/examples/visualize-activations-of-a-convolutional-neural-network.html