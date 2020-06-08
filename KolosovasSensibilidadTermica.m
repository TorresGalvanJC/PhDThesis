% *Sensibilidad Térmica*

% Evaluación al redondear los valores de la sensibilidad térmica de las
% % cámaras FLIR y su cambio en la clasificación de termogramas

clc; clear

load('E:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\Results\ResNet101\Mini10\Unbalanced\RN_UnCl_MinBat10_24.mat');
delete(findall(0));

imagesFolder = 'C:\Users\ADMIN\Documents\Universidad\Doctorado\Termogramas\Termograma Redondeado\Redondeado 040';
dirStruct = dir(fullfile(imagesFolder, '*.tif'));
nImages = numel(dirStruct);
imageName = cell([nImages, 1]);
NetPrediction2 = cell([nImages, 1]);
NetScores2 = zeros(31,1);;

ioi_text_waitbar(0, 'Please wait...');
for iImages = 1:nImages
    currentFile = dirStruct(iImages).name;
    TheImage = imread(fullfile(imagesFolder, currentFile));
    TheImage2 = imresize(TheImage,[224 224]);
    [NetPrediction, NetScores] = classify(netTransfer,TheImage2,'ExecutionEnvironment','cpu');
    [s,idx] = sort(NetScores(:),'descend');
                    idx = idx(1:1:2);
                    classNamesTop = netTransfer.Layers(end).ClassNames(idx);
    NetPrediction2{iImages} = cellstr(NetPrediction);         
    NetScores = NetScores(idx)*100;
    NetScores = round(NetScores,2);
    NetScores2(iImages) = (NetScores(1,1));
    imageName{iImages} = {currentFile(1:7)};
    ioi_text_waitbar(iImages/nImages, sprintf('Reading image %d from %d', iImages, nImages));
end
ioi_text_waitbar('Clear');

TableResults040 = table(imageName, NetPrediction2, NetScores2);

%%

imagesFolder = 'C:\Users\ADMIN\Documents\Universidad\Doctorado\Termogramas\Termograma Redondeado\Redondeado 045';
dirStruct = dir(fullfile(imagesFolder, '*.tif'));
nImages = numel(dirStruct);
imageName = cell([nImages, 1]);
NetPrediction2 = cell([nImages, 1]);
NetScores2 = zeros(31,1);;

ioi_text_waitbar(0, 'Please wait...');
for iImages = 1:nImages
    currentFile = dirStruct(iImages).name;
    TheImage = imread(fullfile(imagesFolder, currentFile));
    TheImage2 = imresize(TheImage,[224 224]);
    [NetPrediction, NetScores] = classify(netTransfer,TheImage2,'ExecutionEnvironment','cpu');
    [s,idx] = sort(NetScores(:),'descend');
                    idx = idx(1:1:2);
                    classNamesTop = netTransfer.Layers(end).ClassNames(idx);
    NetPrediction2{iImages} = cellstr(NetPrediction);         
    NetScores = NetScores(idx)*100;
    NetScores = round(NetScores,2);
    NetScores2(iImages) = (NetScores(1,1));
    imageName{iImages} = {currentFile(1:7)};
    ioi_text_waitbar(iImages/nImages, sprintf('Reading image %d from %d', iImages, nImages));
end
ioi_text_waitbar('Clear');

TableResults045 = table(imageName, NetPrediction2, NetScores2);


%%
imagesFolder = 'C:\Users\ADMIN\Documents\Universidad\Doctorado\Termogramas\Termograma Redondeado\Redondeado 050';
dirStruct = dir(fullfile(imagesFolder, '*.tif'));
nImages = numel(dirStruct);
imageName = cell([nImages, 1]);
NetPrediction2 = cell([nImages, 1]);
NetScores2 = zeros(31,1);;

ioi_text_waitbar(0, 'Please wait...');
for iImages = 1:nImages
    currentFile = dirStruct(iImages).name;
    TheImage = imread(fullfile(imagesFolder, currentFile));
    TheImage2 = imresize(TheImage,[224 224]);
    [NetPrediction, NetScores] = classify(netTransfer,TheImage2,'ExecutionEnvironment','cpu');
    [s,idx] = sort(NetScores(:),'descend');
                    idx = idx(1:1:2);
                    classNamesTop = netTransfer.Layers(end).ClassNames(idx);
    NetPrediction2{iImages} = cellstr(NetPrediction);         
    NetScores = NetScores(idx)*100;
    NetScores = round(NetScores,2);
    NetScores2(iImages) = (NetScores(1,1));
    imageName{iImages} = {currentFile(1:7)};
    ioi_text_waitbar(iImages/nImages, sprintf('Reading image %d from %d', iImages, nImages));
end
ioi_text_waitbar('Clear');

TableResults050 = table(imageName, NetPrediction2, NetScores2);

%%

imagesFolder = 'C:\Users\ADMIN\Documents\Universidad\Doctorado\Termogramas\Termograma Redondeado\Redondeado 080';
dirStruct = dir(fullfile(imagesFolder, '*.tif'));
nImages = numel(dirStruct);
imageName = cell([nImages, 1]);
NetPrediction2 = cell([nImages, 1]);
NetScores2 = zeros(31,1);;

ioi_text_waitbar(0, 'Please wait...');
for iImages = 1:nImages
    currentFile = dirStruct(iImages).name;
    TheImage = imread(fullfile(imagesFolder, currentFile));
    TheImage2 = imresize(TheImage,[224 224]);
    [NetPrediction, NetScores] = classify(netTransfer,TheImage2,'ExecutionEnvironment','cpu');
    [s,idx] = sort(NetScores(:),'descend');
                    idx = idx(1:1:2);
                    classNamesTop = netTransfer.Layers(end).ClassNames(idx);
    NetPrediction2{iImages} = cellstr(NetPrediction);         
    NetScores = NetScores(idx)*100;
    NetScores = round(NetScores,2);
    NetScores2(iImages) = (NetScores(1,1));
    imageName{iImages} = {currentFile(1:7)};
    ioi_text_waitbar(iImages/nImages, sprintf('Reading image %d from %d', iImages, nImages));
end
ioi_text_waitbar('Clear');

TableResults080 = table(imageName, NetPrediction2, NetScores2);

%%

imagesFolder = 'C:\Users\ADMIN\Documents\Universidad\Doctorado\Termogramas\Termograma Redondeado\Redondeado 100';
dirStruct = dir(fullfile(imagesFolder, '*.tif'));
nImages = numel(dirStruct);
imageName = cell([nImages, 1]);
NetPrediction2 = cell([nImages, 1]);
NetScores2 = zeros(31,1);;

ioi_text_waitbar(0, 'Please wait...');
for iImages = 1:nImages
    currentFile = dirStruct(iImages).name;
    TheImage = imread(fullfile(imagesFolder, currentFile));
    TheImage2 = imresize(TheImage,[224 224]);
    [NetPrediction, NetScores] = classify(netTransfer,TheImage2,'ExecutionEnvironment','cpu');
    [s,idx] = sort(NetScores(:),'descend');
                    idx = idx(1:1:2);
                    classNamesTop = netTransfer.Layers(end).ClassNames(idx);
    NetPrediction2{iImages} = cellstr(NetPrediction);         
    NetScores = NetScores(idx)*100;
    NetScores = round(NetScores,2);
    NetScores2(iImages) = (NetScores(1,1));
    imageName{iImages} = {currentFile(1:7)};
    ioi_text_waitbar(iImages/nImages, sprintf('Reading image %d from %d', iImages, nImages));
end
ioi_text_waitbar('Clear');

TableResults100 = table(imageName, NetPrediction2, NetScores2);

%%
load('E:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\ResNet 101\Balanced Classes Brasil + IJC Results\Calis_50.mat');
delete(findall(0));

imagesFolder = 'C:\Users\ADMIN\Documents\Universidad\Doctorado\Termogramas\Termograma Redondeado\Redondeado 040';
dirStruct = dir(fullfile(imagesFolder, '*.tif'));
nImages = numel(dirStruct);
imageName = cell([nImages, 1]);
NetPrediction2 = cell([nImages, 1]);
NetScores2 = zeros(31,1);;

ioi_text_waitbar(0, 'Please wait...');
for iImages = 1:nImages
    currentFile = dirStruct(iImages).name;
    TheImage = imread(fullfile(imagesFolder, currentFile));
    TheImage2 = imresize(TheImage,[224 224]);
    [NetPrediction, NetScores] = classify(netTransfer,TheImage2,'ExecutionEnvironment','cpu');
    [s,idx] = sort(NetScores(:),'descend');
                    idx = idx(1:1:2);
                    classNamesTop = netTransfer.Layers(end).ClassNames(idx);
    NetPrediction2{iImages} = cellstr(NetPrediction);         
    NetScores = NetScores(idx)*100;
    NetScores = round(NetScores,2);
    NetScores2(iImages) = (NetScores(1,1));
    imageName{iImages} = {currentFile(1:7)};
    ioi_text_waitbar(iImages/nImages, sprintf('Reading image %d from %d', iImages, nImages));
end
ioi_text_waitbar('Clear');

TableResults1040 = table(imageName, NetPrediction2, NetScores2);

%%

imagesFolder = 'C:\Users\ADMIN\Documents\Universidad\Doctorado\Termogramas\Termograma Redondeado\Redondeado 045';
dirStruct = dir(fullfile(imagesFolder, '*.tif'));
nImages = numel(dirStruct);
imageName = cell([nImages, 1]);
NetPrediction2 = cell([nImages, 1]);
NetScores2 = zeros(31,1);;

ioi_text_waitbar(0, 'Please wait...');
for iImages = 1:nImages
    currentFile = dirStruct(iImages).name;
    TheImage = imread(fullfile(imagesFolder, currentFile));
    TheImage2 = imresize(TheImage,[224 224]);
    [NetPrediction, NetScores] = classify(netTransfer,TheImage2,'ExecutionEnvironment','cpu');
    [s,idx] = sort(NetScores(:),'descend');
                    idx = idx(1:1:2);
                    classNamesTop = netTransfer.Layers(end).ClassNames(idx);
    NetPrediction2{iImages} = cellstr(NetPrediction);         
    NetScores = NetScores(idx)*100;
    NetScores = round(NetScores,2);
    NetScores2(iImages) = (NetScores(1,1));
    imageName{iImages} = {currentFile(1:7)};
    ioi_text_waitbar(iImages/nImages, sprintf('Reading image %d from %d', iImages, nImages));
end
ioi_text_waitbar('Clear');

TableResults1045 = table(imageName, NetPrediction2, NetScores2);


%%
imagesFolder = 'C:\Users\ADMIN\Documents\Universidad\Doctorado\Termogramas\Termograma Redondeado\Redondeado 050';
dirStruct = dir(fullfile(imagesFolder, '*.tif'));
nImages = numel(dirStruct);
imageName = cell([nImages, 1]);
NetPrediction2 = cell([nImages, 1]);
NetScores2 = zeros(31,1);;

ioi_text_waitbar(0, 'Please wait...');
for iImages = 1:nImages
    currentFile = dirStruct(iImages).name;
    TheImage = imread(fullfile(imagesFolder, currentFile));
    TheImage2 = imresize(TheImage,[224 224]);
    [NetPrediction, NetScores] = classify(netTransfer,TheImage2,'ExecutionEnvironment','cpu');
    [s,idx] = sort(NetScores(:),'descend');
                    idx = idx(1:1:2);
                    classNamesTop = netTransfer.Layers(end).ClassNames(idx);
    NetPrediction2{iImages} = cellstr(NetPrediction);         
    NetScores = NetScores(idx)*100;
    NetScores = round(NetScores,2);
    NetScores2(iImages) = (NetScores(1,1));
    imageName{iImages} = {currentFile(1:7)};
    ioi_text_waitbar(iImages/nImages, sprintf('Reading image %d from %d', iImages, nImages));
end
ioi_text_waitbar('Clear');

TableResults1050 = table(imageName, NetPrediction2, NetScores2);

%%

imagesFolder = 'C:\Users\ADMIN\Documents\Universidad\Doctorado\Termogramas\Termograma Redondeado\Redondeado 080';
dirStruct = dir(fullfile(imagesFolder, '*.tif'));
nImages = numel(dirStruct);
imageName = cell([nImages, 1]);
NetPrediction2 = cell([nImages, 1]);
NetScores2 = zeros(31,1);;

ioi_text_waitbar(0, 'Please wait...');
for iImages = 1:nImages
    currentFile = dirStruct(iImages).name;
    TheImage = imread(fullfile(imagesFolder, currentFile));
    TheImage2 = imresize(TheImage,[224 224]);
    [NetPrediction, NetScores] = classify(netTransfer,TheImage2,'ExecutionEnvironment','cpu');
    [s,idx] = sort(NetScores(:),'descend');
                    idx = idx(1:1:2);
                    classNamesTop = netTransfer.Layers(end).ClassNames(idx);
    NetPrediction2{iImages} = cellstr(NetPrediction);         
    NetScores = NetScores(idx)*100;
    NetScores = round(NetScores,2);
    NetScores2(iImages) = (NetScores(1,1));
    imageName{iImages} = {currentFile(1:7)};
    ioi_text_waitbar(iImages/nImages, sprintf('Reading image %d from %d', iImages, nImages));
end
ioi_text_waitbar('Clear');

TableResults1080 = table(imageName, NetPrediction2, NetScores2);

%%

imagesFolder = 'C:\Users\ADMIN\Documents\Universidad\Doctorado\Termogramas\Termograma Redondeado\Redondeado 100';
dirStruct = dir(fullfile(imagesFolder, '*.tif'));
nImages = numel(dirStruct);
imageName = cell([nImages, 1]);
NetPrediction2 = cell([nImages, 1]);
NetScores2 = zeros(31,1);;

ioi_text_waitbar(0, 'Please wait...');
for iImages = 1:nImages
    currentFile = dirStruct(iImages).name;
    TheImage = imread(fullfile(imagesFolder, currentFile));
    TheImage2 = imresize(TheImage,[224 224]);
    [NetPrediction, NetScores] = classify(netTransfer,TheImage2,'ExecutionEnvironment','cpu');
    [s,idx] = sort(NetScores(:),'descend');
                    idx = idx(1:1:2);
                    classNamesTop = netTransfer.Layers(end).ClassNames(idx);
    NetPrediction2{iImages} = cellstr(NetPrediction);   
    NetScores = NetScores(idx)*100;
    NetScores = round(NetScores,2);
    NetScores2(iImages) = (NetScores(1,1));
    imageName{iImages} = {currentFile(1:7)};
    ioi_text_waitbar(iImages/nImages, sprintf('Reading image %d from %d', iImages, nImages));
end
ioi_text_waitbar('Clear');

TableResults1100 = table(imageName, NetPrediction2, NetScores2);

cd('C:\Users\ADMIN\Documents\Universidad\Doctorado\Termogramas\Termograma Redondeado');
save('KolosovasQuestion.mat','TableResults040', 'TableResults045',...
            'TableResults050', 'TableResults080', 'TableResults100',...
            'TableResults1040', 'TableResults1045',...
            'TableResults1050', 'TableResults1080', 'TableResults1100', 'TMatrix040',...
            'TMatrix045', 'TMatrix050','TMatrix080','TMatrix100','TMatrix1040',...
            'TMatrix1045','TMatrix1050','TMatrix1080','TMatrix1100');
        

% HolisA = TableResults040{1:31,{'NetScores2'}};
% Matrix040 = zeros(31,1);
% for iNumber=1:31
%     HolisAA = HolisA{iNumber,1};
%     HolisAAA = cell2mat(HolisAA);
%     Matrix040(iNumber,1) = HolisAAA
% end
% TMatrix040 = array2table(Matrix040);
% T = addvars(TableResults040, 'T')
% TableResults0040 = [TableResults040 TMatrix040]
% TableResults0040 = removevars(TableResults040,{'NetScores2'});
% %%
% HolisA = TableResults1040{1:31,{'NetScores2'}}
% Matrix1040 = zeros(31,1);
% for iNumber=1:31
%     HolisAA = HolisA{iNumber,1};
%     HolisAAA = cell2mat(HolisAA);
%     Matrix1040(iNumber) = HolisAAA
% end
% TMatrix1040 = array2table(Matrix1040);
% TableResults01040 = [TableResults1040 TMatrix1040]
% TableResults1040 = removevars(TableResults1040,{'NetScores2'});
% 
% %%
% HolisA = TableResults045{1:31,{'NetScores2'}}
% Matrix045 = zeros(31,1);
% for iNumber=1:31
%     HolisAA = HolisA{iNumber,1};
%     HolisAAA = cell2mat(HolisAA);
%     Matrix045(iNumber) = HolisAAA
% end
% TMatrix045 = array2table(Matrix045);
% TableResults045 = [TableResults045 TMatrix045]
% %TableResults045 = removevars(TableResults045,{'NetScores2'});
% 
% %%
% HolisA = TableResults1045{1:31,{'NetScores2'}}
% Matrix1045 = zeros(31,1);
% for iNumber=1:31
%     HolisAA = HolisA{iNumber,1};
%     HolisAAA = cell2mat(HolisAA);
%     Matrix1045(iNumber) = HolisAAA
% end
% TMatrix1045 = array2table(Matrix1045);
% TableResults1045 = [TableResults1045 TMatrix1045]
% %TableResults1045 = removevars(TableResults1045,{'NetScores2'});
% 
% %%
% HolisA = TableResults050{1:31,{'NetScores2'}}
% Matrix050 = zeros(31,1);
% for iNumber=1:31
%     HolisAA = HolisA{iNumber,1};
%     HolisAAA = cell2mat(HolisAA);
%     Matrix050(iNumber) = HolisAAA
% end
% TMatrix050 = array2table(Matrix050);
% TableResults050 = [TableResults050 TMatrix050]
% %TableResults050 = removevars(TableResults050,{'NetScores2'});
% 
% %%
% HolisA = TableResults1050{1:31,{'NetScores2'}}
% Matrix1050 = zeros(31,1);
% for iNumber=1:31
%     HolisAA = HolisA{iNumber,1};
%     HolisAAA = cell2mat(HolisAA);
%     Matrix1050(iNumber) = HolisAAA
% end
% TMatrix1050 = array2table(Matrix1050);
% TableResults1050 = [TableResults1050 TMatrix1050]
% %TableResults1050 = removevars(TableResults1050,{'NetScores2'});
% 
% %%
% HolisA = TableResults080{1:31,{'NetScores2'}}
% Matrix080 = zeros(31,1);
% for iNumber=1:31
%     HolisAA = HolisA{iNumber,1};
%     HolisAAA = cell2mat(HolisAA);
%     Matrix080(iNumber) = HolisAAA
% end
% TMatrix080 = array2table(Matrix080);
% TableResults080 = [TableResults080 TMatrix080]
% %TableResults080 = removevars(TableResults080,{'NetScores2'});
% %%
% HolisA = TableResults1080{1:31,{'NetScores2'}}
% Matrix1080 = zeros(31,1);
% for iNumber=1:31
%     HolisAA = HolisA{iNumber,1};
%     HolisAAA = cell2mat(HolisAA);
%     Matrix1080(iNumber) = HolisAAA
% end
% TMatrix1080 = array2table(Matrix1080);
% TableResults1080 = [TableResults1080 TMatrix1080]
% %TableResults1080 = removevars(TableResults1080,{'NetScores2'});
% %%
% HolisA = TableResults100{1:31,{'NetScores2'}}
% Matrix100 = zeros(31,1);
% for iNumber=1:31
%     HolisAA = HolisA{iNumber,1};
%     HolisAAA = cell2mat(HolisAA);
%     Matrix100(iNumber) = HolisAAA
% end
% TMatrix100 = array2table(Matrix100);
% TableResults100 = [TableResults100 TMatrix100]
% %TableResults100 = removevars(TableResults100,{'NetScores2'});
% %%
% HolisA = TableResults1100{1:31,{'NetScores2'}}
% Matrix1100 = zeros(31,1);
% for iNumber=1:31
%     HolisAA = HolisA{iNumber,1};
%     HolisAAA = cell2mat(HolisAA);
%     Matrix1100(iNumber) = HolisAAA
% end
% TMatrix1100 = array2table(Matrix1100);
% TableResults1100 = [TableResults1100 TMatrix1100]
% %TableResults1100 = removevars(TableResults1100,{'NetScores2'});
% 
% 
% 
% clear HolisA HolisAA HolisAAA TableResults2040