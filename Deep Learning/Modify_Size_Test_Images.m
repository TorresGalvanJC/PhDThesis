%% Modify size of the Images

% In each section we've to rename the pathName...
% and the Files with the name of each folder
%% Atlético de San Luis

pathName = 'C:\Users\ADMIN\Documents\Universidad\Doctorado\Pruebas Deep Learning\Reducidas Imágenes Deporte Para Matlab\Atlético San Luis'; % Folder de imagenes
codeFolder = 'C:\Users\ADMIN\Dropbox\Doctorado\MATLAB Codigos\Práctica';
cd(codeFolder);

% Resize all images of the folder

dirStruct = dir(fullfile(pathName, '*.jpg'));
ASLnFiles = numel(dirStruct);



for iFiles = 1:ASLnFiles,
    currentFile = dirStruct(iFiles).name;
    Photo = imread(fullfile(pathName, currentFile));
    Resized = imresize(Photo, [240 320]);
    imwrite(Resized,fullfile(pathName, currentFile))
end

%% Pumas

pathName = 'C:\Users\ADMIN\Documents\Universidad\Doctorado\Pruebas Deep Learning\Reducidas Imágenes Deporte Para Matlab\Pumas'; % Folder de imagenes
codeFolder = 'C:\Users\ADMIN\Dropbox\Doctorado\MATLAB Codigos\Práctica';
cd(codeFolder);

% Resize all images of the folder

dirStruct = dir(fullfile(pathName, '*.jpg'));
PUMASnFiles = numel(dirStruct);

for iFiles = 1:PUMASnFiles,
    currentFile = dirStruct(iFiles).name;
    Photo = imread(fullfile(pathName, currentFile));
    Resized = imresize(Photo, [240 320]);
    imwrite(Resized,fullfile(pathName, currentFile))
end

%% Sultanes
% Folder de imagenes
pathName = 'C:\Users\ADMIN\Documents\Universidad\Doctorado\Pruebas Deep Learning\Reducidas Imágenes Deporte Para Matlab\Sultanes';
codeFolder = 'C:\Users\ADMIN\Dropbox\Doctorado\MATLAB Codigos\Práctica';
cd(codeFolder);

% Resize all images of the folder

dirStruct = dir(fullfile(pathName, '*.jpg'));
MTYnFiles = numel(dirStruct);

for iFiles = 1:MTYnFiles,
    currentFile = dirStruct(iFiles).name;
    Photo = imread(fullfile(pathName, currentFile));
    Resized = imresize(Photo, [240 320]);
    imwrite(Resized,fullfile(pathName, currentFile))
end

%% Toros 2016
% Folder de imagenes
pathName = 'C:\Users\ADMIN\Documents\Universidad\Doctorado\Pruebas Deep Learning\Reducidas Imágenes Deporte Para Matlab\Toros 2016';
codeFolder = 'C:\Users\ADMIN\Dropbox\Doctorado\MATLAB Codigos\Práctica';
cd(codeFolder);

% Resize all images of the folder

dirStruct = dir(fullfile(pathName, '*.jpg'));
ToroNFiles = numel(dirStruct);

for iFiles = 1:ToroNFiles,
    currentFile = dirStruct(iFiles).name;
    Photo = imread(fullfile(pathName, currentFile));
    Resized = imresize(Photo, [240 320]);
    imwrite(Resized,fullfile(pathName, currentFile))
end

%% Yaquis
% Folder de imagenes
pathName = 'C:\Users\ADMIN\Documents\Universidad\Doctorado\Pruebas Deep Learning\Reducidas Imágenes Deporte Para Matlab\Yaquis';
codeFolder = 'C:\Users\ADMIN\Dropbox\Doctorado\MATLAB Codigos\Práctica';
cd(codeFolder);

% Resize all images of the folder

dirStruct = dir(fullfile(pathName, '*.jpg'));
YaquisnFiles = numel(dirStruct);

for iFiles = 1:YaquisnFiles,
    currentFile = dirStruct(iFiles).name;
    Photo = imread(fullfile(pathName, currentFile));
    Resized = imresize(Photo, [240 320]);
    imwrite(Resized,fullfile(pathName, currentFile))
end