
codeFolder = 'C:\Users\ADMIN\Dropbox\Doctorado\MATLAB Codigos\Práctica\Prueba Deep Learning';
cd(codeFolder);

% In each section we've to rename the pathName...
% and the Files with the name of each folder
%% Futbol

pathName = 'C:\Users\ADMIN\Documents\Universidad\Doctorado\Pruebas Deep Learning\Reducidas\Futbol'; % Folder de imagenes

% Resize all images of the folder

dirStruct = dir(fullfile(pathName, '*.jpg'));
FUTnFiles = numel(dirStruct);

for iFiles = 1:FUTnFiles,
    currentFile = dirStruct(iFiles).name;
    Photo = imread(fullfile(pathName, currentFile));
    Resized = imresize(Photo, [240 320]);
    imwrite(Resized,fullfile(pathName, currentFile))
end

%% Béisbol

pathName = 'C:\Users\ADMIN\Documents\Universidad\Doctorado\Pruebas Deep Learning\Reducidas\Béisbol'; % Folder de imagenes

% Resize all images of the folder

dirStruct = dir(fullfile(pathName, '*.jpg'));
BEISnFiles = numel(dirStruct);

for iFiles = 1:BEISnFiles,
    currentFile = dirStruct(iFiles).name;
    Photo = imread(fullfile(pathName, currentFile));
    Resized = imresize(Photo, [240 320]);
    imwrite(Resized,fullfile(pathName, currentFile))
end

%% Básquetbol
% Folder de imagenes
pathName = 'C:\Users\ADMIN\Documents\Universidad\Doctorado\Pruebas Deep Learning\Reducidas\Básquetbol';

% Resize all images of the folder

dirStruct = dir(fullfile(pathName, '*.jpg'));
BASQnFiles = numel(dirStruct);

for iFiles = 1:BASQnFiles,
    currentFile = dirStruct(iFiles).name;
    Photo = imread(fullfile(pathName, currentFile));
    Resized = imresize(Photo, [240 320]);
    imwrite(Resized,fullfile(pathName, currentFile))
end

%% Tauromaquia
% Folder de imagenes
pathName = 'C:\Users\ADMIN\Documents\Universidad\Doctorado\Pruebas Deep Learning\Reducidas\Tauromaquia';

% Resize all images of the folder

dirStruct = dir(fullfile(pathName, '*.jpg'));
ToroNFiles = numel(dirStruct);

for iFiles = 1:ToroNFiles,
    currentFile = dirStruct(iFiles).name;
    Photo = imread(fullfile(pathName, currentFile));
    Resized = imresize(Photo, [240 320]);
    imwrite(Resized,fullfile(pathName, currentFile))
end
