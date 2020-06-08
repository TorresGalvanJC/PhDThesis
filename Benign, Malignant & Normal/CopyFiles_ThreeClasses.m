%% Benign 500 Times

% All the images in a folder can be copied...
% a specific number of times with different name

ImagesFolder = 'K:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\Termogramas\Tumors\Training\Benign';

% For create a new Folder
%mkdir 'C:\Users\ADMIN\Documents\Universidad\Doctorado\Pruebas Deep Learning\Reducidas' 'Hola Imagenes'

% Name of the new folder
HiImages = 'K:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\Termogramas\Tumors\Data Augmented\500 Times\Benign';

% Specify the number of Copues of the files
nCopies = 500;

cd(ImagesFolder)
% Only take in count all the images that are in the folder...
% with the .jpg extention
dirStruct = dir(fullfile(ImagesFolder, '*.tif'));

% Number of files with that characteristics
BaseImages = numel(dirStruct);

for nImages = 1:BaseImages
    currentFile = dirStruct(nImages).name;
    currentFile2 = dirStruct(nImages).name(1:end-4);
     for kCopies = 1:nCopies
         cd(ImagesFolder)
         copyfile (currentFile, HiImages)
         cd(HiImages)
         CopyNumber = ['_Copy(', num2str(kCopies),').tif'];
         nameFile = [currentFile2 CopyNumber];
         movefile(currentFile, nameFile)      
         % Update progress bar
         %ioi_text_waitbar(kCopies/nCopies, sprintf('Coping image %d from %d', kCopies, nCopies));
     end 
     % Update progress bar
         %ioi_text_waitbar(nImages/BaseImages, sprintf('Coping image %d from %d', nImages, BaseImages));
end

%ioi_text_waitbar('Clear');

% END OF FUNCTION

%% Malignant 500 Times

% All the images in a folder can be copied...
% a specific number of times with different name

ImagesFolder = 'K:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\Termogramas\Tumors\Training\Malign';

% For create a new Folder
%mkdir 'C:\Users\ADMIN\Documents\Universidad\Doctorado\Pruebas Deep Learning\Reducidas' 'Hola Imagenes'

% Name of the new folder
HiImages = 'K:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\Termogramas\Tumors\Data Augmented\500 Times\Malign';

% Specify the number of Copues of the files
nCopies = 500;

cd(ImagesFolder)
% Only take in count all the images that are in the folder...
% with the .jpg extention
dirStruct = dir(fullfile(ImagesFolder, '*.tif'));

% Number of files with that characteristics
BaseImages = numel(dirStruct);

for nImages = 1:BaseImages
    currentFile = dirStruct(nImages).name;
    currentFile2 = dirStruct(nImages).name(1:end-4);
     for kCopies = 1:nCopies
         cd(ImagesFolder)
         copyfile (currentFile, HiImages)
         cd(HiImages)
         CopyNumber = ['_Copy(', num2str(kCopies),').tif'];
         nameFile = [currentFile2 CopyNumber];
         movefile(currentFile, nameFile)      
         % Update progress bar
         %ioi_text_waitbar(kCopies/nCopies, sprintf('Coping image %d from %d', kCopies, nCopies));
     end 
     % Update progress bar
         %ioi_text_waitbar(nImages/BaseImages, sprintf('Coping image %d from %d', nImages, BaseImages));
end

%ioi_text_waitbar('Clear');

% END OF FUNCTION

%% Normal 500 Times

% All the images in a folder can be copied...
% a specific number of times with different name

ImagesFolder = 'K:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\Termogramas\Tumors\Training\Normal';

% For create a new Folder
%mkdir 'C:\Users\ADMIN\Documents\Universidad\Doctorado\Pruebas Deep Learning\Reducidas' 'Hola Imagenes'

% Name of the new folder
HiImages = 'K:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\Termogramas\Tumors\Data Augmented\500 Times\Normal';

% Specify the number of Copues of the files
nCopies = 500;

cd(ImagesFolder)
% Only take in count all the images that are in the folder...
% with the .jpg extention
dirStruct = dir(fullfile(ImagesFolder, '*.tif'));

% Number of files with that characteristics
BaseImages = numel(dirStruct);

for nImages = 1:BaseImages
    currentFile = dirStruct(nImages).name;
    currentFile2 = dirStruct(nImages).name(1:end-4);
     for kCopies = 1:nCopies
         cd(ImagesFolder)
         copyfile (currentFile, HiImages)
         cd(HiImages)
         CopyNumber = ['_Copy(', num2str(kCopies),').tif'];
         nameFile = [currentFile2 CopyNumber];
         movefile(currentFile, nameFile)      
         % Update progress bar
         %ioi_text_waitbar(kCopies/nCopies, sprintf('Coping image %d from %d', kCopies, nCopies));
     end 
     % Update progress bar
         %ioi_text_waitbar(nImages/BaseImages, sprintf('Coping image %d from %d', nImages, BaseImages));
end

%ioi_text_waitbar('Clear');

% END OF FUNCTION

%% Benign 500 Times

% All the images in a folder can be copied...
% a specific number of times with different name

ImagesFolder = 'K:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\Termogramas\Tumors\Training\Benign';

% For create a new Folder
%mkdir 'C:\Users\ADMIN\Documents\Universidad\Doctorado\Pruebas Deep Learning\Reducidas' 'Hola Imagenes'

% Name of the new folder
HiImages = 'K:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\Termogramas\Tumors\Data Augmented\Balanced Class\Benign';

% Specify the number of Copues of the files
nCopies = 216;

cd(ImagesFolder)
% Only take in count all the images that are in the folder...
% with the .jpg extention
dirStruct = dir(fullfile(ImagesFolder, '*.tif'));

% Number of files with that characteristics
BaseImages = numel(dirStruct);

for nImages = 1:BaseImages
    currentFile = dirStruct(nImages).name;
    currentFile2 = dirStruct(nImages).name(1:end-4);
     for kCopies = 1:nCopies
         cd(ImagesFolder)
         copyfile (currentFile, HiImages)
         cd(HiImages)
         CopyNumber = ['_Copy(', num2str(kCopies),').tif'];
         nameFile = [currentFile2 CopyNumber];
         movefile(currentFile, nameFile)      
         % Update progress bar
         %ioi_text_waitbar(kCopies/nCopies, sprintf('Coping image %d from %d', kCopies, nCopies));
     end 
     % Update progress bar
         %ioi_text_waitbar(nImages/BaseImages, sprintf('Coping image %d from %d', nImages, BaseImages));
end

%ioi_text_waitbar('Clear');

% END OF FUNCTION

%% Malignant 500 Times

% All the images in a folder can be copied...
% a specific number of times with different name

ImagesFolder = 'K:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\Termogramas\Tumors\Training\Malign';

% For create a new Folder
%mkdir 'C:\Users\ADMIN\Documents\Universidad\Doctorado\Pruebas Deep Learning\Reducidas' 'Hola Imagenes'

% Name of the new folder
HiImages = 'K:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\Termogramas\Tumors\Data Augmented\Balanced Class\Malign';

% Specify the number of Copues of the files
nCopies = 200;

cd(ImagesFolder)
% Only take in count all the images that are in the folder...
% with the .jpg extention
dirStruct = dir(fullfile(ImagesFolder, '*.tif'));

% Number of files with that characteristics
BaseImages = numel(dirStruct);

for nImages = 1:BaseImages
    currentFile = dirStruct(nImages).name;
    currentFile2 = dirStruct(nImages).name(1:end-4);
     for kCopies = 1:nCopies
         cd(ImagesFolder)
         copyfile (currentFile, HiImages)
         cd(HiImages)
         CopyNumber = ['_Copy(', num2str(kCopies),').tif'];
         nameFile = [currentFile2 CopyNumber];
         movefile(currentFile, nameFile)      
         % Update progress bar
         %ioi_text_waitbar(kCopies/nCopies, sprintf('Coping image %d from %d', kCopies, nCopies));
     end 
     % Update progress bar
         %ioi_text_waitbar(nImages/BaseImages, sprintf('Coping image %d from %d', nImages, BaseImages));
end

%ioi_text_waitbar('Clear');

% END OF FUNCTION

%% Normal 500 Times

% All the images in a folder can be copied...
% a specific number of times with different name

ImagesFolder = 'K:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\Termogramas\Tumors\Training\Normal';

% For create a new Folder
%mkdir 'C:\Users\ADMIN\Documents\Universidad\Doctorado\Pruebas Deep Learning\Reducidas' 'Hola Imagenes'

% Name of the new folder
HiImages = 'K:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\Termogramas\Tumors\Data Augmented\Balanced Class\Normal';

% Specify the number of Copues of the files
nCopies = 675;

cd(ImagesFolder)
% Only take in count all the images that are in the folder...
% with the .jpg extention
dirStruct = dir(fullfile(ImagesFolder, '*.tif'));

% Number of files with that characteristics
BaseImages = numel(dirStruct);

for nImages = 1:BaseImages
    currentFile = dirStruct(nImages).name;
    currentFile2 = dirStruct(nImages).name(1:end-4);
     for kCopies = 1:nCopies
         cd(ImagesFolder)
         copyfile (currentFile, HiImages)
         cd(HiImages)
         CopyNumber = ['_Copy(', num2str(kCopies),').tif'];
         nameFile = [currentFile2 CopyNumber];
         movefile(currentFile, nameFile)      
         % Update progress bar
         %ioi_text_waitbar(kCopies/nCopies, sprintf('Coping image %d from %d', kCopies, nCopies));
     end 
     % Update progress bar
         %ioi_text_waitbar(nImages/BaseImages, sprintf('Coping image %d from %d', nImages, BaseImages));
end

%ioi_text_waitbar('Clear');

% END OF FUNCTION