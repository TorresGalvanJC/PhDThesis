% Generate the variations in the images

clc; clear
ImgFolder ='C:\Users\ADMIN\Documents\Universidad\Doctorado\Termogramas\Termogramas TIF\Try';
ImgResults = 'C:\Users\ADMIN\Documents\Universidad\Doctorado\Termogramas\Termogramas TIF\Catch';
cd(ImgFolder)

%%

dirStruct = dir(fullfile(ImgFolder, '*.tif'));

% Number of files with that characteristics
BaseImages = numel(dirStruct);
nCopies = 3;

% Notes for the images:
% 1 = Rotation
% 2 = X Reflection
% 3 = Y Reflection
% 4 = Shear in X
% 5 = Shear in Y
% 6 = Translate X
% 7 = Translate Y
% Each number is the combination of the modifications of the images

for nImages = 1:BaseImages
    currentFile = dirStruct(nImages).name;
    currentFile2 = dirStruct(nImages).name(1:end-4);
    
    TheImg = imread(currentFile);
    imshow(TheImg);
    
    for kCopies = 1:nCopies
        %% Rotation
        
        % Random rotation between two angles
        
        Rot = randi([0, 359], 1, 1);
        if Rot ~= 0
            TheImg1 = imrotate(TheImg, Rot); % Generate rotations
            imwrite(TheImg1, fullfile(ImgResults, [currentFile2, '_1_', num2str(kCopies),'.tif']));
        end
        % WORKS OK
    end
    
    
    %% X Reflection (Mirror)
    % A = randi([0, 1], 1, 1);
    % if A == 1
    TheImg2 = flipdim(TheImg,2);
    % imshow(TheImg2);
    imwrite(TheImg2,fullfile(ImgResults, [currentFile2, '_2.tif']));
    % end
    
    % WORKS OK
    
    %% Y Reflection (Mirror)
    
    % A = randi([0, 1], 1, 1);
    % if A == 1
    TheImg3 = flipud(TheImg);
    % imshow(TheImg3);
    imwrite(TheImg3,fullfile(ImgResults, [currentFile2, '_3.tif']));
    %end
    
    % WORKS OK
    
%     %% Shear in X
%     
%     for kCopies = 1:nCopies
%         A = (randi(201) - 151) ./ 100;
%         
%         if A ~= 0
%             T = maketform('affine', [1 0 0; A 1 0; 0 0 1] );
%             
%             black = [0 0 0]';
%             R = makeresampler({'cubic','nearest'},'fill');
%             TheImg4 = imtransform(TheImg,T,R,'FillValues',black);
%             imwrite(TheImg4, fullfile(ImgResults, [currentFile2, '_4_', num2str(kCopies),'.tif']));
%             %     h2 = figure;
%             %     imshow(TheImg5);
%             %     title('Sheared Image');
%         end
%     end
%     % Works OK
%     %% Shear in Y
%     
%     for kCopies = 1:nCopies
%         A = (randi(201) - 151) ./ 100;
%         if A ~= 0
%             T = maketform('affine', [1 A 0; 0 1 0; 0 0 1] );
%             black = [0 0 0]';
%             R = makeresampler({'cubic','nearest'},'fill');
%             TheImg5 = imtransform(TheImg,T,R,'FillValues',black);
%             imwrite(TheImg5, fullfile(ImgResults, [currentFile2, '_5_', num2str(kCopies),'.tif']));
%             %     h2 = figure;
%             %     imshow(TheImg6);
%             %     title('Sheared Image');
%         end
%     end
%     
%     % Shear in Y Works OK
    
    %% Translate in X
    for kCopies = 1:nCopies
        A = randi([-150, 150], 1, 1);
        if A ~= 0
            TheImg6 = imtranslate(TheImg,[A, 0]);
            imwrite(TheImg6, fullfile(ImgResults, [currentFile2, '_6_', num2str(kCopies),'.tif']));
        end
    end
    % Translate in X Works OK
    
    %% Translate in Y
    for kCopies = 1:nCopies
        A = randi([-150, 150], 1, 1);
        if A ~= 0
            TheImg7 = imtranslate(TheImg,[0, A]);
            imwrite(TheImg7, fullfile(ImgResults, [currentFile2, '_7_', num2str(kCopies),'.tif']));
        end
    end
    %% Combination 67
    for kCopies = 1:nCopies
        A = randi([-150, 150], 1, 1);
        if A ~= 0
            TheImg67 = imtranslate(TheImg,[A, 0]);
            
            B = randi([-50, 50], 1, 1);
            if B ~= 0
                TheImg67 = imtranslate(TheImg67,[0, B]);
            end
            imwrite(TheImg67, fullfile(ImgResults, [currentFile2, '_67_', num2str(kCopies),'.tif']));
        end
    end
    
    %% Combination 56
    
    
%     for kCopies = 1:nCopies
%         A = (randi(201) - 151) ./ 100;
%         if A ~= 0
%             T = maketform('affine', [1 A 0; 0 1 0; 0 0 1] );
%             black = [0 0 0]';
%             R = makeresampler({'cubic','nearest'},'fill');
%             TheImg56 = imtransform(TheImg,T,R,'FillValues',black);
%             
%             %%
%             B = randi([-50, 50], 1, 1);
%             if B ~= 0
%                 TheImg56 = imtranslate(TheImg56,[B, 0]);
%             end
%             
%             %%
%             imwrite(TheImg56, fullfile(ImgResults, [currentFile2, '_56_', num2str(kCopies),'.tif']));
%             %     h2 = figure;
%             %     imshow(TheImg6);
%             %     title('Sheared Image');
%         end
%     end
%     %% Combination 567
%     for kCopies = 1:nCopies
%         C = (randi(201) - 151) ./ 100;
%         if C ~= 0
%             T = maketform('affine', [1 C 0; 0 1 0; 0 0 1] );
%             black = [0 0 0]';
%             R = makeresampler({'cubic','nearest'},'fill');
%             TheImg567 = imtransform(TheImg,T,R,'FillValues',black);
%             %
%             A = randi([-150, 150], 1, 1);
%             if A ~= 0
%                 TheImg567 = imtranslate(TheImg,[A, 0]);
%                 
%                 B = randi([-50, 50], 1, 1);
%                 if B ~= 0
%                     TheImg567 = imtranslate(TheImg567,[0, B]);
%                 end
%             end
%             %
%             imwrite(TheImg567, fullfile(ImgResults, [currentFile2, '_567_', num2str(kCopies),'.tif']));
%         end
%     end
%     %% Combination 45
%     
%     for kCopies = 1:nCopies
%         A = (randi(201) - 151) ./ 100;
%         
%         if A ~= 0
%             T = maketform('affine', [1 0 0; A 1 0; 0 0 1] );
%             
%             black = [0 0 0]';
%             R = makeresampler({'cubic','nearest'},'fill');
%             TheImg45 = imtransform(TheImg,T,R,'FillValues',black);
%             
%             %%
%             AA = (randi(201) - 151) ./ 100;
%             if AA ~= 0
%                 T = maketform('affine', [1 AA 0; 0 1 0; 0 0 1] );
%                 black = [0 0 0]';
%                 RA = makeresampler({'cubic','nearest'},'fill');
%                 TheImg45 = imtransform(TheImg45,T,RA,'FillValues',black);
%             end
%             %%
%             imwrite(TheImg45, fullfile(ImgResults, [currentFile2, '_45_', num2str(kCopies),'.tif']));
%             
%         end
%     end
    % Works OK
    
%     %% Combination 456
%     
%     for kCopies = 1:nCopies
%         A = (randi(201) - 151) ./ 100;
%         
%         if A ~= 0
%             T = maketform('affine', [1 0 0; A 1 0; 0 0 1] );
%             
%             black = [0 0 0]';
%             R = makeresampler({'cubic','nearest'},'fill');
%             TheImg456 = imtransform(TheImg,T,R,'FillValues',black);
%             
%             AA = (randi(201) - 151) ./ 100;
%             if AA ~= 0
%                 T = maketform('affine', [1 AA 0; 0 1 0; 0 0 1] );
%                 black = [0 0 0]';
%                 RA = makeresampler({'cubic','nearest'},'fill');
%                 TheImg456 = imtransform(TheImg456,T,RA,'FillValues',black);
%                 
%             end
%             
%             AAA = randi([-50, 50], 1, 1);
%             if AAA ~= 0
%                 TheImg456 = imtranslate(TheImg456,[AAA, 0]);
%                 
%             end
%             imwrite(TheImg456, fullfile(ImgResults, [currentFile2, '_456_', num2str(kCopies),'.tif']));
%             
%         end
%     end
%     % Works OK
    
%     %% Combination 4567
%     
%     for kCopies = 1:nCopies
%         A = (randi(201) - 151) ./ 100;
%         
%         if A ~= 0
%             T = maketform('affine', [1 0 0; A 1 0; 0 0 1] );
%             
%             black = [0 0 0]';
%             R = makeresampler({'cubic','nearest'},'fill');
%             TheImg4567 = imtransform(TheImg,T,R,'FillValues',black);
%             
%             
%             AA = (randi(201) - 151) ./ 100;
%             if AA ~= 0
%                 TA = maketform('affine', [1 AA 0; 0 1 0; 0 0 1] );
%                 black = [0 0 0]';
%                 RA = makeresampler({'cubic','nearest'},'fill');
%                 TheImg4567 = imtransform(TheImg4567,TA,RA,'FillValues',black);
%                 
%             end
%             
%             AAA = randi([-50, 50], 1, 1);
%             if AAA ~= 0
%                 TheImg4567 = imtranslate(TheImg4567,[AAA, 0]);
%             end
%             %%
%             AAAA = randi([-50, 50], 1, 1);
%             if AAAA ~= 0
%                 TheImg4567 = imtranslate(TheImg4567,[0, AAAA]);
%                 
%             end
%             %%
%             imwrite(TheImg4567, fullfile(ImgResults, [currentFile2, '_4567_', num2str(kCopies),'.tif']));
%             
%         end
%     end
%     % Works OK
%     
    %% Combination 34567
    
    for kCopies = 1:nCopies
        A = (randi(201) - 151) ./ 100;
        
%         if A ~= 0
%             T = maketform('affine', [1 0 0; A 1 0; 0 0 1] );
%             
%             black = [0 0 0]';
%             R = makeresampler({'cubic','nearest'},'fill');
%             TheImg34567 = imtransform(TheImg,T,R,'FillValues',black);
%             
            
%             AA = (randi(201) - 151) ./ 100;
%             if AA ~= 0
%                 TA = maketform('affine', [1 AA 0; 0 1 0; 0 0 1] );
%                 black = [0 0 0]';
%                 RA = makeresampler({'cubic','nearest'},'fill');
%                 TheImg34567 = imtransform(TheImg34567,TA,RA,'FillValues',black);
%                 
%             end
            
            AAA = randi([-150, 150], 1, 1);
            if AAA ~= 0
                TheImg34567 = imtranslate(TheImg,[AAA, 0]);
            end
            %%
            AAAA = randi([-150, 150], 1, 1);
            if AAAA ~= 0
                TheImg34567 = imtranslate(TheImg34567,[0, AAAA]);
                
            end
            TheImg34567 = flipud(TheImg34567);
            %%
            imwrite(TheImg34567, fullfile(ImgResults, [currentFile2, '_34567_', num2str(kCopies),'.tif']));
            
%         end
    end
    % Works OK
    %% Combination 234567
    
    for kCopies = 1:nCopies
%         A = (randi(201) - 151) ./ 100;
        
%         if A ~= 0
%             T = maketform('affine', [1 0 0; A 1 0; 0 0 1] );
%             
%             black = [0 0 0]';
%             R = makeresampler({'cubic','nearest'},'fill');
%             TheImg234567 = imtransform(TheImg,T,R,'FillValues',black);
%             
%             
%             AA = (randi(201) - 151) ./ 100;
%             if AA ~= 0
%                 TA = maketform('affine', [1 AA 0; 0 1 0; 0 0 1] );
%                 black = [0 0 0]';
%                 RA = makeresampler({'cubic','nearest'},'fill');
%                 TheImg234567 = imtransform(TheImg234567,TA,RA,'FillValues',black);
%                 
%             end
            
            AAA = randi([-150, 150], 1, 1);
            if AAA ~= 0
                TheImg234567 = imtranslate(TheImg,[AAA, 0]);
            end
            %%
            AAAA = randi([-150, 150], 1, 1);
            if AAAA ~= 0
                TheImg234567 = imtranslate(TheImg234567,[0, AAAA]);
                
            end
            TheImg234567 = flipud(TheImg234567);
            TheImg234567 = flipdim(TheImg234567,2);
            %%
            imwrite(TheImg234567, fullfile(ImgResults, [currentFile2, '_234567_', num2str(kCopies),'.tif']));
            
%         end
    end
    %% Combination 1234567
    
    for kCopies = 1:nCopies
        A = (randi(201) - 151) ./ 100;
        
%         if A ~= 0
%             T = maketform('affine', [1 0 0; A 1 0; 0 0 1] );
%             
%             black = [0 0 0]';
%             R = makeresampler({'cubic','nearest'},'fill');
%             TheImg1234567 = imtransform(TheImg,T,R,'FillValues',black);
%             
%             
%             AA = (randi(201) - 151) ./ 100;
%             if AA ~= 0
%                 TA = maketform('affine', [1 AA 0; 0 1 0; 0 0 1] );
%                 black = [0 0 0]';
%                 RA = makeresampler({'cubic','nearest'},'fill');
%                 TheImg1234567 = imtransform(TheImg1234567,TA,RA,'FillValues',black);
%                 
%             end
            
            AAA = randi([-150, 150], 1, 1);
            if AAA ~= 0
                TheImg1234567 = imtranslate(TheImg,[AAA, 0]);
            end
            %%
            AAAA = randi([-150, 150], 1, 1);
            if AAAA ~= 0
                TheImg1234567 = imtranslate(TheImg1234567,[0, AAAA]);
                
            end
            TheImg1234567 = flipud(TheImg1234567);
            TheImg1234567 = flipdim(TheImg1234567,2);
            Rot = randi([0, 359], 1, 1);
            if Rot ~= 0
                TheImg1234567 = imrotate(TheImg1234567, Rot); % Generate rotations
                
            end
            %%
            imwrite(TheImg1234567, fullfile(ImgResults, [currentFile2, '_1234567_', num2str(kCopies),'.tif']));
            
%         end
        
        % WORKS OK
    end
    % Works OK
    
    %% Combination 123456
    
    for kCopies = 1:nCopies
%         A = (randi(201) - 151) ./ 100;
%         
%         if A ~= 0
%             T = maketform('affine', [1 0 0; A 1 0; 0 0 1] );
%             
%             black = [0 0 0]';
%             R = makeresampler({'cubic','nearest'},'fill');
%             TheImg123456 = imtransform(TheImg,T,R,'FillValues',black);
%             
%             
%             AA = (randi(201) - 151) ./ 100;
%             if AA ~= 0
%                 TA = maketform('affine', [1 AA 0; 0 1 0; 0 0 1] );
%                 black = [0 0 0]';
%                 RA = makeresampler({'cubic','nearest'},'fill');
%                 TheImg123456 = imtransform(TheImg123456,TA,RA,'FillValues',black);
%                 
%             end
%             
            AAA = randi([-150, 150], 1, 1);
            if AAA ~= 0
                TheImg123456 = imtranslate(TheImg,[AAA, 0]);
            end
            %%
            
            TheImg123456 = flipud(TheImg123456);
            TheImg123456 = flipdim(TheImg123456,2);
            Rot = randi([0, 359], 1, 1);
            if Rot ~= 0
                TheImg123456 = imrotate(TheImg123456, Rot); % Generate rotations
                
            end
            %%
            imwrite(TheImg123456, fullfile(ImgResults, [currentFile2, '_123456_', num2str(kCopies),'.tif']));
            
%         end
        
        % WORKS OK
    end
    % Works OK
%     %% Combination 12345
%     
%     for kCopies = 1:nCopies
%         A = (randi(201) - 151) ./ 100;
%         
%         if A ~= 0
%             T = maketform('affine', [1 0 0; A 1 0; 0 0 1] );
%             
%             black = [0 0 0]';
%             R = makeresampler({'cubic','nearest'},'fill');
%             TheImg12345 = imtransform(TheImg,T,R,'FillValues',black);
%             
%             
%             AA = (randi(201) - 151) ./ 100;
%             if AA ~= 0
%                 TA = maketform('affine', [1 AA 0; 0 1 0; 0 0 1] );
%                 black = [0 0 0]';
%                 RA = makeresampler({'cubic','nearest'},'fill');
%                 TheImg12345 = imtransform(TheImg12345,TA,RA,'FillValues',black);
%                 
%             end
%             
%             
%             %%
%             
%             TheImg12345 = flipud(TheImg12345);
%             
%             TheImg12345 = flipdim(TheImg12345,2);
%             
%             Rot = randi([0, 359], 1, 1);
%             if Rot ~= 0
%                 TheImg12345 = imrotate(TheImg12345, Rot); % Generate rotations
%                 
%             end
%             %%
%             imwrite(TheImg12345, fullfile(ImgResults, [currentFile2, '_12345_', num2str(kCopies),'.tif']));
%             
%         end
%         
%         % WORKS OK
%     end
%     % Works OK
%     %% Combination 1234
%     
%     for kCopies = 1:nCopies
%         A = (randi(201) - 151) ./ 100;
%         
%         if A ~= 0
%             T = maketform('affine', [1 0 0; A 1 0; 0 0 1] );
%             
%             black = [0 0 0]';
%             R = makeresampler({'cubic','nearest'},'fill');
%             TheImg1234 = imtransform(TheImg,T,R,'FillValues',black);
%             
%             %%
%             
%             TheImg1234 = flipud(TheImg1234);
%             
%             TheImg1234 = flipdim(TheImg1234,2);
%             
%             Rot = randi([0, 359], 1, 1);
%             if Rot ~= 0
%                 TheImg1234 = imrotate(TheImg1234, Rot); % Generate rotations
%                 
%             end
%             %%
%             imwrite(TheImg1234, fullfile(ImgResults, [currentFile2, '_1234_', num2str(kCopies),'.tif']));
%             
%         end
%         
%         % WORKS OK
%     end
%     % Works OK
    %% Combination 123
    
    for kCopies = 1:nCopies
        
        TheImg123 = flipud(TheImg);
        
        TheImg123 = flipdim(TheImg123,2);
        
        Rot = randi([0, 359], 1, 1);
        if Rot ~= 0
            TheImg123 = imrotate(TheImg123, Rot); % Generate rotations
            
        end
        
        imwrite(TheImg123, fullfile(ImgResults, [currentFile2, '_123_', num2str(kCopies),'.tif']));
        
    end
    
    % WORKS OK
    %% Combination 12
    
    for kCopies = 1:nCopies
        
        
        TheImg12 = flipdim(TheImg,2);
        
        Rot = randi([0, 359], 1, 1);
        if Rot ~= 0
            TheImg12 = imrotate(TheImg12, Rot); % Generate rotations
        end
        imwrite(TheImg12, fullfile(ImgResults, [currentFile2, '_12_', num2str(kCopies),'.tif']));
        
    end
    
    % WORKS OK
    
    %% Combination 23456
    
    for kCopies = 1:nCopies
%         A = (randi(201) - 151) ./ 100;
%         
%         if A ~= 0
%             T = maketform('affine', [1 0 0; A 1 0; 0 0 1] );
%             
%             black = [0 0 0]';
%             R = makeresampler({'cubic','nearest'},'fill');
%             TheImg23456 = imtransform(TheImg,T,R,'FillValues',black);
%             
%             
%             AA = (randi(201) - 151) ./ 100;
%             if AA ~= 0
%                 TA = maketform('affine', [1 AA 0; 0 1 0; 0 0 1] );
%                 black = [0 0 0]';
%                 RA = makeresampler({'cubic','nearest'},'fill');
%                 TheImg23456 = imtransform(TheImg23456,TA,RA,'FillValues',black);
%                 
%             end
            
            AAA = randi([-150, 150], 1, 1);
            if AAA ~= 0
                TheImg23456 = imtranslate(TheImg,[AAA, 0]);
            end
            %%
            
            TheImg23456 = flipud(TheImg23456);
            TheImg23456 = flipdim(TheImg23456,2);
            
            %%
            imwrite(TheImg23456, fullfile(ImgResults, [currentFile2, '_23456_', num2str(kCopies),'.tif']));
            
%         end
        
        % WORKS OK
    end
    % Works OK
    
%     %% Combination 2345
%     
%     for kCopies = 1:nCopies
%         A = (randi(201) - 151) ./ 100;
%         
%         if A ~= 0
%             T = maketform('affine', [1 0 0; A 1 0; 0 0 1] );
%             
%             black = [0 0 0]';
%             R = makeresampler({'cubic','nearest'},'fill');
%             TheImg2345 = imtransform(TheImg,T,R,'FillValues',black);
%             
%             
%             AA = (randi(201) - 151) ./ 100;
%             if AA ~= 0
%                 TA = maketform('affine', [1 AA 0; 0 1 0; 0 0 1] );
%                 black = [0 0 0]';
%                 RA = makeresampler({'cubic','nearest'},'fill');
%                 TheImg2345 = imtransform(TheImg2345,TA,RA,'FillValues',black);
%                 
%             end
%             
%             TheImg2345 = flipud(TheImg2345);
%             TheImg2345 = flipdim(TheImg2345,2);
%             
%             %%
%             imwrite(TheImg2345, fullfile(ImgResults, [currentFile2, '_2345_', num2str(kCopies),'.tif']));
%             
%         end
%         
%         % WORKS OK
%     end
%     % Works OK
%     %% Combination 234
%     
%     for kCopies = 1:nCopies
%         A = (randi(201) - 151) ./ 100;
%         
%         if A ~= 0
%             T = maketform('affine', [1 0 0; A 1 0; 0 0 1] );
%             
%             black = [0 0 0]';
%             R = makeresampler({'cubic','nearest'},'fill');
%             TheImg234 = imtransform(TheImg,T,R,'FillValues',black);
%             
%             
%             TheImg234 = flipud(TheImg234);
%             TheImg234 = flipdim(TheImg234,2);
%             
%             %%
%             imwrite(TheImg234, fullfile(ImgResults, [currentFile2, '_234_', num2str(kCopies),'.tif']));
%             
%         end
%         
%         % WORKS OK
%     end
%     % Works OK
    %% Combination 23
    
    for kCopies = 1:nCopies
        
        
        TheImg23 = flipud(TheImg);
        TheImg23 = flipdim(TheImg23,2);
        
        %%
        imwrite(TheImg23, fullfile(ImgResults, [currentFile2, '_23_', num2str(kCopies),'.tif']));
        
    end
    %% Combination 3456
    
    for kCopies = 1:nCopies
%         A = (randi(201) - 151) ./ 100;
%         
%         if A ~= 0
%             T = maketform('affine', [1 0 0; A 1 0; 0 0 1] );
%             
%             black = [0 0 0]';
%             R = makeresampler({'cubic','nearest'},'fill');
%             TheImg3456 = imtransform(TheImg,T,R,'FillValues',black);
%             
%             
%             AA = (randi(201) - 151) ./ 100;
%             if AA ~= 0
%                 TA = maketform('affine', [1 AA 0; 0 1 0; 0 0 1] );
%                 black = [0 0 0]';
%                 RA = makeresampler({'cubic','nearest'},'fill');
%                 TheImg3456 = imtransform(TheImg3456,TA,RA,'FillValues',black);
%                 
%             end
            
            AAA = randi([-150, 150], 1, 1);
            if AAA ~= 0
                TheImg3456 = imtranslate(TheImg,[AAA, 0]);
            end
            %%
            
            TheImg3456 = flipud(TheImg3456);
            
            imwrite(TheImg3456, fullfile(ImgResults, [currentFile2, '_3456_', num2str(kCopies),'.tif']));
            
%         end
        
        % WORKS OK
    end
    % Works OK
    
%     %% Combination 345
%     
%     for kCopies = 1:nCopies
%         A = (randi(201) - 151) ./ 100;
%         
%         if A ~= 0
%             T = maketform('affine', [1 0 0; A 1 0; 0 0 1] );
%             
%             black = [0 0 0]';
%             R = makeresampler({'cubic','nearest'},'fill');
%             TheImg345 = imtransform(TheImg,T,R,'FillValues',black);
%             
%             
%             AA = (randi(201) - 151) ./ 100;
%             if AA ~= 0
%                 TA = maketform('affine', [1 AA 0; 0 1 0; 0 0 1] );
%                 black = [0 0 0]';
%                 RA = makeresampler({'cubic','nearest'},'fill');
%                 TheImg345 = imtransform(TheImg345,TA,RA,'FillValues',black);
%                 
%             end
%             
%             
%             TheImg345 = flipud(TheImg345);
%             
%             imwrite(TheImg345, fullfile(ImgResults, [currentFile2, '_345_', num2str(kCopies),'.tif']));
%             
%         end
%         
%         % WORKS OK
%     end
%     % Works OK
%     
%     %% Combination 34
%     
%     for kCopies = 1:nCopies
%         A = (randi(201) - 151) ./ 100;
%         
%         if A ~= 0
%             T = maketform('affine', [1 0 0; A 1 0; 0 0 1] );
%             
%             black = [0 0 0]';
%             R = makeresampler({'cubic','nearest'},'fill');
%             TheImg34 = imtransform(TheImg,T,R,'FillValues',black);
%             
%             TheImg34 = flipud(TheImg34);
%             
%             imwrite(TheImg34, fullfile(ImgResults, [currentFile2, '_34_', num2str(kCopies),'.tif']));
%             
%         end
%         
%         % WORKS OK
%     end
%     % Works OK
%     
    
end
% Works OK
