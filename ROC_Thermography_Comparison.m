%% Balanced Class Data

clc; clear
load('E:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\Results\ResNet101\Iteraciones\Balanced\RN_BalCl_MinBat10_1239.mat',...
    'targets', 'scores');

targets_BalClass = targets;
scores_BalClass = scores;
% targets = repmat(targets, [1000, 1]);
% scores = repmat(scores, [1000, 1]);
% [X,Y,T,AUC] = perfcurve(targets,scores(:,1),0, 'NBoot',100,'XVals',0:0.02:1);
[X_BalClass, Y_BalClass, T_BalClass, AUC_BalClass] = perfcurve(targets_BalClass, scores_BalClass(:,1),0);

[c_BalClass, cm_BalClass, ind_BalClass, per_BalClass] = confusion(~targets_BalClass', scores_BalClass(:,1)');

Se_BalClass = cm_BalClass(2,2)/(cm_BalClass(2,2) + cm_BalClass(2,1));
Sp_BalClass = cm_BalClass(1,1)/(cm_BalClass(1,1) + cm_BalClass(1,2));

clearvars targets scores targets_BalClass scores_BalClass c_BalClass cm_BalClass ind_BalClass per_BalClass T_BalClass

%% Unbalanced Class Data

load('E:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\Results\ResNet101\Iteraciones\Unbalanced\RN_UnCl_MinBat10_448.mat',...
    'targets', 'scores');
targets_UnClass = targets;
scores_UnClass = scores;
% targets = repmat(targets, [1000, 1]);
% scores = repmat(scores, [1000, 1]);
% [X,Y,T,AUC] = perfcurve(targets,scores(:,1),0, 'NBoot',100,'XVals',0:0.02:1);
[X_UnClass, Y_UnClass, T_UnClass, AUC_UnClass] = perfcurve(targets_UnClass, scores_UnClass(:,1),0);

[c_UnClass, cm_UnClass, ind_UnClass, per_UnClass] = confusion(~targets_UnClass', scores_UnClass(:,1)');

Se_UnClass = cm_UnClass(2,2)/(cm_UnClass(2,2) + cm_UnClass(2,1));
Sp_UnClass = cm_UnClass(1,1)/(cm_UnClass(1,1) + cm_UnClass(1,2));

clearvars targets scores targets_UnClass scores_UnClass c_UnClass cm_UnClass ind_UnClass per_UnClass T_UnClass

%% Plot ROC

% ROCFigue = figure
ROCFigue = figure('units','normalized','outerposition',[0 0 1 1]); 

set(gcf, 'Color', 'w')
% title('Thermography')

myFontSize = 15;

grid on
hold on

plot(X_BalClass(:,1),X_BalClass(:,1),'Color',[0.75 0.75 0.75], 'LineWidth', 2) % Middle Diagonal

% Balanced Class Green
Line_BalClass = plot(X_BalClass(:,1), Y_BalClass(:,1), 'k-', 'LineWidth', 2);

Point_BalClass = plot(1-Sp_BalClass, Se_BalClass, 'o', 'LineWidth', 2,'MarkerEdgeColor', 'k', 'LineWidth', 2, 'MarkerFaceColor','k', 'MarkerSize',8);

% Unbalanced Class Black

Line_UnClass = plot(X_UnClass(:,1), Y_UnClass(:,1), '-', 'color', [0 0.4078 0.2784], 'LineWidth', 2);

Point_UnClass = plot(1-Sp_UnClass, Se_UnClass, 'o', 'LineWidth', 2,'MarkerEdgeColor', [0 0.4078 0.2784], 'LineWidth', 2, 'MarkerFaceColor', [0 0.4078 0.2784], 'MarkerSize',8);



%% Other Points Blue and red

Antony = plot([(1-0.6868)], [1], 'o','LineWidth',2,'MarkerEdgeColor','b','MarkerFaceColor','b','MarkerSize',8);

AroraScreening = plot([(1-0.118)], [0.967], '^','LineWidth',2,'MarkerEdgeColor','b','MarkerFaceColor','b','MarkerSize',8);
AroraClinical = plot([(1-0.441)], [0.90], '^','LineWidth',2,'MarkerEdgeColor','r','MarkerFaceColor','r','MarkerSize',8);
AroraANN = plot([(1-0.265)], [0.967], 'v','LineWidth',2,'MarkerEdgeColor','b','MarkerFaceColor','b','MarkerSize',8);

Button = plot([(1-0.57)], [0.75], 'o','LineWidth',2,'MarkerEdgeColor','r','MarkerFaceColor','r','MarkerSize',8);

Kontos = plot([(1-0.85)], [0.25], '*','LineWidth',2,'MarkerEdgeColor','b','MarkerFaceColor','b','MarkerSize',8);
% Unbalanced, Manual

Tang = plot([(1-0.44)], [0.936], '*','LineWidth',2,'MarkerEdgeColor','r','MarkerFaceColor','r','MarkerSize',8);
% PPV 53, NPV 91.2
% Unbalanced, Identifica puntos calientes

% 4 different studies from Wishart
WishartSent = plot([(1-0.41)], [0.53], '>','LineWidth',2,'MarkerEdgeColor','b','MarkerFaceColor','b','MarkerSize',8);
% PPV 59, NPV 36
% Unbalanced, Sentinel, Medición parámetros temperatura
WishartSentiNN = plot([(1-0.74)], [0.48], '>','LineWidth',2,'MarkerEdgeColor','r','MarkerFaceColor','r','MarkerSize',8);
% PPV 73, NPV 48
% Unbalanced, Sentinel Neural Network
WishartManual = plot([(1-0.48)], [0.78], '<','LineWidth',2,'MarkerEdgeColor','b','MarkerFaceColor','b','MarkerSize',8);
% PPV 69, NPV 49
WishartNoTouchANN = plot([(1-0.48)], [0.70], '<','LineWidth',2,'MarkerEdgeColor','r','MarkerFaceColor','r','MarkerSize',8);
% PPV 67, NPV 51
% Unbalanced, No touch artificial intelligence


Keyserlink = plot([(1-0.81)], [0.83], 's','LineWidth',2,'MarkerEdgeColor','b','MarkerFaceColor','b','MarkerSize',8);
% Manual

Sella = plot([(1-0.725)], [0.905], 's','LineWidth',2,'MarkerEdgeColor','r','MarkerFaceColor','r','MarkerSize',8);

OmnrTher = plot([(1-0.578)], [0.816], '+','LineWidth',2,'MarkerEdgeColor','r','MarkerFaceColor','r','MarkerSize',8);
OmnrTherMam = plot([(1-0.444)], [0.962], '+','LineWidth',2,'MarkerEdgeColor','b','MarkerFaceColor','b','MarkerSize',8);

Krawczyk = plot([(1-0.901)], [0.803], 'p','LineWidth',2,'MarkerEdgeColor','b','MarkerFaceColor','b','MarkerSize',8);
% Accuracy 90.3
% Unbalanced; Make a combination of the classifiers; Subespacios de
% características de datos balanceados

Guirro = plot([(1-0.382)], [0.857], 'p','LineWidth',2,'MarkerEdgeColor','r','MarkerFaceColor','r','MarkerSize',8);
% AUC 65
% Unbalanced, Afectados y controles, Compara las temperaturas de los senos entre paciente afectado y control

% Dynamic protocol, uses Inception V3 and SVM when are doubts, extracts the
% Region of Interest

%% Legends

% BalClass = sprintf('Bal Class = %0.2f ± %0.2f\n', AUC_BalClass(1), mean(abs(AUC_BalClass(1)-AUC_BalClass([2,3]))));
% UnbalClass = sprintf('Unbal Class = %0.2f ± %0.2f\n', AUC_UnClass(1), mean(abs(AUC_UnClass(1)-AUC_UnClass([2,3]))));

BalClass = sprintf('Balanced Class AUC = %0.4f', AUC_BalClass);
UnClass = sprintf('Unbalanced Class AUC = %0.4f', AUC_UnClass);



% For the legends 
% Legend_BalClass = plot(X_BalClass(:,1), Y_BalClass(:,1), '-o', 'color', [0 0.4078 0.2784],...
%     'LineWidth', 2, 'MarkerFaceColor',[0 0.4078 0.2784],'MarkerSize',8);
% Legend_UnClass = plot(X_UnClass(:,1), Y_UnClass(:,1), 'k-o', 'LineWidth', 2, 'MarkerFaceColor',...
%     'k','MarkerSize',8);

Legend_UnClass = plot(nan, nan, '-o', 'color', [0 0.4078 0.2784],...
    'LineWidth', 2, 'MarkerFaceColor',[0 0.4078 0.2784],'MarkerSize',8);
Legend_BalClass = plot(nan, nan, 'k-o', 'LineWidth', 2, 'MarkerFaceColor',...
    'k','MarkerSize',8);

% 
% 
legend([Legend_BalClass, Legend_UnClass,...
    AroraANN, AroraClinical, AroraScreening,...
    Button, Guirro, Keyserlink, Kontos, Krawczyk, Antony, OmnrTher, OmnrTherMam, ...
    Sella, Tang, WishartManual, WishartNoTouchANN, WishartSent, WishartSentiNN],...
    ... Titles of the points
    {BalClass, UnClass, 'Arora ANN', 'Arora Clinical', 'Arora Screening',...
    'Button', 'Guirro', 'Keyserlink', 'Kontos', 'Krawczyk',...
    'Morales-Cervantes', 'Omranipour Thermography',...
    'Omranipour Thermography & Mammography', 'Sella', 'Tang',...
    'Wishart Manual', 'Wishart No Touch NN', 'Wishart Sentinel', 'Wishart Sentinel NN'},...
    'FontSize', myFontSize, 'Location', 'eastoutside');



%% Generalities of the figure
axis square
 box on
 axis tight
xlabel('1-Specificity', 'FontSize', 20)
ylabel('Sensitivity', 'FontSize', 20)
set(gca,  'FontSize', 20)

% title('ROC Curves of Study')
% print('-bestfit','BestFitFigure','-dpdf')

% Put the folder where is going to be saved
% print(ROCFigue, 'Fig4.png', '-dpng', '-r800');
