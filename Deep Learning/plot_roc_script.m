% This is for the special case of Breast Cancer Thermography, where 
% 0 = anormal and 1 = normal, it's upside down of the usual case

% Compute performance curve
% load('C:\Users\ADMIN\Documents\Universidad\Doctorado\Termogramas\ResNet 101\200 Brasil + IJC Results\Primeras Aproximaciones\Calis_68.mat')
 load('E:\Archivos Juan Carlos Torres\MATLAB\Deep Learning\Results\ResNet101\Iteraciones\Balanced\RN_BalCl_MinBat10_1239.mat')
targets = repmat(targets, [10, 1]);
scores = repmat(scores, [10, 1]);
% [X,Y,T,AUC] = perfcurve(targets,scores(:,1),0, 'NBoot',100,'XVals',0:0.02:1);
[X,Y,T,AUC] = perfcurve(targets,scores(:,1),0);
close all
[c,cm,ind,per] = confusion(~targets',scores(:,1)');

%% Plot ROC
figure; set(gcf, 'Color', 'w')
title('Thermography')
% errorbar(X(:,1),Y(:,1), Y(:,1)-Y(:,2), Y(:,3)-Y(:,1), 'k-', 'LineWidth', 2)
plot(X(:,1),Y(:,1), 'k-', 'LineWidth', 2)
grid on
hold on
myFontSize = 14;
% plot(1-per(1,4), per(1,3), 'ro', 'LineWidth', 2)
plot(1-cm(1,1)/(cm(1,1) + cm(2,1)), cm(2,2)/(cm(2,2) + cm(1,2)), 'ro', 'LineWidth', 2)
% plot(1-cm(2,2)/(cm(2,2) + cm(1,2)), cm(1,1)/(cm(1,1) + cm(2,1)), 'ro', 'LineWidth', 2)
axis square
xlabel('1-Specificity', 'FontSize', myFontSize)
ylabel('Sensitivity', 'FontSize', myFontSize)
set(gca,  'FontSize', myFontSize)
plot(X(:,1),X(:,1),'Color',[0.75 0.75 0.75], 'LineWidth', 2)
text(0.5,0.2,sprintf('AUC = %0.2f', AUC), 'FontSize', myFontSize)



%% Free Code for Practice

TheFig = Figure
hold on

LineBalClass = plot([0:1],[0:1], 'r-o', 'LineWidth', 2,'MarkerEdgeColor','r','MarkerFaceColor','r','MarkerSize',8);
LineBalClass = plot([0:1],[0:1], 'r-o', 'LineWidth', 2,'MarkerEdgeColor','r','MarkerFaceColor','r','MarkerSize',8);

LineNoBalClass = plot(XNoBalClas,YNoBalClas, 'b-o', 'LineWidth', 2);

TorresBalClass = plot([(1-SpBalC)], [SeBalC], 'o','LineWidth',2,'MarkerEdgeColor','r','MarkerFaceColor','r','MarkerSize',8);
%%
clear; clc;
load('C:\Users\ADMIN\Documents\Universidad\Doctorado\Termogramas\ResNet 101\200 Brasil + IJC Results\Primeras Aproximaciones\Calis_68.mat');
close all

accuracy = mean(predictedLabels == validationImages.Labels)
targets = grp2idx(validationImages.Labels);
targets(targets==2) = 0;
targets = ~targets;
outputs = grp2idx(predictedLabels);
outputs(outputs==2) = 0;
outputs = ~outputs;
[c,cm,ind,per] = confusion(targets', outputs');

Confusion_Matrix=figure; plotconfusion(targets', outputs')
addpath(genpath('C:\Users\ADMIN\Dropbox\Doctorado\MATLAB Codigos'))

cp = my_classperf(cm);

[X,Y,T, AUC] = perfcurve (~targets', scores(:,1)', 0)

SeNoBalC = cp.Se;
SpNoBalC = cp.Sp;

plot(1-SpNoBalC, SeNoBalC, 'ro', 'LineWidth', 2)

%% Another Free Code for Practice

%% Charge Balanced Class File
load('C:\Users\ADMIN\Documents\Universidad\Doctorado\Termogramas\ResNet 101\Balanced Class Brasil + IJC Results\Primeras Aproximaciones\Calis_45.mat',...
    'targets', 'scores', 'cp', 'cm');

close all


SeBalC = cp.Se;
SpBalC = cp.Sp;

plot(1-cm(2,2)/(cm(2,2) + cm(1,2)), cm(1,1)/(cm(1,1) + cm(2,1)), 'ro', 'LineWidth', 2)

%%

%% Iris data example
load fisheriris
% Use only the first two features as predictor variables. Define a binary classification problem by using only the measurements that correspond to the species versicolor and virginica.
targets = meas(51:end,1:2);
% Define the binary response variable.
resp = (1:100)'>50;  % Versicolor = 0, virginica = 1
% Fit a logistic regression model.
mdl = fitglm(targets,resp,'Distribution','binomial','Link','logit');
% Compute the ROC curve. Use the probability estimates from the logistic regression model as scores.
scores = mdl.Fitted.Probability;
targets = species(51:end,:);
[X,Y,T,AUC] = perfcurve(targets,scores,'virginica');
targets = strcmp(targets, 'virginica');
[c,cm,ind,per] = confusion(targets',scores');

%% Plot ROC
figure; set(gcf, 'Color', 'w')
title('Iris data example')
plot(X,Y, 'k-', 'LineWidth', 2)
hold on
myFontSize = 14;
plot(1-per(1,4), per(1,3), 'ro', 'LineWidth', 2)
axis square
xlabel('1-Specificity', 'FontSize', myFontSize)
ylabel('Sensitivity', 'FontSize', myFontSize)
set(gca,  'FontSize', myFontSize)
plot(X,X,'Color',[0.75 0.75 0.75], 'LineWidth', 2)
text(0.5,0.2,sprintf('AUC = %0.2f', AUC), 'FontSize', myFontSize)

