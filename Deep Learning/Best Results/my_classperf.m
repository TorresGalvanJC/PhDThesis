function cp = my_classperf(ConfMtx)
% Evaluate performance of classifier based on the confusion matrix
% Always assume the first index as the positive class and the second as the
% control class. Predicted values are considered row-wise and ground truth
% (true labels) are column-wise.
cp.TP       = ConfMtx(1,1);                  % True positives
cp.FP       = ConfMtx(2,1);                  % Type I error
cp.FN       = ConfMtx(1,2);                  % Type II error
cp.TN       = ConfMtx(2,2);                  % True negatives
cp.P        = cp.TP + cp.FN;            % Positives (total)
cp.N        = cp.TN + cp.FP;            % Negatives (total)
cp.Se       = cp.TP / (cp.TP + cp.FN);  % Sensitivity (recall or True Positive Rate)
cp.Sp       = cp.TN / (cp.TN + cp.FP);  % Specificity (True Negative Rate)
cp.PPV      = cp.TP / (cp.TP + cp.FP);  % Positive Predictive Value (precision)
cp.NPV      = cp.TN / (cp.TN + cp.FN);  % Negative Predictive Value
cp.Prev     = cp.P / (cp.P + cp.N);     % Prevalence
cp.Acc      = (cp.TP + cp.TN) / (cp.P + cp.N);      % Accuracy
cp.Error    = 1 - cp.Acc;               % Error
cp.BA       = 0.5*(cp.Se + cp.Sp);      % Balanced Accuracy
cp.Gmean    = sqrt(cp.Sp * cp.Se);      % G-mean
cp.Fmeasure = (2*cp.PPV*cp.Se) / (cp.PPV + cp.Se);  % F-measure
cp.MCC      = (cp.TP*cp.TN - cp.FP*cp.FN) / sqrt(cp.P*cp.N*(cp.TP + cp.FP)*(cp.TN + cp.FN)); % Matthews Correlation Coefficient
cp.Plh      = cp.Se / (1 - cp.Sp);      % Positive Likelihood 	
cp.Nlh      = (1 - cp.Se) / cp.Sp;      % Negative Likelihood 	
cp.DOR      = cp.Plh / cp.Nlh;          % Diagnostic Odds Ratio
cp.p0       = cp.Acc;                   % total accuracy p0 for Cohen's kappa
cp.pe 	    = ( ( cp.P*( (cp.TP + cp.FP)/(cp.P + cp.N) ) ) + (cp.TP*( (cp.FN+cp.TN)/(cp.P + cp.N) )) ) / (cp.P+cp.N); % random accuracy pe for Cohen's kappa
cp.kappa    = (cp.p0 - cp.pe) / (1 - cp.pe);    % Cohen's kappa
cp.Matthews = ( ( (cp.TP*cp.TN)-(cp.FP*cp.FN) ) / ...
    ((cp.TP + cp.FP)*(cp.TP + cp.FN)*(cp.TN - cp.FP)*(cp.TN + cp.FN))^1/2 ); % Matthews Correletion Coefficient
cp.FDR      = cp.FP / (cp.TP + cp.FP);       % False Discovery Rate
cp.FPR      = cp.FP / (cp.FP + cp.TN);       % False Positive Rate
cp.FNR      = cp.FN / (cp.TP + cp.FN);       % False Negative Rate
% Mutual information
% Area Under Curve (AUC)
end

% EOF
