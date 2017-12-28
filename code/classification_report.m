function [ m_f1, conf ] = classification_report( y_true, y_pred, plot_conf)
%CLASSIFICATION_REPORT Prints a classification report
%   [ m_f1, conf ] = classification_report( y_true, y_pred, plot_conf) only the
%   confusion matrix is returned, however, you can modify the function to
%   return more stats e.g. precision, recall, f1-score or support.
%   Input:  y_true: true labels
%           y_pred: predicted labels
%           plot_conf: plots the normalized confusion matrix if set to true
%   Output: m_f1: weighted average of f1-score
%           conf: confusion matrix
assert( isvector(y_true), 'y_true should be a vector');
assert( isvector(y_pred), 'y_pred should be a vector');
assert( all(ismember(unique(y_pred), unique(y_true))),...
    'some values in y_pred are not present in y_true');
C = max(y_true) - min(y_true) + 1;
assert( C == length(unique(y_true)),...
    'category values must be consecutive integers');

% calls a matlab function confusionmat() in the statistics toolbox
conf = confusionmat(y_true, y_pred);

% precision, recall and f1-scores are all column vectors
support = sum(conf,2);
precision = diag(conf) ./ sum(conf)';
recall = diag(conf) ./ support;
precision(find(isnan(precision))) = 0;
recall(find(isnan(recall))) = 0;
f1score = 2 * precision .* recall ./ (precision + recall);
f1score(find(isnan(f1score))) = 0;

% calculate weighted averages
m_precision = sum (precision .* support) / sum(support);
m_recall = sum (recall .* support) / sum(support);
m_f1 = sum (f1score .* support) / sum(support);

% print results as a table
fprintf('=====================================\n');
fprintf('Precision  Recall  F1-score  Support\n');
for i = 1:C
    fprintf('   %.2f     %.2f     %.2f      %d\n',...
        precision(i), recall(i), f1score(i), support(i));
end
fprintf('-------------------------------------\n');
fprintf('   %.2f     %.2f     %.2f      %d\n\n',...
    m_precision, m_recall, m_f1, sum(support));
fprintf('=====================================\n');
% plot a normalized confusion matrix if requested
if plot_conf == true
    normalized_conf = conf ./ repmat(support, [1 C]);
    imagesc(normalized_conf);
    title('Normalized Confusion Matrix');
    colorbar();
end


end

