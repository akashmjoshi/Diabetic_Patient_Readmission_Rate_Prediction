% test the classification_report() function

%% read results
% assume we have some results with label values [0 1 2]
result = csvread('results.csv');
ypred = result(:,1);
ytrue = result(:,2);

%% test help message
help classification_report

%% sanity check on classification restuls
classification_report(ytrue, ypred, true);

%% test handling more than 3 classes
temp_pred = [ypred; 3];
temp_true = [ytrue; 3];
classification_report(temp_true, temp_pred, true);

%% test handling shifted labels
temp_pred = ypred + 100;
temp_true = ytrue + 100;
classification_report(temp_true, temp_pred, true);

%% test handling unknown labels
temp_pred = ypred + 100; % some values in ypred are not in ytrue
temp_true = ytrue + 101;
classification_report(temp_true, temp_pred, true);

%% test handling non consecutive labels
temp_pred = [ypred; 4];
temp_true = [ytrue; 4];
classification_report(temp_true, temp_pred, true);

%% test handling majority guess
% some NaN values are expected
temp_true = ytrue;
temp_pred = zeros(size(ytrue));
classification_report(temp_true, temp_pred, true);

%% test handling random guess
temp_true = ytrue;
temp_pred = randi(3,size(ytrue))-1;
classification_report(temp_true, temp_pred, true);