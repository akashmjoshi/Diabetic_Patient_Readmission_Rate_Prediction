%% Data Seperation 
function [feature_train,label_train,feature_test,label_test] = dataseparation(diabeticdata)
warning off;
label_dataset = (diabeticdata(:,end));
label_dataset(find(strcmp(table2cell(label_dataset), 'NO')),1) = {'3'};   
label_dataset(find(strcmp(table2cell(label_dataset), '>30')),1) = {'2'};   
label_dataset(find(strcmp(table2cell(label_dataset), '<30')),1) = {'1'};
label_dataset = double(cell2mat(table2cell(label_dataset)))-48;  
cv = cvpartition(label_dataset,'Kfold',2);
feature_train = (diabeticdata(find(cv.training(1)==1),1:49)); 
label_train = (label_dataset(find(cv.training(1)==1))); 
feature_test = (diabeticdata(find(cv.training(1)==0),1:49));  
label_test = (label_dataset(find(cv.training(1)==0)));
end

