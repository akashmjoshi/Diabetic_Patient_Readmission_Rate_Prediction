function [ACC,VAR] = cvsvmtrain(cv1,cv2,cv3,training_label_vector, training_instance_matrix, stringsend,nfold)
for i = 1 : nfold
    cv = [cv1.training(i) ; cv2.training(i) ; cv3.training(i)];
    model = svmtrain((double(training_label_vector(find(cv==1),1))),...
        training_instance_matrix(find(cv==1),:),stringsend);
    [~, accuracy(:,:,i),~] = svmpredict((double(training_label_vector(find(cv==0),1))),...
        training_instance_matrix(find(cv==0),:),model, '-q');
end
ACC = mean(accuracy(1,1,:));
VAR = var(accuracy(1,1,:));
end