function [Scores,Lpred] = LDAandQDAfunctionTest(Xtest, Means, Covariances, Priors, Classifier)

%% Prediction of test data set using LDA and QDA 
%Xtest: test samples to predict the labels
%Mean Priors, Covarinaces matrix from training of LDA or QDA 
%Classifier : To choose LDA or QDA 
% scores: posteriors of each sample for each class arranged in rows to column
% Lpred : predicted labels for class i in column i.
%%

[N,d] = size(Xtest);
Scores = zeros(N,3);

switch Classifier 
  
    case 1
% LDA
for i = 1 : 3
Scores(:,i)= log (Priors(i,1))+  (-0.5)*Means(i,:)*inv(Covariances(:,:,1))*Means(i,:)'+(Means(i,:) * inv(Covariances(:,:,1))* Xtest(:,:)')  ;
end
    case 2 
% QDA       
for i = 1: 3
     for j = 1 : N 
 Scores(j,i) =-(0.5)*log(det(Covariances(:,:,i)))+log(Priors(i,1))-((0.5).*(Xtest(j,:)-Means(i,:))*inv(Covariances(:,:,i))*(Xtest(j,:)-Means(i,:))') ;
     end
end
 
end


Lpred = zeros(N,3);
for i = 1:N
    if Scores(i,1) > Scores(i,2) &&  Scores(i,1) > Scores(i,3)
        Lpred (i,1) = 1;
    
    else if  Scores(i,2) > Scores(i,1) &&  Scores(i,2) > Scores(i,3)
             Lpred (i,2) = 2;
        
        else if  Scores(i,3) > Scores(i,1) &&  Scores(i,3) > Scores(i,2)
                 Lpred (i,3) = 3;
      
            end
        end
    end
end


