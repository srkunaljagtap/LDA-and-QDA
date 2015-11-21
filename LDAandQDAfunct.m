function [Mean,Covariance, Prior] = LDAandQDAfunct(Xtrain,Ltrain,Number_Classes,classifier_type)
 %%
% Xtrain: contains Ntrain training samples arranged in rows (each
% sample is D-dimensional) 
% Ltrain:contains the corresponding class labels in a single
% column. 
% If C is the number of classes represented in Xtrain, you can safely
%  assume that the labels in Ltrain and Ltest will belong to f1; 2; : : : ;Cg. 
% You should also assume that Xtrain always contains
% at least one sample from each one of the C classes. 
% classifier type equal to 1; 2; 3 specifies
% the use of a LDA model (1 for the general, 2 for the Naive Bayes and 
% 3 for the isotropic variants) and
% classifier type equal to 4; 5; 6 specifies 
% the use of a QDA model (4 for the general, 5 for the Naive
% Bayes and 3 for the isotropic variants). 
% Regarding the output variables 
% Means : should contain the estimated means in rows 
% (row k contains the estimate for class k), 
% Covariances : should contain the covariance matrix 
% estimates of the classes and Priors 2 RC should contain the estimated
% class priors.
%% Divide the Xtrain into sets according to their classes
[N1,D1]= size(Xtrain);
k = Number_Classes;
for i= 1 : k
    Indx{:,i} = find (Ltrain == i);
end
Strain = cell(1,k);
for i = 1:k
   Strain{1,i} = Xtrain(Indx{:,i},:); 
end

%%
% mean calculations
Mean = cell(1,k);
for i = 1:k
   Mean{1,i} = mean(Strain{1,i});
end

%%
% prior probability calculation
L = zeros(1,k);
Prior = zeros(1,k);
for i=1:k 
   L(1,i) = length(Strain{1,i});
   Prior(1,i) =  L(1,i) / N1 ;
end

%% Find out Covariance matrices according to type of classifier
% Case 1 LDA general
% Case 2  LDA isotrpic
% Case 3 LDA Naive bayes
% Case 4 QDA general
% Case 5 QDA isotropic
% Case 6 QDA Naive bayes

Covariances = cell(1,k);
I=eye (D1);
Covariance = zeros(D1,D1,k);
switch classifier_type
      
    case 1
 for i = 1:k      
Covariances{1,i} =(1/L(1,i))*((Strain{1,i}-repmat(Mean{1,i},L(1,i),1))'* (Strain{1,i}-repmat(Mean{1,i},L(1,i),1)));  
Covariance(:,:,1) = Covariance(:,:,1) + Covariances{1,i};
 end 
Covariance(:,:,1) = (1/k) * Covariance(:,:,1);   
    
    case 2   
 for i = 1:k          
Covariances{1,i} = (1/L(1,i))* (norm(Strain{1,i}-repmat(Mean{1,i},L(1,i),1)).^2);  
Covariance(:,:,1)  = Covariance(:,:,1) + Covariances{1,i};
 end
Covariance(:,:,1) = (1/k)*(Covariance(:,:,1) * I );        
        
    case 3
for i = 1:k 
Covariances{1,i} =(1/L(1,i))* sum((Strain{1,i}-repmat(Mean{1,i},L(1,i),1)).^2);  
Covariance(1,:,1)  = Covariance(1,:,1) + Covariances{1,i};
end
Covariance(:,:,1)= (1/k) * diag(Covariance(1,:,1)); 
 
    case 4 
 for i = 1:k      
 Covariance(:,:,i)  = (1/L(1,i))* ((Strain{1,i}-repmat(Mean{1,i},L(1,i),1))'* (Strain{1,i}-repmat(Mean{1,i},L(1,i),1)));  
 end
    case 5
for i = 1:k             
Covariance(:,:,i)  = ((1/L(1,i))* (norm(Strain{1,i}-repmat(Mean{1,i},L(1,i),1))).^2) * I;  
end
    case 6
for i = 1:k             
Covariance(:,:,i) = diag((1/L(1,i))* sum((Strain{1,i}-repmat(Mean{1,i},L(1,i),1)).^2));  
end

end
