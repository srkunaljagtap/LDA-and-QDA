function [Means,Covariances, Priors] = LDAandQDAfunct(Xtrain,Ltrain,classifier_type)
 %%
 %Xtrain: contains Ntrain training samples arranged in rows (each
% sample is D-dimensional) 
% Ltrain:contains the corresponding class labels in a single
% column. 
% If C is the number of classes represented in Xtrain, you can safely
%  assume that the labels in Ltrain and Ltest will belong to f1; 2; : : : ;Cg. 
% You should also assume that Xtrain always contains
% at least one sample from each one of the C classes. 
% classifier type equal to 1; 2; 3 specifies
% the use of a LDA model (1 for the general, 2 for the Naive Bayes and 
%3 for the isotropic variants) and
% classifier type equal to 4; 5; 6 specifies 
%the use of a QDA model (4 for the general, 5 for the Naive
% Bayes and 3 for the isotropic variants). 
% Regarding the output variables 
% Means : should contain the estimated means in rows 
% (row k contains the estimate for class k), 
% Covariances : should contain the covariance matrix 
% estimates of the classes and Priors 2 RC should contain the estimated
% class priors.
%% Divide the Xtrain into sets according to their classes
[N d] = size(Xtrain);

indx1= find(Ltrain==1);
indx2= find(Ltrain==2);
indx3= find(Ltrain==3);

Strain1(:,1) = Xtrain(indx1,1);
Strain1(:,2) = Xtrain(indx1,2);
Strain2(:,1) = Xtrain(indx2,1);
Strain2(:,2) = Xtrain(indx2,2);
Strain3(:,1) = Xtrain(indx3,1);
Strain3(:,2) = Xtrain(indx3,2);
%%
% mean calculation
M1=mean(Strain1);
M2=mean(Strain2);
M3=mean(Strain3);
Means = vertcat (M1,M2,M3);

%%
% prior probability calculation
L1 = length (Strain1);
L2 = length (Strain2);
L3 = length (Strain3);
p1=L1/N;
p2=L2/N;
p3=L3/N;
Priors = vertcat(p1,p2,p3);
%% Find out Covariance matrices according to type of classifier
% Case 1 LDA general
% Case 2  LDA isotrpic
% Case 3 LDA Naive bayes
% Case 4 QDA general
% Case 5 QDA isotropic
% Case 6 QDA Naive bayes

C1=zeros(d,d);
C2=zeros(d,d);
C3=zeros(d,d);
I=eye (2);
switch classifier_type
    
    
    case 1
       
 C1 = (1/L1)* ((Strain1(:,:)-repmat(Means(1,:),L1,1))'* (Strain1(:,:)-repmat(Means(1,:),L1,1)));  
 C2 =(1/L2)*  ((Strain2(:,:)-repmat(Means(2,:),L2,1))'* (Strain2(:,:)-repmat(Means(2,:),L2,1)));  
 C3 =(1/L3)*  ((Strain3(:,:)-repmat(Means(3,:),L3,1))'* (Strain3(:,:)-repmat(Means(3,:),L3,1)));  
        
  Covariances(:,:,1)= (1/3) * ((C1 +C2+ C3));    
    
    case 2    
C1 = (1/L1)* (norm(Strain1(:,:)-repmat(Means(1,:),L1,1)).^2);  
C2 =(1/L2)*  (norm(Strain2(:,:)-repmat(Means(1,:),L2,1)).^2);  
C3 =(1/L3)*  (norm(Strain3(:,:)-repmat(Means(1,:),L3,1)).^2);  
       
 Covariances(:,:,1)= (1/3)*((C1+C2+C3)* I );        
        
      case 3
C1 =(1/L1)* sum((Strain1(:,:)-repmat(Means(1,:),L1,1)).^2);  
C2 =(1/L2)* sum((Strain2(:,:)-repmat(Means(1,:),L2,1)).^2);  
C3 =(1/L3)* sum((Strain3(:,:)-repmat(Means(1,:),L3,1)).^2);  
      
 Covariances(:,:,1)= (1/3) * diag(C1+ C2+ C3);    
          
    case 4 
 Covariances(:,:,1)  = (1/L1)* ((Strain1(:,:)-repmat(Means(1,:),L1,1))'* (Strain1(:,:)-repmat(Means(1,:),L1,1)));  
 Covariances(:,:,2)  = (1/L2)*  ((Strain2(:,:)-repmat(Means(2,:),L2,1))'* (Strain2(:,:)-repmat(Means(2,:),L2,1)));  
 Covariances(:,:,3)  = (1/L3)*  ((Strain3(:,:)-repmat(Means(3,:),L3,1))'* (Strain3(:,:)-repmat(Means(3,:),L3,1)));  
        
    case 5
Covariances(:,:,1)  = ((1/L1)* (norm(Strain1(:,:)-repmat(Means(1,:),L1,1)).^2)) * I;  
Covariances(:,:,2)  = ((1/L2)* (norm(Strain2(:,:)-repmat(Means(1,:),L2,1)).^2)) * I;  
Covariances(:,:,3)  = ((1/L3)* (norm(Strain3(:,:)-repmat(Means(1,:),L3,1)).^2)) * I;     
        
       
    case 6
Covariances(:,:,1) =diag((1/L1)* sum((Strain1(:,:)-repmat(Means(1,:),L1,1)).^2));  
Covariances(:,:,2) =diag((1/L2)* sum((Strain2(:,:)-repmat(Means(1,:),L2,1)).^2));  
Covariances(:,:,3) =diag((1/L3)* sum((Strain3(:,:)-repmat(Means(1,:),L3,1)).^2));  
      
   
end