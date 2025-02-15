function [TrainingAccuracy, TestingAccuracy, trainOutlabel, testOutlabel,trainYhat,testYhat] = DGRL(trainX, train_label,testX,test_label, options)
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This is an implementation of DGRL with Matlab 2017b. 
% If you use it in your research, please cite the paper "Discriminative Graph Regularized Representation Learning for Recognition"
%% Parameters:
% trainX:  Nfeatures * Nsamples matrix
% train_label: Nsamples * 1 matrix(vector)
% testX:  Nfeatures * Msamples matrix
% test_label: Msamples * 1 matrix(vector)
% options: 
%   options.bCentered
%   options.k
%   options.t
%   options.theo
%   options.alpha
%   options.lambda
%   options.GraphyMode: 1.'SOLPP';2.'DRLSC'
%   options.LaplacianNorm
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if (~exist('options','var'))
    options = [];
end

if ~isfield(options,'bCentered') || isempty(options.bCentered)
    options.bCentered = 0;
end

if ~isfield(options,'k') || isempty(options.k)
    options.k = 5;
end

if ~isfield(options,'t') || isempty(options.t)
    options.t = 1;
end

if ~isfield(options,'theo') || isempty(options.theo)
    options.theo = 0.5;   
end

if ~isfield(options,'alpha') || isempty(options.alpha)
    options.alpha = 1;
end

if ~isfield(options,'lambda') || isempty(options.lambda)
    options.lambda = 1;
end

if ~isfield(options,'GraphyMode') || isempty(options.GraphyMode)
    options.GraphyMode = 'SOLPP';
end

if ~isfield(options,'LaplacianNorm') || isempty(options.LaplacianNorm)
    options.LaplacianNorm = 'StandardNorm';
end


if ~options.bCentered
    
    mean_trainX = mean(trainX,2);                
    trainX = bsxfun(@minus,trainX,mean_trainX);  
    testX = bsxfun(@minus,testX,mean_trainX);    
end


G = label2matrix(train_label);                   
trainY = G*sqrt(inv(G'*G));                      



switch lower(options.GraphyMode)
    case {lower('SOLPP')}
        graphoptions = [];
        graphoptions.NeighborMode = 'Supervised';  
        graphoptions.k = options.k;                
        graphoptions.gnd = train_label;            
        graphoptions.WeightMode = 'HeatKernel';    
        graphoptions.t = options.t;                
        
        %%%%%%%%%%%
        W1 = constructW1(trainX',graphoptions);         
        W2 = constructW2(trainX',graphoptions);         
        W = W1 + W2;                                    
        L = calDifLaplacian(W,options.LaplacianNorm);   
        
        %%%%%%%%%%%
        clear W1 W2 W
    case {lower('DRLSC')}
        
        graphoptions = [];
        graphoptions.WeightMode = 'Binary';        
        graphoptions.k = options.k;                
        graphoptions.bSelfConnected = 1;           
        
        %%%%%%%%%%%
        W = constructW(trainX',graphoptions);      
        lm = G*G';                                 
        lms = sparse(lm);
        Gw = W;
        Gb = W;
        Gw = Gw.*lms;                              
        Gb = Gb.*not(lms);                         
        
        %%%%%%%%%%%
        Lw = calDifLaplacian(Gw,options.LaplacianNorm);   
        Lb = calDifLaplacian(Gb,options.LaplacianNorm);   
        theo = options.theo;
        L = theo*Lw-(1-theo)*Lb;
        
        %%%%%%%%%%%
        clear W lm lms Gw Gb Lw Lb G theo
    otherwise
        error('GraphyMode does not exist!');
end

%%
[U1,Sigma1,V1] = mySVD(trainX);             

%%%%%%%%%%% 
Alpha = options.alpha;                      
Sigma2 = Sigma1*V1'*(eye(size(trainX,2))+Alpha*full(L))*V1*Sigma1;         
[P,D1] = eig(Sigma2);                       
[P,D1] = eigsorts(P,D1,'descend');          

%%%%%%%%%%% 
Lambda = options.lambda;                    
Sigma3 = D1+Lambda*eye(size(D1));
Hb = trainX*trainY;
B = Sigma3^(-1/2)*P'*U1'*Hb;
[Ub,Sigmab,Vb] = mySVD(B);

%%%%%%%%%%% 
q = min(rank(Hb),size(trainY,2)-1);          
M = Sigma3^(-1/2)*Ub;                        
Ms = M(:,1:q);                                                           
A = U1*P*Ms;


%%  
trainYhat = trainX'*A*A'*Hb;
[~,trainOutlabel] = max(trainYhat,[],2);
TrainingAccuracy = mean(train_label==trainOutlabel)


%% 
testYhat = testX'*A*A'*Hb;
[~,testOutlabel] = max(testYhat,[],2);
TestingAccuracy = mean(test_label==testOutlabel)

