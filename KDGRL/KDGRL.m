function [TrainingAccuracy, TestingAccuracy, trainOutlabel, testOutlabel,trainYhat,testYhat] = KDGRL(trainX, train_label,testX,test_label, options)

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This is an implementation of KDGRL with Matlab 2017b. 
% If you use it in your research, please cite the paper "Discriminative Graph Regularized Representation Learning for Recognition"
%% Parameters:
% trainX: Nfeatures * Nsamples  matrix
% train_label: Nsamples * 1 matrix(vector)
% testX: Nfeatures * Msamples matrix
% test_label: Msamples * 1 matrix(vector)

% options: 
%   options.gamma
%   options.k
%   options.t
%   options.theo: 0<theo<1
%   options.alpha: alpha>=0
%   options.lambda: lambda>=0
%   options.GraphyMode: 1.'SOLPP';2.'DRLSC'
%   options.LaplacianNorm

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if (~exist('options','var'))
    options = [];
end

if ~isfield(options,'gamma') || isempty(options.gamma)
    options.gamma = 1;
end

if ~isfield(options,'k') || isempty(options.k)
    options.k = 5;
end

if ~isfield(options,'t') || isempty(options.t)
    options.t = 1;
end

if ~isfield(options,'theo') || isempty(options.theo)
    options.theo = 0.9;   %L = theo*Lw-(1-theo)*Lb;
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




%%%%%%%%%%% 
G = label2matrix(train_label);  
trainY = G*sqrt(inv(G'*G));     
clear G

%%%%%%%%%%% 
kerneloptions = [];
kerneloptions.KernelType =  'Gaussian';           
kerneloptions.t = options.gamma;                 
K = constructKernel(trainX',[],kerneloptions);    

%%%%%%%%%%% 
nSmp = size(trainX,2);
En = (1/nSmp)*ones(nSmp,nSmp);
K_Center = K-K*En-En*K+En*K*En;

%% 
switch lower(options.GraphyMode)
    case {lower('SOLPP')}
        graphoptions = [];
        graphoptions.NeighborMode = 'Supervised';    
        graphoptions.k = options.k;                  
        graphoptions.gnd = train_label;              
        graphoptions.WeightMode = 'HeatKernel';      
        graphoptions.t = options.t;                  
        
        %%%%%%%%%%%
        W1 = constructWK2(trainX',graphoptions,kerneloptions);   
        W2 = constructWK3(trainX',graphoptions,kerneloptions);   
        W = W1 + W2;                                             
        L = calDifLaplacian(W,options.LaplacianNorm);            
        
        %%%%%%%%%%%
        clear W1 W2 W graphoptions;
        
    case {lower('DRLSC')}
        Label = unique(train_label);
        nLabel = length(Label);
        
        Ww = zeros(nSmp,nSmp);
        Wb = ones(nSmp,nSmp);
        for idx=1:nLabel
            classIdx = find(train_label==Label(idx));
            Ww(classIdx,classIdx) = 1;
            Wb(classIdx,classIdx) = 0;
        end
        
        %%%%%%%%%%% 
        k = options.k;                                  
        D = HibertDist2(trainX',[],kerneloptions,0);    
        [dump idx] = sort(D,2);                        
        clear D dump
        idx = idx(:,1:k+1);
        
        %%%%%%%%%%% 
        G = sparse(repmat([1:nSmp]',[k+1,1]),idx(:),ones(prod(size(idx)),1),nSmp,nSmp);
        G = max(G,G');                                  
        Ww = Ww.*G;                                     
        Wb = Wb.*G;                                    
        clear G
        
        %%%%%%%%%%% 
        Lw = calDifLaplacian(Ww,options.LaplacianNorm); 
        Lb = calDifLaplacian(Wb,options.LaplacianNorm); 
        theo = options.theo;
        L = theo*Lw-(1-theo)*Lb;
        
        %%%%%%%%%%%
        clear Label nLabel Ww Wb idx Lw Lb theo k
        
    otherwise
        error('GraphyMode does not exist!');
end

%%
[U1,Sigma1,V1] = mySVD(K_Center);  % t = rank(K);% [U1,Sigma1,V1] = svds(K,t); %[U1,Sigma1,V1] = svd(K,'econ');

%%%%%%%%%%%
Alpha = options.alpha;             
Lambda = options.lambda;           

%%%%%%%%%%% 
Sigma2 = Sigma1*Sigma1+Lambda*Sigma1+Alpha*Sigma1*U1'*full(L)*U1*Sigma1;
[P,D1] = eig(Sigma2);              
[P,D1] = eigsorts(P,D1,'descend'); 

%%%%%%%%%%% 
Hb = K_Center*trainY;
B = D1^(-1/2)*P'*U1'*Hb;           
[Ub,Sigmab,Vb] = mySVD(B);

%%%%%%%%%%% 
q = min(rank(Hb),size(trainY,2)-1); 
M = D1^(-1/2)*Ub;                   
Ms = M(:,1:q);                      
A = U1*P*Ms;                       

%%
trainYhat = K_Center*A*A'*Hb;
[~,trainOutlabel] = max(trainYhat,[],2);
TrainingAccuracy = mean(train_label==trainOutlabel)


%%
mSmp = size(testX,2);            
Emn = (1/nSmp)*ones(mSmp,nSmp);
Kzx = constructKernel(testX',trainX',kerneloptions);
Kzx_Center = Kzx - Kzx*En - Emn*K + Emn*K*En;

testYhat = Kzx_Center*A*A'*Hb;
[~,testOutlabel] = max(testYhat,[],2);
TestingAccuracy = mean(test_label==testOutlabel)

