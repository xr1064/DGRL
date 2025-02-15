function J =Auto_best_RBF(data,gnd,sigma)


kerneloptions = [];
kerneloptions.KernelType =  'Gaussian';
kerneloptions.t = sigma;
K = constructKernel(data,[],kerneloptions);

%%================
nSmp = size(data,1);
En = (1/nSmp)*ones(nSmp,nSmp);
K = K-K*En-En*K+En*K*En;
%%================

Label = unique(gnd);
nLabel = length(Label);
withinSum = 0;
numSam = zeros(1,nLabel);
for idx=1:nLabel
    classIdx = find(gnd==Label(idx));
    Kw = K(classIdx,classIdx);
    withinSum = withinSum+sum(Kw(:));
    numSam(idx) = length(classIdx);
end

betweenSum = sum(K(:))-withinSum;
w = withinSum/sum(numSam.^2);
b = betweenSum/((sum(numSam))^2-sum(numSam.^2));

J = 1-w+b;