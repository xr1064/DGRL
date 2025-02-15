function L = constructWK1(gnd,data,options,k1,theo1)
%             Input:
%               data       - Data matrix. Each row vector of fea is a data point.
%
%               gnd     - Label vector.  
%
%               options - Struct value in Matlab. The fields in options
%                         that can be set:
%                     k          = 0  
%                                     Wb:
%                                       Put an edge between two nodes if and
%                                       only if they belong to different classes. 
%                                     Ww:
%                                       Put an edge between two nodes if and
%                                       only if they belong to same class. 
%                                > 0
%                                     Wb:
%                                       Put an edge between two nodes if
%                                       they belong to different classes
%                                       and they are among the k nearst
%                                       neighbors of each other. 
%                                     Ww:
%                                       Put an edge between two nodes if
%                                       they belong to same class and they
%                                       are among the k nearst neighbors of
%                                       each other.  
%                     beta         [0,1] Paramter to tune the weight between
%                                        within-class graph and between-class
%                                        graph. Default 0.1. 
%                                        beta*L_b+(1-beta)*W_w 
%



if (~exist('options','var'))
   options = [];
end


[nSmp,nFea] = size(data);
if length(gnd) ~= nSmp
    error('gnd and data mismatch!');
end

k = 0;
if k1 < nSmp-1
    k = k1;
end


theo = 0.1;
if (theo1 > 0) && (theo1 < 1)
    theo = theo1;
end


Label = unique(gnd);
nLabel = length(Label);

Ww = zeros(nSmp,nSmp);
Wb = ones(nSmp,nSmp);
for idx=1:nLabel
    classIdx = find(gnd==Label(idx));
    Ww(classIdx,classIdx) = 1;
    Wb(classIdx,classIdx) = 0;
end

if k > 0
    D = HibertDist2(data,[],options,0);
    [dump idx] = sort(D,2); % sort each row
    clear D dump
    idx = idx(:,1:k+1);
    
    G = sparse(repmat([1:nSmp]',[k+1,1]),idx(:),ones(prod(size(idx)),1),nSmp,nSmp);
    G = max(G,G');
    Ww = Ww.*G;
    Wb = Wb.*G;
    clear G
end

Lw = calDifLaplacian(Ww,'StandardNorm'); 
Lb = calDifLaplacian(Wb,'StandardNorm'); 
L = theo*Lw-(1-theo)*Lb;