function D = HibertDist2(fea_a,fea_b,options,bSqrt)
%EUDIST2 Efficiently Compute the Euclidean Distance Matrix by Exploring the
%Matlab matrix operations.
%
%   D = EuDist(fea_a,fea_b)
%   fea_a:    nSample_a * nFeature
%   fea_b:    nSample_b * nFeature
%   D:      nSample_a * nSample_a
%       or  nSample_a * nSample_b
%
%    Examples:
%
%       a = rand(500,10);
%       b = rand(1000,10);
%
%       A = EuDist2(a); % A: 500*500
%       D = EuDist2(a,b); % D: 500*1000
%
%   version 2.1 --November/2011
%   version 2.0 --May/2009
%   version 1.0 --November/2005
%
%   Written by Deng Cai (dengcai AT gmail.com)


if ~exist('bSqrt','var')
    bSqrt = 1;
end

if (~exist('fea_b','var')) || isempty(fea_b)
    K = constructKernel(fea_a,[],options);
    %计算映射样本中心化后的K
    n = size(fea_a,1);
    En = (1/n)*ones(n,n);
    K = K-K*En-En*K+En*K*En;
    
    aa = diag(K);
        
    if issparse(aa)
        aa = full(aa);
    end
    
    D = bsxfun(@plus,aa,aa') - 2*K;
    D(D<0) = 0;
    if bSqrt
        D = sqrt(D);
    end
    D = max(D,D');
else
    fea = [fea_a;fea_b];
    n = size(fea,1);
    na = size(fea_a,1);
    nb = size(fea_b,1);
    K = constructKernel(fea,[],options);
    Kab = constructKernel(fea_a,fea_b,options);
    Ka = constructKernel(fea_a,fea,options);
    Kb = constructKernel(fea,fea_b,options);
    Kabcenter = Kab-(1/n)*Ka*ones(n,1)*ones(1,nb)-(1/n)*ones(na,1)*ones(1,n)*Kb+(1/n)*(1/n)*ones(na,1)*ones(1,n)*K*ones(n,1)*ones(1,nb);
    En = (1/n)*ones(n,n);
    Kcenter = K-K*En-En*K+En*K*En;
    kd = diag(Kcenter);
    aa = kd(1:na);
    bb = kd(na+1:end);

    if issparse(aa)
        aa = full(aa);
        bb = full(bb);
    end

    D = bsxfun(@plus,aa,bb') - 2*Kabcenter;
    D(D<0) = 0;
    if bSqrt
        D = sqrt(D);
    end
end

