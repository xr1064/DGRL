function L = calDifLaplacian(W, type)

D = full(sum(W,2));
sizeW = length(D);



if strcmp(type,'StandardNorm')     
    D=spdiags(D,0,sizeW,sizeW);
    L = D - W;
elseif strcmp(type,'RandomWalkNorm')       
    D=1./D;
    D(D==inf)=0;
    D=spdiags(D,0,sizeW,sizeW);
    W=D*W;
    L=speye(size(W,1))-W;  
elseif strcmp(type,'LaplacianNorm')  
    D=sqrt(1./D);
    D(D==inf)=0;  
    D=spdiags(D,0,sizeW,sizeW);
    W=D*W*D;
    L=speye(size(W,1))-W;
elseif strcmp(type,'NJW')
    D=spdiags(D,0,sizeW,sizeW);
    L = (D^-1/2) * W * (D^-1/2);
elseif strcmp(type,'MS')
    D=spdiags(D,0,sizeW,sizeW);
    L = (D^-1) * W;
end

end