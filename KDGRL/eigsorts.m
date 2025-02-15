function [Vsort,Dsort] = eigsorts(V,D,MODE)   
d = diag(D);
[Ds,D_index] = sort(d,MODE);  
Dsort = diag(Ds);
Vsort = V(:,D_index);


