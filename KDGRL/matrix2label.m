function label_vec = matrix2label(matrix)
   [n,c] = size(matrix);
   label_vec = zeros(1,n);
   for i = 1:c
       label_vec(matrix(:,i)==1) = i;
   end
end