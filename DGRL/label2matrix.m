function label_mat = label2matrix(label)
    uq_label = unique(label);
    n = length(label);
    c = length(uq_label);
    label_mat = zeros(n,c);
    for i = 1:c
        index = label == i;
        label_mat(index,i) = 1;
    end
end
