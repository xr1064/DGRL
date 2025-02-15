% Method for nearest centroid classifier
function [retClasses]= classifyNC(manip_test, manip_train, gnd_Train)
    retClasses = zeros(1,size(manip_test,2));
    faceClassId = unique(gnd_Train);
    classCentroid = zeros(size(manip_train,1), size(faceClassId,2));
    
    for i=faceClassId % uniq gnd_Train
        manip_group = manip_train(:, gnd_Train==i);
        classCentroid(:,i) = mean(manip_group, 2); % i'th face class centroid
    end
    calcedDist = calcDistanceNC(manip_test, classCentroid, faceClassId);
    
    % take minimum distance for the testset
    for i=1:size(manip_test,2)
       [~, gndClass]= min(calcedDist(i,:));
       retClasses(i) = gndClass;
    end
end

% ED between two generated sets for NC 
function ret = calcDistanceNC(manip_test, classCentroid, faceClassId)
    calcDist = zeros(size(manip_test,2),15);
    for i=1:size(manip_test,2)
        for j=faceClassId % j is label of person
            calcDist(i,j) = norm(manip_test(:,i) - classCentroid(:,j));
        end
    end
    ret = calcDist;
end