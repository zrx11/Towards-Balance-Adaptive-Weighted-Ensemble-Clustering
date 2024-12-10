clear;
clc;

addpath('data');
addpath('Measures');
filelist = dir('data\*.mat');
len = length(filelist);

clusterNum = 20;
lambda = [10^-4, 10^-3, 10^-2, 10^-1];
gamma = [10^-3, 10^-2, 10^-1, 1];

for i = 1:len
    file = filelist(i).name;
    filename = file(1:length(file)-4);
    
    NMI = zeros(length(lambda)*length(gamma), 2);
    ACC = zeros(length(lambda)*length(gamma), 2);
    ARI = zeros(length(lambda)*length(gamma), 2);
    Nen = zeros(length(lambda)*length(gamma), 2);
    
    for j = 1:length(lambda)
        for k = 1:length(gamma)
            [bst, str] = CDEC_demo(file, clusterNum, lambda(j), gamma(k));
            NMI((j-1)*length(gamma)+k, 1) = str(1);
            NMI((j-1)*length(gamma)+k, 2) = str(2);
            ACC((j-1)*length(gamma)+k, 1) = str(3);
            ACC((j-1)*length(gamma)+k, 2) = str(4);
            ARI((j-1)*length(gamma)+k, 1) = str(5);
            ARI((j-1)*length(gamma)+k, 2) = str(6);
            Nen((j-1)*length(gamma)+k, 1) = str(7);
            Nen((j-1)*length(gamma)+k, 2) = str(8);
        end
    end
    [~, p] = max(ACC(:, 1));
    outcome = [NMI(p, 1), NMI(p, 2); ACC(p, 1), ACC(p, 2); ARI(p, 1), ARI(p, 2); Nen(p, 1), Nen(p, 2)];
    save(['outs\', filename, '.mat'], 'NMI', 'ACC', 'ARI', 'Nen', 'outcome', 'bst');
end
