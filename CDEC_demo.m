function [bst, str] = CDEC_demo(file, T, lambda, gamma)
datas = load(file);
attrib = datas.X;
%归一化
newAttrib = mapminmax(attrib', 0, 1);
attrib = newAttrib';
label = datas.Y;
K = length(unique(label));
cntTimes = 10;
NMI_oig = zeros(cntTimes, 1);
NMI_str = zeros(cntTimes, 1);
Acc_oig = zeros(cntTimes, 1);
Acc_str = zeros(cntTimes, 1);
ARI_oig = zeros(cntTimes, 1);
ARI_str = zeros(cntTimes, 1);
Nen_oig = zeros(cntTimes, 1);
Nen_str = zeros(cntTimes, 1);

% 10次运行
parfor i = 1:cntTimes
    label_pred = kmeans(attrib, K);
    result = ClusteringMeasure(label, label_pred);
    NMI_oig(i) = result(2);
    Acc_oig(i) = cluster_acc(label, label_pred);
    ARI_oig(i) = RandIndex(label, label_pred);
    Nen_oig(i) = Nentro(label_pred);
end

parfor i = 1:cntTimes
    label_final = CDEC(T, K, attrib, lambda, gamma);
    result = ClusteringMeasure(label, label_final);
    NMI_str(i) = result(2);
    Acc_str(i) = cluster_acc(label, label_final);
    ARI_str(i) = RandIndex(label, label_final);
    Nen_str(i) = Nentro(label_final);
end

[~, p] = max(Acc_oig);
bst = [NMI_oig(p); 0; Acc_oig(p); 0; ARI_oig(p); 0; Nen_oig(p); 0];
str = [mean(NMI_str); std(NMI_str); mean(Acc_str); std(Acc_str); mean(ARI_str); std(ARI_str); mean(Nen_str); std(Nen_str)];
end