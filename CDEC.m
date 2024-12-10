% T为基聚类器的数量，K为原始数据集类别数
function label_final = CDEC(T, K, attrib, lambda, gamma)
% 样本数量
n = size(attrib, 1);
% 初始化alpha,(T,1)
Alpha = ones(T, 1) / T;
% 基聚类器的簇数
m = randi([2, max(50, floor(sqrt(n)))], T, 1);
baseCluster = cell(1, T);
parfor i = 1:T
    label_now = kmeans(attrib, m(i, 1));
    resultMat = zeros(n, m(i, 1));
    idx = sub2ind([n, m(i, 1)], (1:n)', label_now);
    resultMat(idx) = 1;
    baseCluster{i} = sparse(resultMat);
end

YY = cell(T, 1);
parfor i = 1:T
    YY_inv = baseCluster{i}' * baseCluster{i};
    YY{i} = baseCluster{i} / YY_inv * baseCluster{i}';
end

% 最大迭代次数
maxIter = 50;
% 已迭代次数
times = 0;
% 目标函数当前值
nowVal = inf;
% 记录目标函数值
lossValues = [];
IndMat = zeros(n, K);
% K-means初始化
now = kmeans(attrib, K);
idx = sub2ind([n, K], (1:n)', now);
IndMat(idx) = 1;

while(times < maxIter)
    % 计算IndMat
    [IndMat, loss] = CoordinateDescent(Alpha, YY, K, attrib, lambda, IndMat);

    HHMat = zeros(n, n);
    parfor i = 1:T
        HHMat = HHMat + Alpha(i, 1)*YY{i};
    end
    
    sumVal = trace(HHMat * HHMat) - 2*loss(1, size(loss, 2)) + K + gamma*(Alpha'*Alpha);
    lossValues = [lossValues, sumVal];
    if abs(nowVal - sumVal) < 1e-5
        break;
    end
    nowVal = sumVal;
    times = times + 1;
    
    % 计算Alpha
    Alpha = get_Alpha(IndMat, T, YY, gamma);
end
label_final = zeros(n, 1);
for i = 1:n
    label_final(i) = find(IndMat(i, :) ~= 0);
end
end

%% 固定IndMat，求Alpha
function Alpha = get_Alpha(IndMat, T, YY, gamma)
%% 矩阵A
A = zeros(T, T);
% 预先计算YY，避免重复计算
for i = 1:T
    for j = i:T
        A(i, j) = trace(YY{i} * YY{j});
        A(j, i) = A(i, j);
    end
end
A = A + gamma*eye(T, T);
% 矩阵B
B = zeros(T, 1);
FF = pinv(IndMat' * IndMat);
FFn = IndMat * FF * IndMat';
parfor i = 1:T
    B(i, 1) = trace(YY{i} * FFn);
end
% 凸二次规划
Aeq = ones(T, 1)';
lb = zeros(T, 1);
% 设置二次规划最大迭代次数
Alpha = quadprog(2*A, -2*B, [], [], Aeq, 1, lb);
end