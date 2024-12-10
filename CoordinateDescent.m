% Alpha为基聚类器的权重，simiMat为基聚类器的相似度矩阵，K为类别数
function [IndMat, loss] = CoordinateDescent(Alpha, YY, K, X, lambda, IndMat)
% 样本个数
n = size(X, 1);

% 初始化相似度矩阵mats
mats = zeros(n, n);

% 基聚类器个数
T = length(Alpha);

% 得到相似度矩阵mats
for i = 1:T
    mats = mats + Alpha(i, 1) * YY{i};
end

% matA维护yAy,matD维护yy
FF = IndMat' * IndMat;
matD = diag(FF);
FHF = IndMat' * mats * IndMat;
matA = diag(FHF);

% flag指示是否收敛，（两次迭代IndMat不变或者目标函数值变化不大）
flag = true;
loss = [];
while(flag)
    flag = false;
    [~,indexFlag]=max(IndMat,[],2);
    for i = 1:n
        % m为yi中1的位置
        m = indexFlag(i);
        % 判断是否为空类
        if matD(m) == 1
            continue
        end
        % 记录最大值及对应位置
        maxVal = -inf;
        p = 0;
        for j = 1:K
            if j == m
                first = matA(j) / matD(j);
                yk = IndMat(:, j);
                second = (matA(j) - 2*(yk' * mats(:, i)) + mats(i, i)) / (matD(j) - 1);
                nowVal = first - second + lambda*(1 - 2*matD(j))/2;
            else
                yk = IndMat(:, j);
                first = (matA(j) + 2*(yk' * mats(:, i)) + mats(i, i)) / (matD(j) + 1);
                second = matA(j) / matD(j);
                nowVal = first - second - lambda*(1 + 2*matD(j))/2;
            end
            % 最大值发生变化时，更新maxVal和p
            if nowVal > maxVal
                maxVal = nowVal;
                p = j;
            end
        end
        % 如果P和m不相等，更新IndMat和matA、matD
        if p ~= m
            flag = true;
            ym = IndMat(:, m);
            yp = IndMat(:, p);
            matA(m) = matA(m) - 2*(ym' * mats(:, i)) + mats(i, i);
            matD(m) = matD(m) - 1;
            matA(p) = matA(p) + 2*(yp' * mats(:, i)) + mats(i, i);
            matD(p) = matD(p) + 1;
            IndMat(i, m) = 0;
            IndMat(i, p) = 1;
        end
    end
    % 计算目标函数值
    newVal = 0;
    for i = 1:K
        newVal = newVal + matA(i) / matD(i) - (lambda*matD(i)^2)/2;
    end
    loss = [loss, newVal];
end