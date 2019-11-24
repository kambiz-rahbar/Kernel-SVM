function [b, S] = trainBinKernelSVM(X, T, Kernel_Param, C, alphaTresholdScale, MaxIter)

N = length(T);

K = SVMKernel(X,X,Kernel_Param);

M = repmat(T,N,1).*repmat(T',1,N).*K;

O = -ones(N,1);

QP_options = optimset('Algorithm', 'interior-point-convex', 'MaxIter', MaxIter);
alpha = quadprog(M,O,[],[],T,0,zeros(N,1),C*ones(N,1),[],QP_options)';
alphaThreshold = mean(abs(alpha)) * alphaTresholdScale;
alpha(abs(alpha) <= alphaThreshold) = 0;

S.x = X(:, alpha>0 & alpha<C);
S.y = T(:, alpha>0 & alpha<C);
S.alpha = alpha(:, alpha>0 & alpha<C);

b = mean(S.y - sum(S.alpha .* S.y .* SVMKernel(S.x, S.x, Kernel_Param)));

