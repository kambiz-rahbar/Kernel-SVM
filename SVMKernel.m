function [K] = SVMKernel(Xi,Xj,Kernel_Param)

N = size(Xi,2);
M = size(Xj,2);

if strcmp(Kernel_Param.type, 'Gaussian')
    sigma = Kernel_Param.sigma;
    K = zeros(N,M);
    for i = 1:N
        for j = 1:M
            K(i,j) = exp( norm(Xi(:,i)-Xj(:,j)).^2 / (-2*sigma^2) );
        end
    end
end