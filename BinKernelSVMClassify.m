function [Y] = BinKernelSVMClassify(Xtest, S, b, Kernel_Param)

Y = sign(sum(S.alpha .* S.y .* SVMKernel(Xtest, S.x ,Kernel_Param),2)+b);
Y = Y';

