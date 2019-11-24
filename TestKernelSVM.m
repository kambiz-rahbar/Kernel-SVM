clc;
clear;
close all;
warning off

X = 2*rand(2,15)-1;
T = -ones(size(X,2),1)';
X = [X X+2];
T = [T -T];

% set SVM parameters
C = 10;
MaxIter = 100;
alphaTresholdScale = 10^-5;

% train binary SVM
Kernel_Param.type = 'Gaussian';
Kernel_Param.sigma = 1.5;
[b, SupVec] = trainBinKernelSVM(X, T, Kernel_Param, C, alphaTresholdScale, MaxIter);

% test binary SVM
Xtest = X;
Y = BinKernelSVMClassify(Xtest, SupVec, b, Kernel_Param);

% check performance
SVM_Performance(T,Y)

% Plot Results
figure(2);
hold on;
plot(X(1,T==-1),X(2,T==-1),'ko','linewidth',2);
plot(X(1,T==1),X(2,T==1),'kx','linewidth',2);

[Xplot,Yplot] = meshgrid(min(X(1, :)):0.005:max(X(1, :)), min(X(2, :)):0.005:max(X(2, :)));
Xplot = reshape(Xplot,[],1);
Yplot = reshape(Yplot,[],1);
plotData = [Xplot Yplot]';
plotIdxp1 = abs(sum(SupVec.alpha .* SupVec.y .* SVMKernel(plotData, SupVec.x ,Kernel_Param),2)+b+1)<10^-3;
plotIdx0 = abs(sum(SupVec.alpha .* SupVec.y .* SVMKernel(plotData, SupVec.x ,Kernel_Param),2)+b)<10^-3;
plotIdxm1 = abs(sum(SupVec.alpha .* SupVec.y .* SVMKernel(plotData, SupVec.x ,Kernel_Param),2)+b-1)<10^-3;
plot(Xplot(plotIdxp1),Yplot(plotIdxp1),'.g','linewidth',2)
plot(Xplot(plotIdx0),Yplot(plotIdx0),'.r','linewidth',2)
plot(Xplot(plotIdxm1),Yplot(plotIdxm1),'.b','linewidth',2)

plot(SupVec.x(1,:), SupVec.x(2,:), 'ob','MarkerSize',15,'linewidth',2);

xlabel('X1');
ylabel('X2');
legend('class #1', 'class #2',...
    'pos margin', 'classification curve', 'neg margin',...
    sprintf('SupVec:(%d)',size(SupVec,2)));

