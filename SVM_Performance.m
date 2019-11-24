function SVM_Performance(T,Y)

figure;
subplot(3,1,1);
plot(T,'ob');
hold on;
plot(Y,'xR');
grid minor;
legend('Targets','Network outputs');
title('Targets data vs Network outputs');
xlabel('sample number');
ylabel('data');

errorAbsValue = T-Y;
errorMean = mean(errorAbsValue);
errorStd = std(errorAbsValue);

subplot(3,1,2);
plot(errorAbsValue,'b');
grid minor;
title(sprintf('RMSE: %.4f',rms(errorAbsValue)));
xlabel('sample number');
ylabel('absolute error');

subplot(3,1,3);
histfit(errorAbsValue,50);
title(sprintf('mean: %.4f , std: %.4f',errorMean,errorStd));
grid minor;
xlabel('Frequency');
ylabel('error');
