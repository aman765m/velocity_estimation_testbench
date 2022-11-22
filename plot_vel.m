close all
clear all
velx_1 = csvread('velx_1.csv');
vely_1 = csvread('vely_1.csv');
figure
plot(velx_1,'r')
hold on
plot(vely_1,'b')
legend('x velocity', 'y velocity')
title('Raw data(without filtering) Square Trajectory')

%online butter filter
velx_1_butter = csvread('velx_1_butter.csv');
vely_1_butter = csvread('vely_1_butter.csv');
figure
plot(velx_1_butter,'r')
hold on
plot(vely_1_butter,'b')
legend('x velocity', 'y velocity')
title('Data with online filtering Square Trajectory')

%%offline butterworth
n = 4;
Wn = 0.07;
[b,a] = butter(n,Wn);
velx_1 = filter(b,a,velx_1);
vely_1 = filter(b,a,vely_1);
figure
plot(velx_1,'r')
hold on
plot(vely_1,'b')
legend('x velocity', 'y velocity')
title('Data with offline filtering Square Trajectory')



%diagonal trajectory (double frequency)
velx_diag = csvread('velx_diag.csv');
vely_diag = csvread('vely_diag.csv');
figure
plot(velx_diag,'r')
hold on
plot(vely_diag,'b')
legend('x velocity', 'y velocity')
title('Data with online filtering Diagonal Trajectory')

%%offline butterworth
n = 4;
Wn = 0.12;
[b,a] = butter(n,Wn);
velx_diag = filter(b,a,velx_diag);
vely_diag = filter(b,a,vely_diag);
figure
plot(velx_diag,'r')
hold on
plot(vely_diag,'b')
legend('x velocity', 'y velocity')
title('Data with offline filtering Diagonal Trajectory')

