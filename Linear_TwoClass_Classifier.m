clear
clc
% Load Training data
Mat = csvread('Otonabee_Fluoranthene_training.csv');

% Convert the first column to two class labels 0 & 1(Fluoranthene concentration < or ≥ 0.5)
y = double(logical(Mat(:,1)'>= 0.5));

% Preprocess Training Data
%... Find the average of 111 emission wavelengths caused by each excitation wavelength of each sample
%... Mean Matrix has 29 rows(no. of features) and 336 columns(no. of samples) 
i = 1;
while i <30
Mean(i,:) = mean(Mat(:,(2+111*(i-1)):(1+111*i)),2)'; 
i = i+1;
end
x = Mean;

% Initialize w 
w0 = [4;0;1;-2;-1;-5;2;-4;-3;1;2;-5;6;3;7;-1;8;-2;4;-1;3;-7;2;4;-5;3;-6;-2;2;1]; 
f = @(w)cross_entropy(w,x,y);

%%% Train classifier %%%
[W,fW] = gradient_descent(f,w0,0.01,10000);
figure
plot(fW,'linewidth',1)
axis([-200 11000 0 33])
title('Cost Function Curve in Training')
xlabel('Iteration');ylabel('Cost Value')

%%% Test classifier %%%
% Load test data
Mat1 = csvread('Otonabee_Fluoranthene_test.csv');
% Preprocess test Data
i = 1;
while i <30
Mean1(i,:) = mean(Mat1(:,(2+111*(i-1)):(1+111*i)),2)'; 
i = i+1;
end
x = Mean1;
% actual labels
y_actual = double(logical(Mat1(:,1)'>= 0.5));
P = length(y_actual);
xbar = [ones(1,P); x];
Sig = sigmoid(xbar'*W(:,end));
y_predicted = double(logical(Sig(:,1)'>= 0.5));
% Number of errors
Errors = sum(logical(y_actual~=y_predicted))
% Accuracy of classifier
Accuracy = sum(logical(y_actual==y_predicted))/P
% Confusion Matrix
a = sum((y_actual==1)&(y_predicted==1));
b = sum((y_actual==1)&(y_predicted==0));
c = sum((y_actual==0)&(y_predicted==1));
d = sum((y_actual==0)&(y_predicted==0));
A_balanced = (a/(a+c) + d/(b+d))/2
CM = [a b;c d];
figure
imagesc(CM) % Display image with scaled colors
colorbar
title('Confusion Matrix')
set(gca, 'XTick', [1 2]) 
set(gca,'XTickLabel',{'1','0'}) 
set(gca, 'YTick', [1 2]) 
set(gca,'YTickLabel',{'1','0'}) 
ylabel('Actual Label')
xlabel('Predicted Label')
text(0.75,1,'68 (95.8%)','FontSize',14)
text(1.75,1,'3 (4.2%)','FontSize',14)
text(0.75,2,'3 (2.3%)','FontSize',14)
text(1.75,2,'126 (97.7%)','FontSize',14)
annotation('textbox',[.01 0 .25 .1],'String','A balanced: 96.72%  A: 97% ','EdgeColor','none')

function cost = cross_entropy(w,x,y)
P = length(y);
xbar = [ones(1,P); x];
SIG = sigmoid(xbar'*w);
cost = (1/P) * ...
    sum( ...
    - repmat(y',1,size(w,2)) .* log(SIG)...
    - (1-repmat(y',1,size(w,2))) .* log(1-SIG));
end

function s = sigmoid(x)
s = 1./(1+exp(-x));
s(s==1) = .9999;
s(s==0) = .0001;
end

function [W,fW] = gradient_descent(f,w0,alpha,n_iter)
k = 1;
W = w0;
fW = f(w0);
while k < n_iter
    grad = approx_grad(f,W(:,k),0.0001);
    W(:,k+1) = W(:,k) - alpha*(grad')/norm(grad);
    fW(k+1) = f(W(:,k+1));
    k = k+1;
end
end

function grad = approx_grad(f,w0,delta)
N = length(w0);
dw = delta*eye(N);
grad = ( f(w0+dw) - f(w0) )/delta;
end