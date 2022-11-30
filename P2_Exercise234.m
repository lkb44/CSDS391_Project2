%% CSDS 391: Project 2
% Leo Bao
% Due: November 29, 2022
clear; close all; clc; 

%% Exercise 2
% Loading in iris data 
t = readtable('irisdata.csv');
x = t{51:150,3}; % Petal length
y = t{51:150,4}; % Petal width 
group = t{51:150,5}; % Classes

% Global variables so DATA and CLASS don't need to be passed into every function 
data = [x,y];
% Creating a vector of target class values
% Value of 0 for the second class and value of 1 for the first class 
for i = 1:100
    if i < 51
        class(i) = 0;
    else 
        class(i) = 1;
    end
end

% Plotting the second and third iris classes 
figure(1)
gscatter(x,y,group)
xlabel('petal length (cm)')
ylabel('petal width (cm)')
title('Iris classes')
hold on

% Plotting the decision boundary with arbitrary weights 
w0 = -63;
w1 = 7;
w2 = 18;
w = [w0 w1 w2]; 
computeDecisionBoundary(w,'Decision Boundary') 

% Producing 3D plot of the neural network output over the input space 
% Loop to compute the sigmoid value for each data pattern 
for i = 1:100
    x1 = data(i,1);
    x2 = data(i,2);
    values(i,1) = sigmoid(w,x1,x2);
end

figure(2)
% Plots the individual data points and their sigmoid vales 
plot3(x,y,values,'r.','Markersize',10) 
hold on

% Plots the plane overtop of the data points
syms xx yy
ezsurf(1/(1+exp(-(w0 + w1*xx + w2*yy))),[3,7],[1,2.5]) 
xlabel('petal length (cm)')
ylabel('petal width (cm)')
zlabel('Sigmoid Value')
title('Output of Neural Network over Input Space')
grid on     

%% Exercise 3
% Computing the mean squared error for an accurate decision boundary with arbitrary weights
w = [w0 w1 w2];
[errorSm] = meanSQerror(data, w, class);

% Computing the mean squared error for inaccurate decision boundary with arbitrary weights 
w0 = -70;
w1 = 5;
w2 = 11;
w = [w0 w1 w2];
%computeDecisionBoundary(w,'Decision Boundary'); 
[errorLg] = meanSQerror(data, w, class);

% Computing the summed gradient 
%gradient = computeGradient(w);
%% Exercise 4
% Choosing random values for the weights
% Weights are scaled so that they aren't too far off to start
w0 = -70*rand();
w1 = 10*rand();
w2 = 18*rand();
w = [w0 w1 w2];

% Replotting classes to show new gradient descent boundaries 
figure(3)
gscatter(x,y,group)
xlabel('petal length (cm)')
ylabel('petal width (cm)')
title('Iris classes')
hold on

% Performing gradient descent on the weights 
gradientDescent(data, w, class)
%% Function to compute and plot the decision boundary for the non-linearity 
% Inputs are weight vector and desicion boundary label 
function computeDecisionBoundary(w,label)
w0 = w(1);
w1 = w(2);
w2 = w(3);
x1 = 0:10;
y1 = -(w1 ./ w2) * x1 - (w0 ./ w2);
plot(x1,y1, 'DisplayName',label);
xlabel('petal length (cm)');
ylabel('petal width (cm)');
title('Iris classes');
end

%% Function to compute neural network using sigmoid
% Inputs are weight vector, petal width and petal length of a pattern 
% Output is the sigmoid value 
function s = sigmoid(w,x1,x2)
w0 = w(1);
w1 = w(2);
w2 = w(3);

% Computing classification value for each pattern 
e = exp(-(w0 + w1*x1 + w2*x2));
s = 1/(1+e);
end


%% Function to compute mean squared error 
% Input is pattern vectors, parameters, and pattern class  
% Output is the mean-squared error 
function error = meanSQerror(data, w, class)
n = size(data,1);
w0 = w(1);
w1 = w(2);
w2 = w(3);

% Computing classification value for each data point
for i = 1:100
    x1 = data(i,1);
    x2 = data(i,2);
    s(i,1) = sigmoid(w,x1,x2);
end

% Computing mean squared error 
summation = 0;
for i = 1:n
    summation = summation + (s(i)-class(i))^2;
    error = (1/n)*summation;
end
end

%% Function to compute the gradient 
% Input is the weight vector 
% Output is a vector of gradients for each weight 
function gradient = computeGradient(data, w, class) 
n = size(data,1);
summation0 = 0;
summation1 = 0;
summation2 = 0;

% Loops to calculate summed gradient for each weight 
for i = 1:n
    x1 = data(i,1);
    x2 = data(i,2);
    s = sigmoid(w,x1,x2);
    grad0 = ((s - class(i))*(s)*(1-s))*1;  % w0
    summation0 = summation0 + grad0;
    grad1 = ((s - class(i))*(s)*(1-s))*x1; % w1
    summation1 = summation1 + grad1;
    grad2 = ((s - class(i))*(s)*(1-s))*x2; % w2
    summation2 = summation2 + grad2;
end
gradient = [(2/n)*summation0; (2/n)*summation1; (2/n)*summation2];
end


%% Function to compute gradient descent 
% Input is the weight vector 
function gradientDescent(data, w, class)
precision = 0.03; % Min value for the error 
err = 1;
error = 1;
ep = 0.1; % learning rate 
maxi = 500; % Max number of iterations so we don't get an infinite loop 
iterations(1) = 0;
i = 1;
currentW = w;
% Plot initial decision boundary 
figure(3) 
computeDecisionBoundary(currentW,'Initial Decision Boundary')
% Loops while error is less than the precision boundary and iterations are
% less than the maximum number of iterations allowed 
while err > precision && i < maxi
    prevW = currentW;
    % Compute gradients
    gradient = computeGradient(data, prevW, class);
    % Gradient descent for each weight 
    currentW(1) = currentW(1) - (ep * gradient(1))
    currentW(2) = currentW(2) - (ep * gradient(2))
    currentW(3) = currentW(3) - (ep * gradient(3))
    % Calculate mean-squared error
    [error(i)] = meanSQerror(data,currentW,class); 
    iterations(i) = i;
    err = error(i);
    i = i+1;
    
    % Plot the middle decision boundary 
    if i == 250
        figure(3)
        computeDecisionBoundary(currentW,'Middle Decision Boundary')
        hold on
    end
end

% Plot the final decision boundary 
figure(3) 
computeDecisionBoundary(currentW,'Final Decision Boundary')

% Plot the learning curve
figure(4)
plot(iterations,error,'b.-') 
title('Learning Curve')
xlabel('Iterations')
ylabel('Error')

end