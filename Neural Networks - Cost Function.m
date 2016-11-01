function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
eye_matrix = eye(num_labels)
y_matrix = eye_matrix(y,:) %carry out vector indexing (y,:) If y is a vector, then each value in y is used as an index into the eye() matrix. The colon character causes all columns of the 'y'th row of the eye() matrix to be duplicated into the output matrix.
a1 = [ones(m, 1) X]; %include a column of 1's as the intercept column to the features matrix X and call this a1
z2 = [a1 * Theta1']; %z2 is a matrix of activation outputs computed through matrix multiplication - Theta1 is a matrix of weights (parameters) that governs the function mapping from the input layer to the hidden layer 
a2 = sigmoid(z2); %apply the sigmoid function to the activation outputs matrix - these activation values are the new features 'learned' by the neural network to compute the output of the hypothesis functions in the output layer
u = size(a2, 1); 
a2 = [ones(u, 1) a2]; %add a column of 1's to account for the intercept
z3 = [a2 * Theta2']; %compute the activation output of the output layer
a3 = sigmoid(z3); %apply the sigmoid function to compute the output of the hypothesis functions in the output layer
e = -y_matrix;
f = log(a3);
i= [e .* f]; %Use element-wise multiplication - this computes the first piece of the cost function formula -y * log(hθ(x(i))k
b = (1-y_matrix);
c = log(1-a3); %the log and sigmoid functions produce element-wise results
d = [b .* c]; %Use element-wise multiplication - this computes the second piece of the cost function formula (1-y) * log(1-hθ(x(i))k
u =(i-d);
v = sum(u); %use double sum to get the cost function as a scalar value
q = sum(v);
x = [(1/m)* q]; %scale the result by 1/m - m is the number of training examples
s1 = sum(sum(Theta1(:,2:end).^2)); %compute the regularization terms for Theta1 and Theta2 
s2 = sum(sum(Theta2(:,2:end).^2)); %sum over from the second column onwards - no need to regularize the intercept term
s = s1 + s2;
t = [(lambda)/(2*m)]; %scale the result by the expression lambda/2m
reg = (t * s);
J = [x + reg]; %J is a variable that denotes the regularized cost function
d3 = [a3 - y_matrix]; %compute the error terms for the output layer
d2 = [d3 * Theta2(:,2:end)] .* sigmoidGradient(z2); %compute the matrix of partial derivatives for the hidden layer - d2 is a matrix that answers the question 'how much were the hidden units of the hidden layer, responsible for misclassification (errors) of the output layer?'
delta1 = [d2' * a1]; %This piece of code pertains to the matrix of partial derivatives for the hidden layer for individual training examples
delta2 = [d3' * a2];%This piece of code pertains to the matrix of error terms for the output layer for individual training examples
k = (1/m); %scale results by 1/m to get two gradients - one with respect to Theta1 and the other with respect to Theta2
Theta1_grad = (k * delta1);
Theta2_grad = (k * delta2);
c = (lambda/m); 
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + c * Theta1(:,2:end); %multiply the gradients by the regulariation term lambda/m - only take into account all columns from column 2 onwards as the intercept term is not regularized
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + c * Theta2(:,2:end); %The result is the accumulated regularized gradient for the cost function J, with respect to Theta1 and Theta2 (excluding the 1st column meant for the intercept term)
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];
end
