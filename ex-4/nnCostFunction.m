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


% Part 1: Feedforwarding neural network to find the cost in variable J

% adding column of 1's to X as bias units. 
X = [ones(m, 1) X];

%Preparing y_matrix
all_combos = eye(num_labels);    
y_matrix = all_combos(y,:);

%Finding h(x) (output value)
z2 = X * Theta1';
a2 = sigmoid(z2);

a2 = [ones((size(a2, 1)), 1) a2];

z3 = a2 * Theta2';
a3 = sigmoid(z3);

J = trace((-1/m) * (y_matrix' * (log(a3)) + ((1 - y_matrix)' * log(1-a3))));



reg = (lambda / (2*m)) * (sum(sum(Theta1(:, 2:end) .^ 2)) + sum(sum(Theta2(:, 2:end).^ 2)));

J = J + reg

% -------------------------------------------------------------
% Part 2: Computing gradients: for loop method

%initializing Delta accumulators
Delta1 = 0;
Delta2 = 0;

%resetting X (removing bias column from part 1)
X = X(:, 2:end);

for t = 1:m
    yk = y_matrix(t, :);
    %layer 1
    a_1 = X(t, :);
    a_1 = [1, a_1];
    %layer 2
    z_2 = a_1 * Theta1'; 
    a_2 = sigmoid(z_2);
    a_2 = [1, a_2];
    %layer 3
    z_3 = a_2 * Theta2';
    a_3 = sigmoid(z_3);
    %delta error
    d_3 = a_3 - yk;
    %sigmoid gradient of z2
    sig_grad_z2 = sigmoidGradient(z_2);
    %delta2 error
    d_2 = (d_3 * Theta2(:, 2:end)) .* sig_grad_z2;
    Delta1 = Delta1 + (d_2' * a_1);
    Delta2 = Delta2 + (d_3' * a_2);

end

Theta1_grad = (1/m) * Delta1;
Theta2_grad = (1/m) * Delta2;

% =========================================================================
% Part 3: Implement regularization with the cost function and gradients.

Theta1(:, 1) = 0;
Theta2(:, 1) = 0;

Theta1_scaled = Theta1 * (lambda / m);
Theta2_scaled = Theta2 * (lambda / m);

Theta1_grad = Theta1_grad + Theta1_scaled;
Theta2_grad = Theta2_grad + Theta2_scaled;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)]


end
