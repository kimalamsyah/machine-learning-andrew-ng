function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

%computing hypothesis and error
h = X * theta;
error = h - y;

%unregularized J
J = (1/ (2*m)) * (sum(error.^2));

%regularized J
theta(1) = 0;
reg = (lambda / (2 * m)) * sum(theta.^2);
J = J + reg

% =========================================================================
%unregularized gradient descent
grad = (X' * error) * (1/m);

%regularized gradient descent
theta(1) = 0;
grad_reg = theta * (lambda / m);
grad = grad + grad_reg;
grad = grad(:);

end
