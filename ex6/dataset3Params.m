function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
C_list = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
s_list = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

% create a blank results matrix
results = zeros(length(C_list) * length(s_list), 3);  

row = 1;
for c = 1:length(C_list)
    C_val = C_list(c)
    for s = 1:length(s_list)
        sigma_val = s_list(s)
        model = svmTrain(X, y, C_val, @(x1, x2)gaussianKernel(x1, x2, sigma_val));
        predictions = svmPredict(model, Xval);
        err_value = mean(double(predictions ~= yval));
        results(row,:) = [err_value, C_val, sigma_val];
        row = row + 1;
    end

[r c] = min(results(:, 1));

C = results(c, 2);
sigma = results(c, 3);

% =========================================================================

end
