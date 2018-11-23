function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

m = length(y);
for iter = 1:num_iters

    t1adjust = alpha * sum( (X * theta  - y ) .* X(:,1) ) / m;
    t2adjust = alpha * sum( (X * theta  - y ) .* X(:,2) ) / m; 

    theta(1) = theta(1) - t1adjust;    
    theta(2) = theta(2) - t2adjust;

    % Save and print the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    fprintf ('Cost function in iteration %i: ', iter );
    fprintf ('%f\n', J_history(iter) )
end

end
