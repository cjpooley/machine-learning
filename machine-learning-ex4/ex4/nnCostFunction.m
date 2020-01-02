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

J_sum = 0;
for i = 1:m
  h_i = sigmoid(Theta2*[1; sigmoid(Theta1*[1, X(i,:)]')]);
  y_i = zeros(num_labels,1);
  y_i(y(i)) = 1;
  J_sum = J_sum + ((-y_i'*log(h_i)) - ((1-y_i)'*log(1-h_i)));
endfor
J = J_sum/m;
Theta1_nobias = Theta1(:,2:end);
Theta2_nobias = Theta2(:,2:end);
reg = (lambda/(2*m))*((sum(Theta1_nobias(:).^2)) + (sum(Theta2_nobias(:).^2)));
J = J + reg;

b = (1:num_labels)';
D_2 = zeros(num_labels, hidden_layer_size + 1);
D_1 = zeros(hidden_layer_size, size(X, 2) + 1);
for t = 1:m
  a_1 = [1; X(t, :)'];
  a_2 = [1; sigmoid(Theta1*a_1)];
  a_3 = sigmoid(Theta2*a_2);
  d_3 = a_3 - (b == y(t));
  d_2 = (Theta2'*d_3).*([1; sigmoidGradient(Theta1*a_1)]); %   26x1.*26x1
  D_2 = D_2 + d_3*a_2'; % 10x1*1x26 -> 10x26
  D_1 = D_1 + d_2(2:end)*a_1'; % 25x1*1x401 -> 25x401
endfor
Theta1_reg = [zeros(size(Theta1,1),1), (lambda/m)*Theta1_nobias];
Theta2_reg = [zeros(size(Theta2,1),1), (lambda/m)*Theta2_nobias];
Theta1_grad = D_1/m + Theta1_reg;
Theta2_grad = D_2/m + Theta2_reg;


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
