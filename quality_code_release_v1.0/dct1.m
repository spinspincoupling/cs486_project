function [X A] = dct1(x, k_max) 
%%% X is the first k_max coefficients of the DCT of x
%%% A is the DCT transformation matrix
N = size(x, 1);

if ~exist('k_max')
  k_max = N;
else
  k_max = min(k_max, N);
end
 
A = cos(pi/N*[0:k_max-1]'*([0:N-1]+.5));
X = A*x;

