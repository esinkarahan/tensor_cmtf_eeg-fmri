function x = posrandn(varargin)
% Generate normal nonnegative random arrays with mean 0 and std 1
% INPUTS:
% varargin is for the dimensions of the array
% OUTPUT:
% x : array of nonnegative random numbers
% 
% Usage: x = posrandn(I1,I2)
%
% Version 1 - May 2015 

I=[];
for i = 1:length(varargin)
    I = [I varargin{i}];
end
x = randn(I);
k = find(x<0);
for i = 1:length(k)
     while(x(k(i))<0)
       x(k(i)) = randn(1);
    end
end
