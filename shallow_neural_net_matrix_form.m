N = 50;
X = linspace(-2*pi, 3, N)';
Y = exp(X) + randn(size(X))*.5;
plot(X, Y, 'o')

%%
clf

K = 15;
netparams = [];
netparams.w1 = randn( K, size(X, 2) );
netparams.b1 = zeros( K, 1 );
netparams.w2 = randn( K, 1 );
netparams.b2 = zeros;


[netparams, L] = fitnet( X, Y, netparams, @ftanh, 50, .05 );

subplot(211)
plot(L)

subplot(212)
hold on 
plot( X, Y, 'o' );

[~, ~, Yh, ~] = netforward(  X, netparams, @ftanh );
plot(X,Yh)
%%
function [X, dX] = frelu(X)
X(X<=0) = 0;
dX = zeros( size(X) );
dX(X>0) = 1;
end

function [X, dX] = ftanh(X)
X = tanh(X);
dX = sech( X ).^2;
end

function [a1, z1, a2, dz1_da1] = netforward( X, netparams, g )
a1 = X * netparams.w1' + netparams.b1';
[z1, dz1_da1] = g( a1 );
a2 = z1 * netparams.w2 + netparams.b2;
end

function [err, dErr] = E( Y, Yh )
err = mean( Y - Yh ).^2;
dErr = -2*( Y - Yh )/size(Y, 1);
end

function [netparams, losses] = fitnet( X, Y, netparams, g, epochs, lr )

losses = zeros(epochs, 1);
for i=1:epochs
[a1, z1, a2, dz1_da1] = netforward( X, netparams, g );
[err, dE_da2] = E( Y, a2 );
losses(i) = err;
dE_dw2 = z1' * dE_da2;
dE_db2 = sum( dE_da2, "all" );
dE_dw1 = ((dE_da2 .* dz1_da1)' .* netparams.w2) * X;
dE_db1 = sum((dE_da2 .* dz1_da1) .* netparams.w2', 1)';

netparams.b1 = netparams.b1 - lr * dE_db1;
netparams.w1 = netparams.w1 - lr * dE_dw1;
netparams.b2 = netparams.b2 - lr * dE_db2;
netparams.w2 = netparams.w2 - lr * dE_dw2;
end

end





