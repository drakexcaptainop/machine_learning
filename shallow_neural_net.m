N = 100;
X = linspace(-4, 4, N)';
L = 5;

Y = exp(X)  +randn(size(X)) * 2;

netparams = [];
netparams.W2 = randn( size(X, 2), L ) * 0.01;
netparams.b2 = zeros( 1, L );
netparams.W1 = randn(L, 1) * 0.01;
netparams.b1 = zeros;

clc
[netparams, L] = trainsnet( netparams, X, Y, @ftanh, @tanh_derivative, 350, .005);


[~,~,a2] = netforward( netparams, X, @ftanh );
subplot(121)
plot(X, a2)
hold on 
plot(X, Y, 'o')
subplot(122)
plot(L)


%%
function z = ftanh( a )
    z = tanh(a);
end

function dz = tanh_derivative(a)

dz = 1 - tanh(a).^2;
end

function err = E(y, a, N)

err = (y-a)^2/N;
end

function dE = derr(y, a, N)

dE = -2*(y-a)/N;
end


function [a1, z1, a2] = netforward(netparams, X, F)

a1 = X * netparams.W2 + netparams.b2;
z1 = F(a1);
a2 = z1 * netparams.W1 + netparams.b1;

end

function [netparams, L] = trainsnet( netparams, X, Y, F, dF, epochs, lr )

L = zeros( epochs, 1 );
N = size(X, 1);
for i=1:epochs
    cl = 0;
    dE_dW1c = zeros( size(netparams.W1) );
    dE_dW2c = zeros( size(netparams.W2) );
    dE_db1c = zeros( size(netparams.b1) );
    dE_db2c = zeros( size(netparams.b2) );

    for j = 1:size(X, 1)
        
        [a1, z1, a2] = netforward( netparams, X(j, :), F );
        Ei = E( Y(j), a2, N );
        cl = Ei + cl;
        dE_da2 = derr( Y(j), a2, N );
        dE_dW2c = dE_dW2c + dE_da2 * (diag( dF(a1) ) * netparams.W1 * X(j, :))';
        dE_db2c = dE_db2c + (dE_da2 * diag( dF(a1) ) * netparams.W1)';
        dE_db1c = dE_db1c + dE_da2;
        dE_dW1c = dE_dW1c + dE_da2 * z1';
    end
    L(i) = cl;

    netparams.W2 = netparams.W2 - lr * dE_dW2c;
    netparams.W1 = netparams.W1 - lr * dE_dW1c;
    netparams.b2 = netparams.b2 - lr * dE_db2c;
    netparams.b1 = netparams.b1 - lr * dE_db1c;
end

end











