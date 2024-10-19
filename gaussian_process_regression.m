N = 100;

x = linspace(-2*pi, 2*pi, N)';
t = sin(x) + randn(size(x))*.2;

X = [x, ones(size(x))];

alpha = [1, 1];
k = @(x1, x2) alpha(1) * exp(-alpha(2) * (x1-x2)'*(x1-x2));

K = kernelmat( X, k );

q = .1;

C = q * eye( N ) + K;

plot(x, t, 'o')
%%

xp = linspace( min(x)-2, max(x) + 2 )';
[Th, Vh, MSV] = predict( [xp, ones(size(xp))], X, t, C, k, true );

hold on


plot(xp, Th)
plot(xp, MSV, 'r--')
plot(x, t, 'ok')

%%

function [Th, Vh, MSV] = predict( Xh, X, T, C, K, includeMSV )

MSV = [];


nh = size(Xh, 1);
Th = zeros(nh, 1);
Vh = zeros(size(Th));
Cinv = inv(C);

if includeMSV
MSV = zeros(size(Vh, 1), 2);
end


for i=1:nh
    k = vecrowkernel( Xh(i, :)', X, K )';
    Th(i) = k' * Cinv * T;
    Vh(i) = K(Xh(i, :)', Xh(i, :)') - k' * Cinv * k;
    MSV(i, :) = [ Th(i) + Vh(i), Th(i) - Vh(i)];
end

end

function R = vecrowkernel(x, X, k)
N=size(X, 1);
R = zeros(N,  1);
for i=1:N

    R(i) = k(x, X(i, :)');
end
R = R';
end

function K = kernelmat(X, k)

N = size(X, 1);
K = zeros(N);
for i = 1:N

    K(i, :) = vecrowkernel( X(i, :)', X, k );
end


end






