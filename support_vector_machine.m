r = [3, 6];

N = 100;
x = linspace(0, 2*pi, N)';

s = .5;

X1 = [ cos(x) * r(1) + randn(size(x)) * s, sin(x) * r(1) + randn(size(x)) * s ];
X2 = [ cos(x) * r(2) + randn(size(x)) * s, sin(x) * r(2) + randn(size(x)) * s ];



y = ones(size(x, 1)*2, 1);
y(1:N) = -1;


X = [X1; X2];
%%
alpha = [1, 1];
k = @(x1, x2) alpha(1) * exp(-alpha(2) * (x1-x2)'*(x1-x2));

Y = diag(y);

K = kernelmat(X, k);

lambda = quadprog(K, -ones( size(y) ),[], [],y',0, zeros(size(y)) );

I = find(lambda > 1e-10);

b = 0;
for i=1:size(I, 1)
    is = 0;
    for j=1:size(I, 1)
        is = is + lambda(j) * y(j) * K(i, j);
    end
    b = b + y(i) - is ;
end
b = b / size(I, 1);


Xi = X(I, :);
Yi = y(I, :);

%%
hold on
plot( X1(:, 1), X1(:, 2), 'ro' )
plot( X2(:, 1), X2(:, 2), 'ko' )

xl = xlim;
yl = ylim;

No = 100;
[Xo, Yo ] = meshgrid( linspace( xl(1), xl(2), No ), linspace( yl(1), yl(2) ), No );
Xdo = [reshape(Xo, No*No, 1), reshape(Yo, No*No, 1)];

T = predict( Xdo, Xi, Yi, b, lambda(I), k );


Zo = reshape(T, size(Xo));

contourf( Xo, Yo, Zo, [xl(1), 0, xl(2)],FaceAlpha=.2 )








%%

function T = predict(Xh, Xi, Yi, b, lambda, k)

T = zeros(size(Xh, 1) ,1);
yT = diag(Yi) * lambda;

for i = 1:numel(T)
    kr = vecrowkernel(Xh(i, :)', Xi, k)';
    T(i) = yT' * kr + b;
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







