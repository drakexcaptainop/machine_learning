N = 100;
u1 = [1, 1];
S1 = randn(2);
S1 = S1'*S1;

u2 = -u1;
S2 = randn(2);
S2 = S2'*S2;




X1 = mvnrnd( u1, S1, N );
X2 = mvnrnd( u2, S2, N );

Y = zeros( N*2, 1 );
Y(1:N) = 1;


X = [X1; X2];
X = [X, ones(N*2, 1)];

[w, L] = train( X, Y, 10 );

subplot(121)

hold on
plot(X1(:, 1), X1(:, 2), 'ro')
plot(X2(:, 1), X2(:, 2), 'bo')
xl = xlim;
yl = ylim;
No = 100;
[Xo, Yo ] = meshgrid( linspace(xl(1), xl(2), No), linspace(yl(1), yl(2), No) );
Zo = [reshape(Xo, No*No, 1), reshape(Yo, No*No, 1)];
Zo = predict(Zo, w);
Zo = reshape(Zo, size(Xo));

contourf(Xo, Yo, Zo, [xl(1), 0.5, xl(2)], FaceAlpha=.2)

subplot(122)
plot(L)







%%
function z = sigm(a)
z = 1./(1+exp(a));
end

function h = H(X, g)
h = inv( X'*diag(g)*X );
end


function gr = grad(X, Y, g)
gr = X'*(Y - g);
end

function l = logistic_likelihood(Y, g)
l = sum(Y.*log( g ) + (1 - Y) .* log(1 - g),"all");
end

function [g, a] = predict(X, w)
    X = [X, ones(size(X, 1), 1)];
    a = X * w;
    g = sigm(a);
end

function [w, L] = train( X, Y, epochs )

L = zeros(epochs);
w = zeros(size(X, 2), 1);

for i = 1:epochs

    g = sigm( X*w );
  
    w = w - H(X, g) * grad(X, Y, g);
    L(1) = logistic_likelihood(Y, g);
    if i>1
        if abs(L(2) - L(1)) / L(1) * 100 < 1e-6
            break
        end
    end
end

end
