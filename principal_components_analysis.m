C = randn(2);
C = C'*C;
N = 100;
X = mvnrnd( [0, 0], C, N );

hold on
plot(X(:, 1), X(:, 2), 'o')

plot( [0, 0], ylim, 'k-' )
plot( xlim,[0, 0], 'k-' )

S = X'*X / N;

[V, D] = eig(S);


[d, I ] = sort( diag(D), 1, "descend");
Vsorted = V(:, I);
Dsorted = diag(d);

b = Vsorted(:, 1);

w = b(2)/b(1);

xl = xlim;

x = linspace(xl(1), xl(2));

plot(x, x*w);


coeff = pca(X, "NumComponents",1, Algorithm="svd");
w2 = coeff(2)/coeff(1);

plot(x, x*w2);

legend({"X", "EIG", "SVD"})
