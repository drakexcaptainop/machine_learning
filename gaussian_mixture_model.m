
img = imresize(im2double(imread('')), [128 128]);
imshow(img)
%%
X = reshape(img, [128^2, 3]);
[llh, us, Ss, ps] = fitgmm(X, 3, 2);
gm = fitgmdist(X, 3);

I1 = gmm_cluster( X,us, Ss, ps  );
I2 = gm.cluster(X);
csl = [0 .5 1];
M1 = zeros(size(X));
M2 = zeros(size(M1));
for i=unique(I)'
    M1(I1==i, :) = csl(i);
    M2(I2==i, :) = csl(i);
end

%%
subplot(141)
imagesc(img);
subplot(142)
imagesc(reshape(1-M1, size(img)))
subplot(143)
imagesc(reshape(1-M2, size(img)))
subplot(144)
colormap gray
imagesc(im2gray(img))

%%

function G = gamma_matrix(X, ps, us, Ss)
N = size(X, 1);
D = numel(ps);
G = zeros(N, D);
for j=1:D
    G(:, j) = mvnpdf( X, us{j}, Ss{j} ) * ps{j};
end
G = G ./ sum(G, 2);
end

function ll = gmm_log_likelihood(G)
ll =  sum( log( sum(G, 2) ) );
end

function I = gmm_cluster(X, us, Ss, ps)
N = size(X, 1);
D = numel(us);

R = zeros( N, D );
for j=1:D
    R(:, j) = mvnpdf( X, us{j}, Ss{j} ) * ps{j};
end
[~, I ] = max(R, [], 2);
end

function [llh, us, Ss, ps] = fitgmm(X, D, epochs)
us = cell(D, 1);
Ss = cell(D, 1);
ps = cell(D, 1);
N = size(X, 1);

I = kmeans(X, D);
for j = 1:D
    us{j} = mean( X(I==j, :), 1 );
    Ss{j} = cov( X(I==j, :) );
    ps{j} = sum(I==j) / N;
end

llh = zeros(epochs, 1);

for i=1:epochs
    G = gamma_matrix(X, ps, us, Ss);
    llh(i) = gmm_log_likelihood(G);
    for k=1:D
        Nk = sum(G(:, k));
        us{k} = (X'*G(:, k))'./Nk;
        dgk = diag( G(:, k) );
        Xz = X - us{k};
        Ss{k} = 1/Nk * Xz'*dgk*Xz;
        ps{k} = Nk/sum(G,"all");
    end
end


end


