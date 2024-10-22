

[X, Y ] = digitTrain4DArrayData;
Y = double(Y);
%%

discriminator = dlnetwork( ...
    [ ...
        imageInputLayer([28 28 1]),
        convolution2dLayer([2, 2], 28),
        reluLayer,
        convolution2dLayer([2, 2], 64),
        reluLayer,
        convolution2dLayer([2, 2], 128),
        reluLayer,
        flattenLayer,
        fullyConnectedLayer(50),
        reluLayer,
        fullyConnectedLayer(10),
        reluLayer,
        fullyConnectedLayer(1),
        sigmoidLayer
    ]);


generator = dlnetwork([ ...
    ...
    ...
        imageInputLayer([28, 28, 1]),
        convolution2dLayer([2, 2], 50,"Padding","same"),
        reluLayer,
        convolution2dLayer([2, 2], 100, 'Padding','same'),
        reluLayer,
        transposedConv2dLayer([2, 2], 100, Stride=1),
        reluLayer,
        transposedConv2dLayer([2, 2], 1,"Cropping",1),
        sigmoidLayer
    ]);

%%


Xdl = dlarray(X, 'SSCB');
Ydl = dlarray(Y, 'SB');



%%

epochs = 1;
batch_size = 50;

for i = 1:epochs
    I = randi(size(X, 4),[batch_size, 1]);
    z = dlarray(randn( 28, 28, 1, numel(I) ), 'SSCB');
    
    fake = forward( generator,  z );

    real = Xdl( :, :, 1, I );

    Ddl = cat(4, fake, real);

    Ysdl = ones( numel(I)*2, 1 );
    Ysdl(1:numel(I)) = 0;
    Ysdl = dlarray(Ysdl, 'SB');

    pred = forward(discriminator, Ddl);

end
%%

epochs=500;



avgGradD = [];
avgSqGradD = [];
avgGradG = [];
avgSqGradG = [];
lrG = .001;
lrD = lrG;


lossD = zeros(epochs, 1);
lossG = zeros(epochs, 1);

for i = 1: epochs
    [dd, dg, lD, lG] = dlfeval(@gan_loss, Xdl, discriminator, generator, 350);
    [discriminator, avgGradD, avgSqGradD] = adamupdate(discriminator, dd, avgGradD, avgSqGradD, i, lrD);
    [generator, avgGradG, avgSqGradG] = adamupdate(generator, dg, avgGradG, avgSqGradG, i, lrG);

    lossD(i) = lD;
    lossG(i) = lG;
end
%%
plot([lossD, lossG])
%%
img = forward(generator, dlarray(randn(28, 28, 1, 1), 'SSCB') );
img = extractdata(img);

imagesc(img)

%%

function [d_grads, g_grads, loss, loss2] = gan_loss(Xdl,  discriminator, generator, batch_size )




I = randi(size(Xdl, 4),[batch_size, 1]);
z = dlarray(randn( 28, 28, 1, numel(I) ), 'SSCB');

fake = forward( generator,  z );

real = Xdl( :, :, 1, I );

Ddl = cat(4, fake, real);

Ysdl = ones( numel(I)*2, 1 );
Ysdl(1:numel(I)) = 0;
Ysdl = dlarray(Ysdl, 'SB');

pred = forward(discriminator, Ddl);

N = batch_size;

loss1 = -mean(log( pred(numel(I):end) )) ;
loss2 = -mean(log(1 - pred(1:numel(I)))) ;

loss = loss1 + loss2;
g_grads = dlgradient(-mean(log( pred(1:batch_size) )), generator.Learnables, RetainData=true);
d_grads = dlgradient(loss, discriminator.Learnables);



end








