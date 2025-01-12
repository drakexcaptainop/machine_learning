from modelBuild import builder, linear, polynomial
import numpy as np
import matplotlib.pyplot as plt
N = 100
x = np.linspace(-10, 10, N)[:, None]
X1 = np.random.randn( N , 1 )
X2 = np.random.exponential( 1, [N, 1] )
X = np.hstack( (X1, X2) )
Y = 2 * X1 + X2 ** 2 + np.random.randn( N, 1 ) 
model1 = builder( linear(), polynomial(3) )
print(X.shape, Y.shape)
model1.fit( X, Y, 100 )
plt.plot( model1._losses )
plt.waitforbuttonpress()
