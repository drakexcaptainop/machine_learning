import numpy as np

def ones_concat(X):
    return np.hstack( (X, np.ones( [X.shape[0], 1] )) )

def add_dim(X):
    if len(X.shape) >= 2:
        return X
    return X[:, None]

class model:
    params: np.array
    def fit():
        return NotImplemented  
    def predict():
        return NotImplemented
    def transform():
        return NotImplemented

class parametric(model):
    def __init__(self):
        super().__init__()
    def _init_params(self, size, default=0 ):
        self.params = np.zeros( size ) + default

class nonparametric(model):
    def __init__(self):
        super().__init__() 

class linear(parametric):
    def __init__(self):
        super().__init__()
    def fit(self, X, y):
        X = add_dim( X )
        X = ones_concat( X )
        self.params = np.linalg.inv(X.T@X)@X.T@y
    def predict(self, X):
        X = add_dim( X )
        X = ones_concat( X )
        return X@self.params
    def _init_params(self, feats, default=0):
        return super()._init_params([feats+1, 1], default)

class naradaya(nonparametric):
    pass 


class polynomial(parametric):
    degree: int
    def __init__(self, degree):
        super().__init__()
        self.degree = degree

    def transform(self, X):
        X = add_dim( X )
        O = X.copy()
        for i in range( self.degree, 1, -1 ):
            X = np.hstack((O**i, X))
        
        return X 
    
    def fit(self, X, y):
        X = add_dim( X )
        X = self.transform( X )
        X = ones_concat( X )
        self.params = np.linalg.inv(X.T@X)@X.T@y
    def predict( self, X ):
        X = add_dim( X )
        X = ones_concat(self.transform( X ))
        return X@self.params
    def _init_params(self, feats, default=0):
        return super()._init_params([self.degree + 1, 1], default)

class builder(model):
    funcs: list[model]
    R: np.array
    Yhats: np.array
    _losses: np.array
    def __init__(self, *funcs):
        super().__init__()
        self.funcs = funcs

    def fit( self, X, y, max_iters=5, eps=1e-4 ):
        self.R = np.zeros( [X.shape[0], len(self.funcs)] )
        self.Yhats = np.zeros( [X.shape[0], len(self.funcs)] )
        self._losses = np.zeros( max_iters )
        if X.shape[1] != len(self.funcs):
            raise Exception("Invalid # of additive functions")
        for _model in self.funcs:
            if isinstance(_model, parametric):
                _model._init_params( 1 )
        
        for _ in range( 0, max_iters ):
            self.compute_yhats( X )
            self.fit_models( X )
            self.update_residuals(X, y)
            self._losses[_]=( np.sum((y - add_dim(np.sum( self.Yhats, 1 )))**2) )
        

        #R[j] = y - Sum(i, Funcs[i](X), i != j )
        #
    def gauss(self,):

        pass
    def compute_yhats(self, X, i: int=None):
        if not (i is None):
            self.Yhats[ :, i ] = self.funcs[i].predict( X[:, i] ).ravel()
        
        for i in range( len(self.funcs) ):
            self.Yhats[ :, i ] = self.funcs[i].predict( X[:, i] ).ravel()
    def fit_models(self, X):
        for i in range(len(self.funcs)):
            self.funcs[i].fit( X[:, i], self.R[:, i] )

    def partial_residual2(self, i, y):
        I = np.arange( i )
        P = add_dim(np.sum(self.Yhats[ :, I[I!=i] ], 1))
        #print(f'{P.shape = }, { y.shape =  }, { (y - P).shape =  }')
        return y - P

    def update_residuals(self, X, y):
        for i in range(len(self.funcs)):
            self.R[:, i] = self.partial_residual2( i, y ).ravel()
         

    def partial_residual( self, i: int, X, y ):
        I = np.arange(len(self.funcs))
        yhat =  self.funcs[i].predict( X[:, i] )
        r = y - np.sum(yhat[ :, I[I!=i] ], 1)
        return r
    
    def residuals(self, X, y):
        R = np.zeros( [X.shape[0], len(self.funcs)] )
        for i in range( len( self.funcs ) ):
            R[:, i] = self.partial_residual( i, X, y )
        return R
            
    def predict(self, X):
        Yhat = np.zeros( [X.shape[0], 1] )
        for i in range( len( self.funcs ) ):
            Yhat += self.funcs[i].predict( X[:, i] )
        return Yhat   