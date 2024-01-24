import numpy as np

class RidgeRegression:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features= X.shape
        X_bias = np.hstack([np.ones((n_samples, 1)), X])## bias 
        R =self.alpha* np.eye(n_features + 1)## regularization
        R[0, 0] =0 #no regualization for bias otherwise high bias
        C = np.linalg.cholesky(X_bias.T @ X_bias + R)## using cholesky here for improved numerical stability
        self.weights = np.linalg.solve(C.T, np.linalg.solve(C, X_bias.T @ y))# explained in the report

        self.bias =self.weights[0]#get bias
        self.weights =self.weights[1:]#get weights

    def predict(self, X):
        return X@self.weights+ self.bias


