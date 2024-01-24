import numpy as np

class RidgeRegression:
    def __init__(self, alpha=1.0):
        if alpha <= 0:
           raise ValueError("The regularization term alpha has to be positive in order to maintain \n"
                             "the positive definiteness of the matrix used in the Cholesky \n"
                             "decomposition.")
        self.alpha = alpha
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Centering X and y
        X_mean = np.mean(X, axis=0)
        y_mean = np.mean(y)
        X_centered = X - X_mean
        y_centered = y - y_mean

        # Regularization matrix for centered X
        R = self.alpha * np.eye(n_features)

        # Cholesky decomposition
        C = np.linalg.cholesky(X_centered.T @ X_centered + R)
        self.weights = np.linalg.solve(C.T, np.linalg.solve(C, X_centered.T @ y_centered))

        # Calculating bias
        self.bias = y_mean - np.dot(X_mean, self.weights)

    def predict(self, X):
        return X @ self.weights + self.bias



