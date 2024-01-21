import numpy as np

class RidgeRegression:
    def __init__(self, alpha=1.0, solver='svd'):
        self.alpha = alpha
        self.solver = solver
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Adding bias term
        X_bias = np.hstack([np.ones((n_samples, 1)), X])

        # Regularization matrix
        R = self.alpha * np.eye(n_features + 1)
        R[0, 0] = 0  # No regularization for the bias term

        # Solving using different methods
        if self.solver == 'svd':
            U, s, Vt = np.linalg.svd(X_bias, full_matrices=False)
            s = s / (s**2 + self.alpha)
            self.coef_ = Vt.T @ np.diag(s) @ U.T @ y
        elif self.solver == 'cholesky':
            C = np.linalg.cholesky(X_bias.T @ X_bias + R)
            self.coef_ = np.linalg.inv(C.T) @ np.linalg.inv(C) @ X_bias.T @ y
        else:
            raise ValueError("Unknown solver '{}'".format(self.solver))

        # Extracting the intercept and coefficients
        self.intercept_ = self.coef_[0]
        self.coef_ = self.coef_[1:]

    def predict(self, X):
        return X @ self.coef_ + self.intercept_


