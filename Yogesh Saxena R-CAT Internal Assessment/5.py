import numpy as np

class SimpleLR:
    def __init__(self):
        self.intercept_ = None
        self.coeff = None

    def fit(self, X, y):
        sample = X.shape[0]
        X_b = np.c_[np.ones((sample, 1)), X] 

        self.coeff = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        self.intercept_ = self.coeff[0]
        self.coeff = self.coeff[1:]

    def predict(self, X):
        sample = X.shape[0]
        X_b = np.c_[np.ones((sample, 1)), X] 
        return X_b.dot(np.concatenate([[self.intercept_], self.coeff]))

if __name__ == "__main__":
    np.random.seed(0)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)

    model = SimpleLR()
    model.fit(X, y)

    print("Intercept:", model.intercept_)
    print("Coefficient:", model.coeff)

    X_new = np.array([[0], [2]])
    y_pred = model.predict(X_new)
    print("Predictions:", y_pred)
