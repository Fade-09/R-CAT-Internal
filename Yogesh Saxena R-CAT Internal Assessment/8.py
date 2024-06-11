import numpy as np

def gradient_descent_linear_regression(X, y, learning_rate=0.01, num_iterations=1000):
    m, n = X.shape  
    w = np.zeros(n)  
    b = 0  
    history = []  
    for i in range(num_iterations):
        y_pred = np.dot(X, w) + b 
        loss = (1 / (2 * m)) * np.sum((y_pred - y) ** 2)
        history.append(loss)      
        dw = (1 / m) * np.dot(X.T, (y_pred - y))
        db = (1 / m) * np.sum(y_pred - y)   
        w -= learning_rate * dw
        b -= learning_rate * db
        if i % 100 == 0:
            print(f"Iteration {i}: Loss = {loss}")
    return w, b, history
X = np.array([[1, 2],
              [2, 3],
              [4, 5],
              [3, 2],
              [5, 4]])
y = np.array([3, 5, 9, 6, 8])
learning_rate = 0.01
num_iterations = 1000
weights, bias, cost_history = gradient_descent_linear_regression(X, y, learning_rate, num_iterations)
print("Weights:", weights)
print("Bias:", bias)
