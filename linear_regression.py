import numpy as np
import pandas as pd
np.random.seed(44) # Set seed for same random

# mean square error (loss function)
def MSE(predict, actual):
    return np.mean(0.5 * (predict - actual)**2)

# Linear Regression (Gradient Descent)
def linear_regression(X, y, iterations, alpha):
    X = np.insert(X, 0, 1, axis=1) # Insert constant ones for bias weights
    n_features = X.shape[1] # number of features
    limit = 1 / n_features**0.5 # min max random weight
    weight = np.random.uniform(-limit, limit, (n_features, )) # initialize weight
    training_errors = []
    for _ in range(iterations):
        y_pred  = X.dot(weight)
        # Calculate l2 loss
        mse = MSE(y_pred, y)
        training_errors.append(mse)
        # Gradient of l2 loss
        grad_w = -(y - y_pred).dot(X)
        weight = weight - (alpha * grad_w)

    return weight, training_errors

from sklearn.datasets import make_regression
X, y = make_regression(n_samples=100, n_features=1, noise=20)
print(y)

alpha = 0.01 # learning rate
iterations = 100 # number of iterations

weight, training_errors = linear_regression(X, y, iterations, alpha)

print(weight)

import matplotlib.pyplot as plt
#Plot the error...
plt.title('MSE Error')
plt.xlabel('No. of iterations')
plt.ylabel('Error')
plt.plot(training_errors)
plt.show()