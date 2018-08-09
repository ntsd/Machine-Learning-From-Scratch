import numpy as np
import pandas as pd
np.random.seed(44) # Set seed for same random

data = pd.read_csv('dataset/house-prices/train.csv')
X = data[['GrLivArea', 'TotalBsmtSF']]
y = data['SalePrice']
X = X.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x))) # normalise
X = X.values
y = y.values

alpha = 0.01 # learning rate
iterations = 1000 # number of iterations
m = y.size # size of data

# GRADIENT DESCENT
def gradient_descent(X, y, iterations, alpha):
    X = np.insert(X, 0, 1, axis=1) # Insert constant ones for bias weights
    theta = np.random.rand(X.shape[1]) # Initiealize theta
    past_costs = []
    past_thetas = [theta]
    for _ in range(iterations):
        prediction = np.dot(X, theta)
        error = prediction - y
        cost = 1/(2*m) * np.dot(error.T, error)
        past_costs.append(cost)
        gradient_error = (1/m) * np.dot(X.T, error)
        theta = theta - (alpha * gradient_error)
        past_thetas.append(theta)
        
    return past_thetas, past_costs

past_thetas, past_costs = gradient_descent(X, y, iterations, alpha)

print(past_costs[-1])

import matplotlib.pyplot as plt
#Plot the cost function...
plt.title('Cost Function J')
plt.xlabel('No. of iterations')
plt.ylabel('Cost')
plt.plot(past_costs)
plt.show()