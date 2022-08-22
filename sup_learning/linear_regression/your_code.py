import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import stats
import pytest
from data_load import extract_data, transform_data


def compute_cost(X, y, t0, t1):
    """Computes the least squares cost function."""
    ### YOUR CODE HERE ###

    ### YOUR CODE HERE ###
    return 


def gradient_descent(x, y, alpha=0.001, ep=0.0001, max_iter=10000):
    """Implements batch gradient descent using a naive approach."""
    converged = False
    iter = 0
    m = x.shape[0] # number of samples

    # initial theta (t0, t1)
    t0 = np.random.random(x.shape[1])
    t1 = np.random.random(x.shape[1])

    # initial cost J(theta)
    ### YOUR CODE HERE ###
    J = None
    ### YOUR CODE HERE ###

    # perform batch gradient descent
    while not converged:
        # for each training sample, compute the gradient (d/d_theta j(theta))
        ### YOUR CODE HERE ###
        grad0 = None 
        grad1 = None
        ### YOUR CODE HERE ###

        # update the temporary theta parameters
        temp0 = t0 - alpha * grad0
        temp1 = t1 - alpha * grad1
    
        # update theta with temp values
        t0 = temp0
        t1 = temp1

        # update cost with new theta parameters
        ### YOUR CODE HERE ###
        J_iter = None
        ### YOUR CODE HERE ###

        if abs(J - J_iter) <= ep:
            converged = True

        J = J_iter   # update cost
        print(f"Iter {iter}: J: {J[0]}")
        iter += 1  # update iter

        if iter == max_iter:
            print('Max interactions exceeded!')
            converged = True

    print(f"Converged at {iter} iterations - J: {J}")
    print(f"theta0: {t0}, theta1: {t1}")
    return t0, t1


if __name__ == "__main__":
    # load and transform data
    data = extract_data(os.path.abspath("data/data.txt"))
    X, y = transform_data(data)

    # call gredient decent, and get intercept(theta0) and slope(theta1)
    theta0, theta1 = gradient_descent(X, y, alpha=0.01, ep=1e-8, max_iter=10000)
    
    # check results with scipy linear regression
    slope, intercept, r_value, p_value, slope_std_error = stats.linregress(X[:, 0], y)
    print(f"Spicy LR model: theta0: {intercept}, theta1: {slope}")
    assert intercept == pytest.approx(theta0, 0.01) and slope == pytest.approx(theta1, 0.01)
    
    # construct and visualize regression line
    y_predict = theta0 + theta1 * X
    fit_line = np.linspace(y_predict.min(), y_predict.max())
    x = np.linspace(X.min(), X.max())
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(x, fit_line, 'r', label='Prediction')
    ax.scatter(X, y, label='Traning Data')
    plt.show()
