import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, log_loss
from sklearn.metrics import confusion_matrix

# Load a generated binary class dataset
X, y = make_classification(n_samples=500,n_classes=2,random_state=42)

# Split it into train (70%) and test (30%) datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Sigmoid activation function.
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Limits y_pred values to the range [epsilon, 1-epsilon]. This is to prevent the log_loss function from escaping invalid values.
def log_loss(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)).mean()



# Implement Logistic Regression (with log loss function) using "loop" style
def train_logistic_regression_loop(X, y, learning_rate=0.01, epochs=1000):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0

    costs = []
    for epoch in range(epochs):
        # Forward pass (loop)
        y_pred = np.zeros(n_samples)
        for i in range(n_samples):
            z = np.dot(X[i], weights) + bias
        y_pred = sigmoid(np.dot(X, weights) + bias)

        # Compute loss (loop)
        cost = 0
        for i in range(n_samples):
            cost += log_loss(y[i], y_pred[i])
        cost /= n_samples  # Average loss over all samples

        costs.append(cost)

        # Backward pass (loop)
        derivative_weights = np.zeros(n_features)
        derivative_bias = 0
        
        for i in range(n_samples):
            derivative_weights += (y_pred[i] - y[i]) * X[i]
            derivative_bias += y_pred[i] - y[i]

        derivative_weights /= n_samples
        derivative_bias /= n_samples

        # Update parameters (loop)
        for i in range(n_features):
            weights[i] -= learning_rate * derivative_weights[i]
        bias -= learning_rate * derivative_bias

    return weights, bias, costs


# Implement Logistic Regression (with log loss function) using "vectorization" style
def train_logistic_regression_vectorized(X, y, learning_rate=0.01, epochs=1000):
  n_samples, n_features = X.shape
  weights = np.zeros(n_features)
  bias = 0

  costs = []
  for epoch in range(epochs):
    # Forward pass (vectorized)
    z = np.dot(X, weights) + bias
    y_pred = sigmoid(z)

    # Compute loss (vectorized)
    cost = np.mean(log_loss(y, y_pred))  # Average loss over all samples
    costs.append(cost)

    # Backward pass (vectorized)
    derivative_weights = (1 / n_samples) * np.dot(X.T, (y_pred - y))
    derivative_bias = (1 / n_samples) * np.sum(y_pred - y)

    # Update parameters (vectorized)
    weights -= learning_rate * derivative_weights
    bias -= learning_rate * derivative_bias

  return weights, bias, costs

# Train Logistic Regression model with 1000 epochs
epochs = 1000

# Loop style
start_time_loop = time.time()
weights_loop, bias_loop, costs_loop = train_logistic_regression_loop(X_train, y_train, epochs=epochs)
end_time_loop = time.time()

# Vectorization style
start_time_vectorization = time.time()
weights_vectorization, bias_vectorization, costs_vectorization = train_logistic_regression_vectorized(X_train, y_train, epochs=epochs)
end_time_vectorization = time.time()

# Show time taken by loop and vectorization styles after training is finished
print("Time taken by loop style: {} seconds".format(end_time_loop - start_time_loop))
print("Time taken by vectorization style: {} seconds".format(end_time_vectorization - start_time_vectorization))



# Draw the cost convergence throughout 1000 epochs
plt.plot(np.arange(epochs), costs_loop, label='Loop')
plt.plot(np.arange(epochs), costs_vectorization, label='Vectorization')
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.title('Cost Convergence')
plt.legend()
plt.show()


# Apply fitted model on testing data
def predict(X, weights, bias):
    return sigmoid(np.dot(X, weights) + bias)

# Show the classification output on a confusion matrix.
y_pred_loop = predict(X_test, weights_loop, bias_loop)
y_pred_vectorization = predict(X_test, weights_vectorization, bias_vectorization)


print("Confusion Matrix (Loop Style):")
print(confusion_matrix(y_test, y_pred_loop.round()))

print("\nConfusion Matrix (Vectorization Style):")
print(confusion_matrix(y_test, y_pred_vectorization.round()))
