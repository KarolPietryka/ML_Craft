import numpy as np
training_data = np.array([1, 2, 3, 4, 5])
target_data = np.array([1, 2, 3, 4, 5])
w = 0
b = 0
n = len(training_data)


def get_mse(training_data, target_data, w, b):
    pred = w * training_data + b
    return np.mean((pred - target_data) ** 2)


def get_gradients(training_data, target_data, w, b):
    pred = w * training_data + b
    error = pred - target_data
    total_w = np.dot(error, training_data) / n
    total_b = np.mean(error)
    return [total_w, total_b]


def gradient_descent(gradients, learning_rate, w, b):
    w += -learning_rate * gradients[0]
    b += -learning_rate * gradients[1]
    return w, b


iteration = 10000
learning_rate = 0.01

mse = get_mse(training_data, target_data, w, b)
print(f"Initial MSE: {mse}")
for i in range(iteration):
    gradients = get_gradients(training_data, target_data, w, b)
    w, b = gradient_descent(gradients, learning_rate, w, b)
    if i % 10 == 0:
        mse = get_mse(training_data, target_data, w, b)
        print(f"MSE for iteration {i}: {mse}. w: {w}, b: {b}")

mse = get_mse(training_data, target_data, w, b)
print(f"Final w: {w}, and b: {b} with MSE: {mse}")
