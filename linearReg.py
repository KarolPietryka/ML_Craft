training_data = [1, 2, 3, 4, 5]
target_data = [-2, -4, -6, -4, -5]
w = 0
b = 0
n = len(training_data)


# why to count it?
def get_mse(training_data, target_data, w, b):
    total = 0.0
    for i in range(n):
        pred_i = w * training_data[i] + b
        total += (pred_i - target_data[i]) ** 2
    return total / n


def get_gradients(training_data, target_data, w, b):
    total_w = 0.0
    total_b = 0.0
    for i in range(n):
        pred_i = w * training_data[i] + b
        total_b += (pred_i - target_data[i])
        total_w += (pred_i - target_data[i]) * training_data[i]
    return [total_w / n, total_b / n]


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
