import numpy as np
import math
import matplotlib.pyplot as plt
from data_process_softmax import get_data_set
from plot_confusion_matrix import plot_Matrix


def sigmoid(t):
    s = 1 / (1 + math.exp(-t))
    if s == 0:
        return 0.0001
    elif s == 1:
        return 0.9999
    else:
        return s

def softmax(w, x, y):
    p = y.shape[0]
    xw = np.dot(x, w)
    N = xw.shape[1]
    cost = np.zeros([1, N])
    for i in range(N):
        for j in range(p):
            cost[0][i] += math.log(1 + np.exp(-y[j][0]*xw[j][i]))
        cost[0][i] = cost[0][i] / p
    return cost

def approx_grad(w, x, y, delta):
    N = w.shape[0]
    dw = np.identity(N) * delta
    grad = (softmax(w + dw, x, y) - softmax(w, x, y)) / delta
    return grad


def gradient_descent(w, x, y, alpha, delta, n_iter):
    trained_w = w
    cost = 1
    k = 0
    # plotting cost function
    plt.ion()
    plt.figure(1)

    while k < n_iter or cost < 0.01:
        grad = approx_grad(trained_w, x, y, delta)
        trained_w = trained_w - alpha * grad.T / np.linalg.norm(grad)
        cost = softmax(trained_w, x, y)

        print('iteration: ', k, ', cost: ', np.squeeze(cost))
        plt.plot(k, np.squeeze(cost), c='r', marker='.', mec='r')

        k = k + 1

    plt.title('Cost Function Curve in Training')
    plt.xlabel('Iteration')
    plt.ylabel('Cost Value')
    plt.savefig('cost_function_curve.jpg', dpi=500)
    plt.show()
    return trained_w

# Train
train_X, train_Y = get_data_set(0)  # 从train取数
w0 = np.ones([30, 1])
w0[:, :] = 0

bias_x = np.ones([train_X.shape[0], 1])
train_X = np.column_stack((bias_x, train_X))
new_w = gradient_descent(w0, train_X, train_Y, 0.001, 0.00001, 7000)
print(new_w)

# Test
test_X, test_Y = get_data_set(1)
bias_x_test = np.ones([test_X.shape[0], 1])
test_X = np.column_stack((bias_x_test, test_X))
test_res = np.dot(test_X, new_w)


for i in range(test_res.shape[0]):
    test_res[i][0] = math.tanh(test_res[i][0]+1)
    if test_res[i][0] > 0:
        test_res[i][0] = 1
    else:
        test_res[i][0] = -1

# Test Result Metrics
test_P = test_res.shape[0]
total_0 = 0
total_1 = 0
test_0 = 0
test_1 = 0
acc_0 = 0
acc_1 = 0
for i in range(test_P):
    if test_Y[i][0] == 1:
        total_1 = total_1 + 1
        if test_res[i][0] == test_Y[i][0]:
            test_1 = test_1 + 1
    if test_Y[i][0] == -1:
        total_0 = total_0 + 1
        if test_res[i][0] == test_Y[i][0]:
            test_0 = test_0 + 1

acc_0 = test_0 / total_0
acc_1 = test_1 / total_1
acc = (acc_0 + acc_1) / 2
print('test accuracy:', acc)
# plot Confusion Matrix
cm = np.zeros((2, 2))
cm[0][0] = test_0
cm[0][1] = total_0 - test_0
cm[1][0] = total_1 - test_1
cm[1][1] = test_1
plot_Matrix(cm, [0, 1])
