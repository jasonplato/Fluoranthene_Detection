#!/usr/bin/env python
# coding: utf-8

# In[74]:


import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

def sigmoid(t):
    s = 1 / (1 + math.exp(-t))
    if s == 0:
        return 0.0001
    elif s == 1:
        return 0.9999
    else:
        return s
def cross_entropy(w, x, y):
    p = y.shape[0]
    xw = np.dot(x, w)
    N = xw.shape[1]
    for i in range(xw.shape[0]):
        for j in range(xw.shape[1]):
            xw[i][j] = sigmoid(xw[i][j])
    cost = np.zeros([1, N])
    for i in range(N):
        for j in range(p):
            cost[0][i] += - y[j][0] * math.log(xw[j][i]) - (1 - y[j][0]) * math.log(1 - xw[j][i])
        cost[0][i] = cost[0][i] / p
    return cost
def approx_grad(w, x, y, delta):
    N = w.shape[0]
    dw = np.identity(N) * delta
    grad = (cross_entropy(w + dw, x, y) - cross_entropy(w, x, y)) / delta
    return grad
def hessian(w,x,y):
    p=y.shape[0]
    d=x.shape[1]
    xw=np.dot(x,w)
    for i in range(xw.shape[0]):
        xw[i][0] = sigmoid(xw[i][0])
    sum=np.zeros([d,d])
    for k in range(p):
        trans=np.transpose(x[k,:])
        sum=sum+(xw[k][0]*(1-xw[k][0])*np.dot(trans,x[k,:]))
    hessian=sum/p
    return hessian
def newton_method(w,x,y,n_iter):
    trained_w = w
    cost = 1
    k = 0
    # plotting cost function
    plt.ion()
    plt.figure(1)
    while k < n_iter:
        gradient = approx_grad(trained_w, x, y, 0.00001)
        hess = hessian(trained_w, x, y)
        A = hess
        b = np.transpose(gradient)
        trained_w = trained_w - np.dot(np.linalg.pinv(A),b)
        cost = cross_entropy(trained_w, x, y)
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
train_X, train_Y = get_data_set(0)
w0 = np.ones([30, 1])
w0[:, :] = 0

bias_x = np.ones([train_X.shape[0], 1])
train_X = np.column_stack((bias_x, train_X))
new_w = newton_method(w0, train_X, train_Y,1000)
print(new_w)
# Test
test_X, test_Y = get_data_set(1)
bias_x_test = np.ones([test_X.shape[0], 1])
test_X = np.column_stack((bias_x_test, test_X))
test_res = np.dot(test_X, new_w)
for i in range(test_res.shape[0]):
    test_res[i][0] = sigmoid(test_res[i][0])
    if test_res[i][0] > 0.5:
        test_res[i][0] = 1
    else:
        test_res[i][0] = 0
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
    else:
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


# In[ ]:




