import numpy as np
from data_process import get_data_set
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
        cost = cross_entropy(trained_w, x, y)

        print('iteration: ', k, ', cost: ', np.squeeze(cost))
        plt.plot(k, np.squeeze(cost), c='r', marker='.', mec='r')

        k = k + 1

    plt.title('Cost Function Curve in Training')
    plt.xlabel('Iteration')
    plt.ylabel('Cost Value')
    plt.show()
    return trained_w


def plot_Matrix(cm, classes, title='Confusion Matrix in Testing', cmap=plt.cm.Blues):
    plt.rc('font', family='Times New Roman', size='8')  # 设置字体样式、大小

    # 按行进行归一化
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # print("Normalized confusion matrix")
    str_cm = cm.astype(np.str).tolist()
    for row in str_cm:
        print('\t'.join(row))
    # 占比1%以下的单元格，设为0，防止在最后的颜色中体现出来
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if int(cm[i, j] * 100 + 0.5) == 0:
                cm[i, j] = 0

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    # ax.figure.colorbar(im, ax=ax) # 侧边的颜色条带

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='Actual',
           xlabel='Predicted')

    # 通过绘制格网，模拟每个单元格的边框
    ax.set_xticks(np.arange(cm.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(cm.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    # 将x轴上的lables旋转45度
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # 标注百分比信息
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if int(cm[i, j] * 100 + 0.5) > 0:
                ax.text(j, i, format(int(cm[i, j] * 100 + 0.5), fmt) + '%',
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig('cm.jpg', dpi=300)
    plt.show()


# Train
train_X, train_Y = get_data_set(0)
w0 = np.ones([30, 1])
w0[:, :] = 0

bias_x = np.ones([train_X.shape[0], 1])
train_X = np.column_stack((bias_x, train_X))
new_w = gradient_descent(w0, train_X, train_Y, 0.001, 0.00001, 40000)
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
