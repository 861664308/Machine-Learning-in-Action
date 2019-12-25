# -*- coding: utf-8 -*-
import numpy as np
class logistic():
    def __init__(self, X, Y, learning_rate = 1e-2, epoch = 10000, process = False):
        self.data = X #大小：样本数*特征数
        self.label = Y #大小：样本数*1
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.process = process
        self.W, self.b = self.training()
    def sigmoid(self,x):
        return .5 * (1 + np.tanh(.5 * x))#1./(1 + np.exp(-x))
    def training(self):
        w = np.random.rand(1, self.data.shape[1]) #大小：1*特征数
        b = 0
        num_sample = self.data.shape[0]
        y_hat = self.sigmoid(w.dot(self.data.T) + b) #大小：1 * 样本数
        epsilon = 1e-6 #加上epsilon保证数值稳定性
        loss = np.sum(-self.label * np.log(y_hat + epsilon) - (1. - self.label) * np.log(1. - y_hat + epsilon))
        if self.process:
            print('Init Loss = ' + str(loss))
        #使用梯度下降进行更新
        for i in range(self.epoch):
            error = y_hat - self.label.T #大小：1 * 样本数
            w -= self.learning_rate / num_sample * error.dot(self.data)
            b -= self.learning_rate / num_sample * np.sum(error)
            y_hat = self.sigmoid(w.dot(self.data.T) + b)
            if (i + 1) % 100 == 0 and self.process:
                loss = np.sum(-self.label * np.log(y_hat + epsilon) - (1. - self.label) * np.log(1. - y_hat + epsilon))
                print('Epoch = ' + str(i + 1) + ', Loss = ' + str(loss))
        return w, b
    def score(self, X, Y):
        data = X.T
        label = Y
        Y_hat = self.sigmoid(self.W.dot(data) + self.b) > 0.5
        Y_hat = Y_hat + 0
        error = np.abs(label - Y_hat.reshape(label.shape))
        score = 1 - np.mean(error)
        return score
if __name__=='__main__':
    from sklearn.datasets import load_breast_cancer #导入乳腺癌数据集
    from sklearn.linear_model import LogisticRegression #导入逻辑回归算法
    from sklearn.model_selection import train_test_split
    training_set = load_breast_cancer()
    data, label = training_set.data, training_set.target.reshape((-1,1)) #获取训练数据
    #划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=.3, shuffle=True)
    #使用自定义模型
    model = logistic(x_train, y_train) #在训练集上训练
    print('自定义模型：')
    print('训练集精度：' + str(model.score(x_train, y_train)))
    print('测试集精度：' + str(model.score(x_test, y_test)))
    #使用sklearn模型
    lr_model = LogisticRegression(solver='liblinear')
    lr_model.fit(x_train, y_train.ravel())
    print('sklearn模型：')
    print('训练集精度：' + str(lr_model.score(x_train, y_train.ravel())))
    print('测试集精度：' + str(lr_model.score(x_test, y_test.ravel())))