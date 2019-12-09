import numpy as np
class perceptron(object):
    def __init__(self, X, Y, learning_rate = 1, epoch = 2000):
        '''
        :param X: 训练数据 p * N
        :param Y: 训练标签 1 * N
        :param learning_rate: 学习率
        '''
        self.data = X
        self.label = Y
        labels = np.unique(self.label)
        self.label[self.label == labels[0]] = -1
        self.label[self.label == labels[1]] = 1
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.W, self.b = self.training()
    def training(self):
        L = self.data.shape[0]
        #初始化参数
        w = np.zeros((1, L))
        b = 0
        flag = True
        i = 0
        while flag and i < self.epoch:
            T = self.label * (w.dot(self.data) + b)
            if (T > 0).all(): #如果T全部大于0，训练结束
                flag = False
            else:
                #随机挑选训练数据
                Index = np.argwhere(T[0] <= 0)#Index是n x 1的数组
                index =Index[int(np.random.rand() * Index.shape[0])][0]
                #更新参数
                w = w + self.learning_rate * self.label[0][index] * self.data[:, index].reshape(w.shape)
                b = b + self.learning_rate * self.label[0][index]
            i += 1
        return w, b
if __name__ == '__main__':
    from sklearn.datasets import make_classification
    from sklearn.linear_model import Perceptron
    #构造数据集
    x, y = make_classification(n_samples=1000,
        n_features=2, n_redundant=0, n_informative=1, n_clusters_per_class=1)
    #使用自己的模型做预测
    model = perceptron(x.T, y.reshape((1, len(y))), epoch=1000)
    #获取权重
    W = model.W
    b = model.b
    #计算训练集上的预测值
    pred = W.dot(x.T) + b
    #计算准确率
    acc = np.sum(pred * y.reshape((1, len(y))) > 0) / len(y)
    #使sklearn做预测
    clf = Perceptron(fit_intercept=False, max_iter=1000, shuffle=False)
    clf.fit(x, y)
    #计算训练集上的准确率
    acc1 = clf.score(x, y)