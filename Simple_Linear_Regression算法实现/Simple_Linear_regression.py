import numpy as np
#用普通for循环实现的线性回归
class Simple_Linear_Regression:
    def __init__(self):
        self.a = None
        self.b = None

    def fit(self, x_train, y_train):
        #确保x_train为一维向量
        assert x_train.ndim == 1
        # 确保x_train和y_train的个数相等
        assert len(x_train) == len(y_train)
        x_mean = x_train.mean()
        y_mean = y_train.mean()

        num = 0.0
        #最小二乘法的分子
        d = 0.0
        #最小二乘法的分母 

        for x, y in zip(x_train, y_train):
            num += (x - x_mean)*(y - y_mean)
            d += (x - x_mean)**2

        self.a = num / d
        self.b = y_mean - self.a * x_mean
        return self

    def predict(self, X):
        # X为数组集
        assert X.ndim == 1
        # X为一维向量
        assert self.a is not None and self.b is not None
        # X为一个数据集
        return np.array([self._predict(x) for x in X])

    def _predict(self,x):
        y_hat = self.a * x + self.b
        return y_hat


