import numpy as np
from Simple_Linear_Regression import metrics
#用向量法实现的线性回归,向量法实现的线性回归法效率比较高
class Simple_Linear_Regression2:
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

        num = (x_train - x_mean).dot(y_train - y_mean)
        #最小二乘法的分子
        d = (x_train - x_mean).dot(x_train - x_mean)
        #最小二乘法的分母
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

    def score(self,x_test, y_test):
        y_predict = self.predict(x_test)
        return metrics.r2_sore(y_test, y_predict)


