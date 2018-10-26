import numpy as np
from Simple_Linear_Regression import metrics
#多元线性回归
class LinearRegression:

    def __init__(self):
        self.coef_ = None   #系数
        self.interception_ = None  #截距
        self._theta = None   #所有的theta值

    def fit_normal(self, X_train, y_train):
        assert X_train.shape[0] == y_train.shape[0]  # b保证X_train和y_train的行数一样
        X_b = np.hstack([np.ones((len(X_train),1)), X_train])  #在X_train的第一列前加一列1方便计算
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train) #将theta的值算出来
        self.interception_ = self._theta[0]    #截距
        self.coef_ = self._theta[1:]           #系数
        return self

    def predict(self,X_predict):
        assert self.interception_ is not None and self.coef_ is not None
        #保证截距和系数的矩阵不能为空
        assert X_predict.shape[1] == len(self.coef_)
        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])  # 在X_train的第一列前加一列1方便计算
        return X_b.dot(self._theta)

    def score(self,X_test, y_test):
        y_predict = self.predict(X_test)
        return metrics.r2_sore(y_test, y_predict)