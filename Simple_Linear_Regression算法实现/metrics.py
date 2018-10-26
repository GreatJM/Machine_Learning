import numpy as np
from math import sqrt

"""线性回归算法的评测函数
   mse函数为均方误差
   rmse函数为均方根误差
   mae函数为平均绝对误差
"""
def accuracy_score(y_true, y_predict):   #求y_true和y_predict之间的准确率
    assert y_predict.shape[0] == y_true.shape[0]
    return np.sum(y_true == y_predict) / len(y_true)

def mse(y_true,y_predict):
    assert len(y_true) == len(y_predict)
    return np.sum((y_predict - y_true)**2) / len(y_true)

def rmse(y_true,y_predict):
    return sqrt(mse(y_true,y_predict))

def mae(y_true,y_predict):
    assert len(y_true) == len(y_predict)
    return np.sum(abs(y_predict - y_true)) / len(y_true)

def r2_sore(y_test, y_predict):      #计算y_test和y_predict之间的R square
    return 1 - mse(y_test, y_predict) / np.var(y_test)
