import numpy as np
from Simple_Linear_Regression import model_selection
import matplotlib.pyplot as plt
from sklearn import datasets
from Simple_Linear_Regression import Simple_Linear_regression2
from Simple_Linear_Regression import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score   #从sklearn中导入R square的包

boston = datasets.load_boston()  #加载波士顿房价的数据集

x = boston.data[:,5]             #只取出5列
y = boston.target
np.max(y)                        #找到房价最大的那个数据
x = x[y<50.0]                    #观察图像可以看出有几个点都是50万元，去除波士顿房价中的极端数据
y = y[y<50.0]
# seed为随机数种子，测试集所占比例默认为0.2
x_train, x_test, y_train, y_test = model_selection.train_test_split(x,y,seed = 666)
reg = Simple_Linear_regression2.Simple_Linear_Regression2()  #实例化一个对象
reg.fit(x_train, y_train)

y_predict = reg.predict(x_test)
mse_error = metrics.mse(y_test, y_predict) #均方误差
rmse_error = metrics.rmse(y_test, y_predict)  #均方根误差
mae_error = metrics.mae(y_test, y_predict) #平均绝对误差
score = metrics.r2_sore(y_test, y_predict) #求出R的平方
score_1 = reg.score(x_test, y_test) #使用自己类中的score函数

print("均方误差为 %f" %mse_error)
print("均方根误差为 %f" %rmse_error)
print("平均绝对误差为 %f" %mae_error)
print("R square 为 %f" %score)
print("使用自己类得到的score为 %f" %score_1)

mse_ = mean_squared_error(y_test, y_predict) #使用sklearn中的mse
mae_ = mean_absolute_error(y_test, y_predict) #使用sklearn中的mae
score_ = r2_score(y_test, y_predict)
print("使用sklearn的均方差为 %f" %mse_)
print("使用sklearn中的平均绝对误差为 %f" %mae_)
print("使用sklearn的r square为 %f" %score_)