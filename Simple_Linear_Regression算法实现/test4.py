import numpy as np
import matplotlib.pyplot as ply
from sklearn import datasets
from Simple_Linear_Regression import model_selection
from Simple_Linear_Regression import LinearRegression
from sklearn.linear_model import LinearRegression    #导入sklearn中的线性回归包
from sklearn.neighbors import KNeighborsRegressor    #导入knn算法中的回归算法
from sklearn.model_selection import GridSearchCV     #导入sklearn中的网格搜索
boston = datasets.load_boston() #导入波士顿数据集
X = boston.data
y = boston.target

X = X[y<50.0]                   #去除极端数据
y = y[y<50.0]

#分离测试集和训练集
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, seed = 666)

reg = LinearRegression()
reg.fit(X_train, y_train)            #训练数据集

coef = reg.coef_                            #系数
interception = reg.intercept_               #截距
score = reg.score(X_test, y_test)

print("多元线性回归的系数为 "+str(coef))             #系数为矩阵
print("多元线性回归的截距为 %f" %interception)
print("多元线性回归模型预测的准确率为 %f" %score)

lin_reg = LinearRegression()                         #实例化一个对象
lin_reg.fit(X_train, y_train)

lin_coef = lin_reg.coef_                             #系数
lin_intercept = lin_reg.intercept_                   #截距
lin_score = lin_reg.score(X_test, y_test)

print("使用sklearn得到的系数为 "+str(lin_coef))
print("使用sklearn得到的截距为 " +str(lin_intercept))
print("使用sklearn得到的score为 "+str(lin_score))


