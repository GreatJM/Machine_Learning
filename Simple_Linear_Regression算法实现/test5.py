import numpy as np
import matplotlib.pyplot as ply
from sklearn import datasets
from sklearn.neighbors import KNeighborsRegressor    #导入knn算法中的回归算法
from sklearn.model_selection import GridSearchCV     #导入sklearn中的网格搜索
from Simple_Linear_Regression import model_selection
boston = datasets.load_boston() #导入波士顿数据集
X = boston.data
y = boston.target

X = X[y<50.0]                   #去除极端数据
y = y[y<50.0]

#分离测试集和训练集
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, seed = 666)
#使用knn回归算法拟合数据
knn_reg = KNeighborsRegressor()
knn_reg.fit(X_train,y_train)
score = knn_reg.score(X_test,y_test)
print("使用knn回归算法得到的score为" +str(score))
