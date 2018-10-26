import numpy as np
import matplotlib.pyplot as plt
from Simple_Linear_Regression import Simple_Linear_regression
#对于Simple_linear_Regression类的测试
x = np.array([1.,2.,3.,4.,5.])
y = np.array([1.,3.,2.,3.,5.])

x_predict = 6.0
reg1 = Simple_Linear_regression.Simple_Linear_Regression()
reg1.fit(x,y)
y_predict = reg1.predict(np.array([x_predict]))
y_hat = reg1.predict(x)

plt.scatter(x,y)
plt.plot(x, y_hat, color="r")
plt.axis([0,6,0,6])
plt.show()
