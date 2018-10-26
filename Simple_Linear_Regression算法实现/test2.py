import numpy as np
from Simple_Linear_Regression import Simple_Linear_regression2
#对于Simple_Linear_Regression类的测试

m = 1000000
big_x = np.random.random(size = m)                        #big_x为随机的1000000个x的值
big_y = big_x * 2.0 + 3.0 + np.random.normal(size = m)    #np.random.normal()为加进去均值为0,方差为1的干扰值

reg2 = Simple_Linear_regression2.Simple_Linear_Regression2()
reg2.fit(big_x,big_y)
print(reg2.a)
print("\n")
print(reg2.b)