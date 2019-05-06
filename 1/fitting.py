'''------------------------------------------------------------------------------
title: fist homework-hard of Dl class
date : 2019-4-26
goal :
进阶题目：
假设有函数y = cos(ax + b),
其中a为学号前两位，b为学号最后两位。则我的函数为 y = cos(18x+33)
首先从此函数中以相同步长（点与点之间在x轴上距离相同），在0<(ax+b)<2pi范围内，采样出2000个点，
然后利用采样的2000个点作为特征点进行**三次函数**拟合。
请提交拟合的三次函数以及对应的图样（包括采样点及函数曲线）。
------------------------------------------------------------------------------'''
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import random
import math

# 函数
Z = np.linspace(0, 2*np.pi, 2000, endpoint=True)
x = (Z-33)/18
C = np.cos(Z)

'''------------------------采样2000个点------------------------'''
a = random.uniform(0, 2*np.pi)

zero_to_a = a
a_to_2pi = 2*np.pi - a
sample = 2000
sample_left = math.ceil(zero_to_a / (2*np.pi) * sample)
sample_right = sample - sample_left

data1 = np.linspace(0, a, sample_left, endpoint=False)
data2 = np.linspace(a, 2*np.pi, sample_right, endpoint=True)

dataz = np.concatenate((data1,data2),axis=0)
datay = np.cos(dataz)
datax = (dataz-33)/18

'''
print('a       = ', a)
print('0-a     = ', zero_to_a)
print('2pi - a = ', a_to_2pi)
print('sample_left:', sample_left)
print('sample_right:', sample_right)
print(data1.shape)
print(data2.shape)
print(datax.shape)
'''

'''--------------------------拟合--------------------------'''
y_fit = np.polyfit(datax, datay, 3)#用3次多项式拟合
polynomial = np.poly1d(y_fit)#多项式表达式
yvals = np.polyval(y_fit, datax)

'''--------------------------绘图--------------------------'''
plt.scatter(datax, datay, marker='x', color='y', label='sample points')

plot1 = plt.plot(datax, datay, '-', label='original values')
plot2 = plt.plot(datax, yvals, 'r', label='poly_fit values')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc=4)#指定legend的位置
plt.title('cos(18x+33) and it\' 3-times poly fitting curve')
plt.savefig('poly_fitting.png')
plt.show()
