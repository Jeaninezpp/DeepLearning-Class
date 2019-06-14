'''
title: fist homework of Dl class
date : 2019-4-25
goal :
	(1)RGB变为BRG，并显示图片
	(2)利用Numpy给改变通道顺序的图片中指定位置打上红框，其中红框左上角和右下角坐标定义方式为：18023033，则左上角坐标为(18, 02),右下角坐标为(18+30, 02+33)  (不可使用opencv中自带的画框工具）
	(3) 利用cv2.imwrite()函数保存加上红框的图片。
'''

import numpy as np
import cv2
from matplotlib import pyplot as plt
# 读取图像，cv2.imread默认读取的是BGR格式
img = cv2.imread('/home/zpp/Pictures/testrgb.jpg')

'''--------fun1--------
RGB变为BRG，并显示图片
--------fun1--------'''
# 方法一：使用cv2.split函数获取通道
b, g, r = cv2.split(img)
img_BRG_bysplit = cv2.merge([b, r, g])

# 方法二：使用数组索引获取通道
b1 = img[:, :, 0]  # blue
g1 = img[:, :, 1]  # green
r1 = img[:, :, 2]  # red
img_BRG_bynp = cv2.merge([b1, r1, g1])

'''---------------------fun2---------------------
	利用Numpy给改变通道顺序的图片中指定位置打上红框，
	其中红框左上角和右下角坐标定义方式为：
	18023033，则左上角坐标为(18, 02),
	右下角坐标为(18+30, 02+33),i.e.(48,35)
	(不可使用opencv中自带的画框工具）
---------------------fun2---------------------'''
im_frame = img_BRG_bysplit.copy()

# 修改指定位置的像素为红色，因为cv2为BGR模式读取，所以为(0,0,255)
im_frame[18:49, 2] = (0, 0, 255)
im_frame[18:49, 35] = (0, 0, 255)
im_frame[18, 2:36] = (0, 0, 255)
im_frame[48, 2:36] = (0, 0, 255)

# 保存加红框的照片
cv2.imwrite('im_frame.jpg', im_frame)


# 显示图片
cv2.imshow('img_original', img)
cv2.imshow('img_BRG_bysplit', img_BRG_bysplit)
cv2.imshow('img_BRG_bynp', img_BRG_bynp)
cv2.imshow('image_frame', im_frame)
k = cv2.waitKey(0)
if k == 27:
	cv2.destroyAllWindows()



