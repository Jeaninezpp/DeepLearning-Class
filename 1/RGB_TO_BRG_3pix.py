'''
(1)RGB变为BRG，并显示图片
(2)利用Numpy给改变通道顺序的图片中指定位置打上红框，
其中红框左上角和右下角坐标定义方式为：18023033，则左上角坐标为(18, 02),右下角坐标为(18+30, 02+33)  (不可使用opencv中自带的画框工具）
(3) 利用cv2.imwrite()函数保存加上红框的图片。
'''

import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('/home/zpp/Pictures/testrgb.jpg')

b, g, r = cv2.split(img)
img_BRG_bysplit = cv2.merge([b, r, g])


im_frame = img_BRG_bysplit.copy()
# b r g
im_frame[18:49, 2] = (0, 0, 255)
im_frame[18:49, 35] = (0, 0, 255)
im_frame[18, 2:36] = (0, 0, 255)
im_frame[48, 2:36] = (0, 0, 255)




plt.subplot(131);plt.imshow(img);plt.title('original');
plt.subplot(132);plt.imshow(img_BRG_bysplit);plt.title('BRG_cv2.split');
plt.subplot(133);plt.imshow(im_frame);plt.title('BRG_frame');
plt.savefig("im_frame_plt.png")
plt.show()



