# ckpt模式保存拟合的模型

# 将恢复模型参数w,b打印

# 绘制图像
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import math

# 相同步长采样2000个点
sample = 2000
Z = np.linspace(0, 2*np.pi, sample, endpoint=True)
x = (Z-33)/18
C = np.cos(Z)
a = random.uniform(0, 2*np.pi)
zero_to_a = a
a_to_2pi = 2*np.pi - a
sample_left = math.ceil(zero_to_a / (2*np.pi) * sample)
sample_right = sample - sample_left
data1 = np.linspace(0, a, sample_left, endpoint=False)
data2 = np.linspace(a, 2*np.pi, sample_right, endpoint=True)
dataz = np.concatenate((data1, data2), axis=0)
datay = np.cos(dataz)
datax = (dataz-33)/18

datax = datax.reshape(-1, 1)
datay = datay.reshape(-1, 1)
'''
w1 = tf.Variable(tf.random_normal([1, 1]), dtype=tf.float32)
w2 = tf.Variable(tf.random_normal([1, 1]), dtype=tf.float32)
w3 = tf.Variable(tf.random_normal([1, 1]), dtype=tf.float32)
bias = tf.Variable(tf.ones([1]), dtype=tf.float32)
saver = tf.train.Saver()
'''
with tf.Session() as sess:
	saver = tf.train.import_meta_graph('./store_ckpt/tf_fit.ckpt.meta')
	saver.restore(sess, tf.train.latest_checkpoint('./store_ckpt'))

	# 获取保存的变量
	print('recover from ckpt:')
	print('w1:', sess.run('w1:0'))
	print('w2:', sess.run('w2:0'))
	print('w3:', sess.run('w3:0'))
	print('bias:', sess.run('bias:0'))

	# 获取placeholder变量
	data = tf.get_default_graph().get_tensor_by_name('inputx:0')
	real_y = tf.get_default_graph().get_tensor_by_name('inputy:0')

	# 获取需要进行计算的operator
	y_label = tf.get_default_graph().get_tensor_by_name('op_to_store:0')

	test_set = sess.run(y_label, feed_dict={data: datax})


'''--------------------------------画图--------------------------------'''

plt.scatter(datax, datay, marker='+', color='y', label='sample points')
plot1 = plt.plot(datax, datay, '-', color='r', label='original')
plot2 = plt.plot(datax, test_set, 'r', color='b', label='poly_fit')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc=4)  # 指定legend的位置
plt.title('poly fitting recovered by ckpt')
plt.savefig('poly_fitting_recovery_ckpt.png')
plt.show()