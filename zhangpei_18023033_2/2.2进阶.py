'''
使用TensorFlow定义函数y = 1 - sin(x)/x, 并求解y达到最小值时对应的x
步骤提示：
(1) 使用TensorFlow给出函数定义
(2) 利用tf.train.GradientDescentOptimizer定义优化器
(3) 启动会话，用梯度下降算法训练一定的迭代次数，在每次迭代之后输出当前的x和y值
'''
import tensorflow as tf


x = tf.Variable(tf.constant(1, tf.float32))
y = 1 - tf.div(tf.sin(x), x)

train = tf.train.GradientDescentOptimizer(0.002).minimize(y)

init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)
	for i in range(10000):
		sess.run(train)
	print('当前最小值为：', sess.run(y), '对应x为:', sess.run(x))
