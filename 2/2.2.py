import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# import tensorflow.contrib.eager as tfe

# tfe.enable_eager_execution()

datax = np.linspace(-100, 100, 2000, endpoint=True)
datax = datax.reshape(1, 2000)
datay = 1 - np.sin(datax)/datax

x = tf.placeholder(tf.float32, [1, 2000])
y = tf.placeholder(tf.float32, [1, 2000])

y_label = 1 - tf.sin(x)/x
# loss = tf.reduce_mean(tf.square(y - y_label))
train = tf.train.GradientDescentOptimizer(0.002).minimize(y_label)
# train = tf.train.AdamOptimizer(0.1).minimize(y)
init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	for i in range(1000):
		sess.run(train, feed_dict={x: datax, y: datay})
		print('i=', i, sess.run(y_label))
