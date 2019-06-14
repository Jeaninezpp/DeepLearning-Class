'''
＠goal:设计变量共享网络进行MNIST分类
＠requirement:X_W需定义为一个函数，共享变量W name='w',一个输入参数X
＠预期结果：训练精度0.85左右
＠输出：可视化共享变量，训练测试精读，训练误差
＠AUTHOR:zp
'''

import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import matplotlib.pyplot as plt
import numpy as np

def X_W(X,reuse=False):
	with tf.variable_scope('mnclass') as scope:
		if reuse:
			scope.reuse_variables()
		weight = tf.get_variable(initializer=tf.truncated_normal_initializer(mean=0., stddev=0.02),
									shape=(392, 10),
							 		name='w')
	return weight,tf.matmul(X, weight)


mnist = input_data.read_data_sets('/home/zpp/download/file/MNIST/', one_hot=True)
#print(mnist.validation.num_examples)
#print(mnist.train.num_examples)
#print(mnist.test.num_examples)

'''定义网络'''
input1 = tf.placeholder(tf.float32,[None, 392])
input2 = tf.placeholder(tf.float32,[None, 392])
y_label = tf.placeholder(tf.float32,[None,10])

b = tf.Variable(tf.constant(0.0), dtype=tf.float32)

_,y1 = X_W(input1)
weight,y2 = X_W(input2,True)

y = tf.nn.softmax(y1+y2+b)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(y), reduction_indices=[1]))
optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
# 准确率
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_label, 1))  #返回行向最大值所在索引
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

'''训练网络'''
iteration = 500
batches = 512
n_batch = int(mnist.train.num_examples/batches)
print(n_batch)
loss = np.zeros([iteration])
acc  = np.zeros([iteration])

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(iteration):
	for batch in range(n_batch):
		b_x,b_y = mnist.train.next_batch(batches)
		_, loss[epoch], w = sess.run([optimizer, cross_entropy, weight],feed_dict={input1: b_x[:, 0:392],
										 						input2: b_x[:, 392:784],
										 						y_label: b_y})
	acc[epoch] = sess.run(accuracy,feed_dict={input1: mnist.test.images[:, 0:392],
										input2: mnist.test.images[:, 392:784],
										y_label: mnist.test.labels})
	if epoch % 10 == 0:
		print("%5d times iterations' accuracy is: %4f　,loss is: %4f" % (epoch, acc[epoch], loss[epoch]))

''' 评测网络'''
print('test:')
print('[accuracy,loss]:', sess.run([accuracy, cross_entropy],
								   feed_dict={input1: mnist.test.images[:, 0:392],
											  input2: mnist.test.images[:, 392:784],
											 y_label: mnist.test.labels}))


'''可视化
for i in range(1, 10):
	weight = w[:, i]
	weight = np.reshape(weight, [14, 28])
	plt.subplot(5, 2, i)
	plt.title(i)
	plt.imshow(weight)
plt.savefig('./weight.png')
'''
plt.figure(num=1)
plt.plot(np.arange(epoch+1), acc, '-', color = 'r', label = 'accuracy')
plt.title('accuracy')
plt.savefig('./acc.png')
plt.show()
'''
plt.figure(num=2)
plt.plot(np.arange(epoch+1), loss, color = 'b', label = 'loss')
plt.title('loss')
plt.savefig('./loss.png')
plt.show()
'''

