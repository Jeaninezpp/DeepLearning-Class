# @Time     :
# @Author   :Jeanine ZP
# @File     :.py
"""
用slim 定义Lenet网络，并训练测试
编写mnist_train.py脚本，训练slim定义的lenet做MNIST字符分类。
	用以前的sess.run 去训练模型。
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow.contrib.slim as slim
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from lenet import lenet5

mnist = input_data.read_data_sets("./data", one_hot=True)


x       = tf.placeholder(tf.float32, [None, 784])
y_label = tf.placeholder(tf.float32, [None, 10])
y       = lenet5(x)
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_label, 1))

#loss = tf.reduce_mean(cross_entropy)
#train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

slim.losses.softmax_cross_entropy(y, y_label)
loss = slim.losses.get_total_loss()
tf.summary.scalar('losses/total_loss', loss)
optimizer = tf.train.GradientDescentOptimizer(0.001)
train_op = slim.learning.create_train_op(loss, optimizer)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(2000):
        xs, ys = mnist.train.next_batch(100)
        _, loss_value = sess.run([train_op, loss], feed_dict={x: xs, y_label: ys})
        if i % 100 == 0:
            print(i, "steps" "loss is:", loss_value)
    print('-----------------------testing-----------------------')
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print('acc:', sess.run(accuracy, feed_dict={x: mnist.test.images, y_label: mnist.test.labels}))
