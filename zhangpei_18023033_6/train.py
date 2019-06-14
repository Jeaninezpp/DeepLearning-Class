import os
import numpy as np
from lenet_sigmoid import lenet
# from lenet_relu import lenet
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

logdir = "lenet_dir"
mnist = input_data.read_data_sets("../exercise/tensorboard/mnist", one_hot=True)
net = lenet()
ms = tf.summary.merge_all()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

writer = tf.summary.FileWriter(logdir)
writer.add_graph(sess.graph)

for i in range(20000):
    batch = mnist.train.next_batch(100)
    input_x = batch[0].reshape((-1, 28, 28, 1))
    x_train = np.pad(input_x, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
    x_train = x_train.reshape((-1, 32, 32, 1))
    if i % 1000 == 0:
        s = sess.run(ms, feed_dict={net.x: x_train, net.y: batch[1]})
        writer.add_summary(s, i)
        Accuracy = sess.run(net.accuracy, feed_dict={net.x: x_train, net.y: batch[1]})
        print("step %d, Training accuracy %g" % (i, Accuracy))
    # Run the training step
    sess.run(net.train_step, feed_dict={net.x: x_train, net.y: batch[1]})



x_100 = np.zeros([10, 100, 784])
y_100 = np.zeros([10, 100, 10])
for m in range(10):
    for n in range(100):
        batch_x, batch_y = mnist.train.next_batch(1)
        if batch_y[0][m] == 1:
            x_100[m][n] = batch_x
            y_100[m][n] = batch_y
x_100 = np.pad(x_100.reshape(1000, 28, 28), ((0, 0), (2, 2), (2, 2)),
               'constant').reshape(10, 100, 1, 32, 32, 1)
y_100 = y_100.reshape(10, 100, 1, 10)


plt.figure('fc2 featuremap')
fc2 = np.zeros([10, 100, 84])
for i in range(10):
    for j in range(100):
        fc2[i][j] = sess.run(net.fc1, feed_dict={net.x: x_100[i][j],
                                                 net.y: y_100[i][j]})
fc2min, fc2max = fc2.min(), fc2.max()
fc2 = (fc2-fc2min)/(fc2max-fc2min)
for r in range(10):
    plt.subplot(2, 5, r+1)
    plt.imshow(fc2[r])
plt.show()
