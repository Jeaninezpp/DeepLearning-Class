import numpy as np
import tensorflow as tf
from copy import deepcopy
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('/home/zpp/download/file/MNIST/', one_hot=False)
#mnist = input_data.read_data_sets('./data/mnist',one_hot=False)
#print(mnist.validation.num_examples)
#print(mnist.train.num_examples)
#print(mnist.test.num_examples)

def submodel(x):
    with tf.variable_scope('mnist', reuse=tf.AUTO_REUSE):
        W1 = tf.get_variable(initializer=tf.truncated_normal_initializer(mean=0., stddev=0.02), shape=(784, 500),
                             name='W1')
        b1 = tf.get_variable(initializer=tf.constant_initializer(value=0.), shape=[500], name='b1')
        W2 = tf.get_variable(initializer=tf.truncated_normal_initializer(mean=0., stddev=0.02), shape=(500, 10),
                             name='W2')
        b2 = tf.get_variable(initializer=tf.constant_initializer(value=0.), shape=[10], name='b2')

        l1 = tf.nn.relu(tf.matmul(x, W1) + b1)
        l2 = tf.nn.relu(tf.matmul(l1, W2) + b2)
        return l2


def model(input1, input2):
    with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
        g1 = submodel(input1)
        g2 = submodel(input2)
        sub1 = tf.subtract(g1, g2, name='sub_g')
        mul1 = tf.multiply(sub1, sub1, name='mul_sub')
        ew = tf.reduce_sum(mul1, axis=1, name='sum_mul')
        ew = tf.reshape(ew, (-1, 1))
        theta = tf.get_variable(initializer=tf.truncated_normal_initializer(0., 1.), shape=[1], name='thresh')
        pred = tf.nn.sigmoid(tf.subtract(ew, theta), name='pred')
        return pred


def prepare(input, label):
    label = np.reshape(label, (-1, 1))

    input1_1 = np.column_stack((input, label))
    input1_1 = input1_1.astype(np.float32)
    input2_1 = deepcopy(input1_1)
    np.random.shuffle(input2_1)

    input1_2 = input1_1[input1_1[:, -1].argsort()]
    input2_2 = input2_1[input2_1[:, -1].argsort()]

    input1 = np.row_stack((input1_1, input1_2))
    input2 = np.row_stack((input2_1, input2_2))
    return input1, input2


def ACC(predict, input1, input2):
    predict = predict > 0.5
    label = 1. - (input1[:, -1:] == input2[:, -1:])
    acc = predict == label
    acc = np.mean(acc)
    return acc


x1 = tf.placeholder(tf.float32, shape=[None, 784], name='x1')
x2 = tf.placeholder(tf.float32, shape=[None, 784], name='x2')
y1 = tf.placeholder(tf.float32, shape=[None, 1], name='y1')
y2 = tf.placeholder(tf.float32, shape=[None, 1], name='y2')
y = tf.subtract(1., tf.cast(tf.equal(y1, y2), dtype=tf.float32), name='y')

pred = model(x1, x2)
# loss = (1-y)*pred^2 + y*exp(-pred)
loss = tf.reduce_mean(
    tf.add(
        tf.multiply(tf.subtract(1., y), tf.multiply(pred, pred)),
        tf.multiply(y, tf.exp(-pred))
    ), name='loss')
optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

if __name__ == '__main__':
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for itera in range(20000):
        input, label = mnist.train.next_batch(64)
        input1, input2 = prepare(input, label)
        l, _ = sess.run([loss, optimizer],
                        feed_dict={x1: input1[:, :-1], x2: input2[:, :-1], y1: input1[:, -1:], y2: input2[:, -1:]})

        # 每隔2000个迭代，计算在验证集上的acc
        if (itera + 1) % 1000 == 0:
            input, label = mnist.validation.images, mnist.validation.labels
            input1, input2 = prepare(input, label)

            predict = sess.run(pred, feed_dict={x1: input1[:, :-1], x2: input2[:, :-1]})
            acc = ACC(predict, input1, input2)
            print(' %d times iteration: accuracy: %.4f , loss: %.4f' % (itera + 1, acc, l))

    # 计算测试集合上的acc
    print()
    input, label = mnist.test.images, mnist.test.labels
    input1, input2 = prepare(input, label)
    predict = sess.run(pred, feed_dict={x1: input1[:, :-1], x2: input2[:, :-1]})
    acc = ACC(predict, input1, input2)
    print('test_accuracy: %.4f' % (acc))

