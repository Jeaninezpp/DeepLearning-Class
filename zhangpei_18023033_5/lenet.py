# @Time     :
# @Author   :Jeanine ZP
# @File     :.py
'''
用slim 定义Lenet网络，并训练测试
要求：
1. Lenet单独定义到Lenet.py
2. 用with slim.arg_scope管理lenet中所有操作的默认参数，例如activation_fn, weights_initializer, 等。。。
'''
import tensorflow as tf
import tensorflow.contrib.slim as slim

def lenet5(inputs):
    inputs = tf.reshape(inputs, [-1, 28, 28, 1])
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                activation_fn=tf.nn.relu,
                weights_initializer=tf.truncated_normal_initializer(0.0, 0.1),
                weights_regularizer=slim.l2_regularizer(0.0005)):
        net = slim.conv2d(inputs, 32, [5, 5], padding='SAME', scope='l1-conv')
        net = slim.max_pool2d(net, 2, stride=2, scope='l2-maxpool')
        net = slim.conv2d(net, 64, [5, 5], padding='SAME', scope='l3-conv')
        net = slim.max_pool2d(net, 2, stride=2, scope='l4-max-pool')
        net = slim.flatten(net, scope='flatten')
        net = slim.fully_connected(net, 500, scope='l5-fc')
        net = slim.fully_connected(net, 10, scope='op')
    return net
