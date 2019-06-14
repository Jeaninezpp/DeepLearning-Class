import tensorflow as tf
from tensorflow.contrib.layers import flatten


class Lenet(object):

    def __init__(self, mu, sigma, dropout, input_x ):
        self.mu = mu
        self.sigma = sigma
        self.dropout = dropout
        self.input_x = input_x

        with tf.variable_scope("Lenet_var") as scope:
            self.train_digits = self.net_build(isTrain = True)
            scope.reuse_variables()
            self.test_digits = self.net_build(isTrain = False)


    def convolutional_layer(self, X, kHeight, kWidth, channel, featureNum, mu, sigma, scope_name):
        #channel = int(X.get_shape()[-1])
        with tf.variable_scope(scope_name) as scope:
            conv_W = tf.Variable(tf.truncated_normal(
                shape=(kHeight, kWidth, channel, featureNum), mean=mu, stddev=sigma), name = 'conv_W')
            conv_b = tf.Variable(tf.zeros(featureNum), name = 'conv_b')
            conv_output = tf.nn.conv2d(X, conv_W, strides=[1, 1, 1, 1], padding='VALID') + conv_b
            return tf.nn.relu(conv_output)


    def pooling_layer(self, X, kHeight, kWidth, xStride, yStride, scope_name):
        with tf.variable_scope(scope_name) as scope:
             return tf.nn.max_pool(X, ksize=[1, kHeight, kWidth, 1], strides=[1, xStride, yStride, 1], padding='VALID')

    def fullyConnected_layer(self, X, inputD, outputD, mu, sigma, reluFlag, scope_name):
        with tf.variable_scope(scope_name) as scope:
                fc_W = tf.Variable(tf.truncated_normal(shape=(inputD, outputD), mean=mu, stddev=sigma))
                fc_b = tf.Variable(tf.zeros(outputD))
                fc = tf.matmul(X, fc_W) + fc_b
                if reluFlag:
                    return tf.nn.relu(fc)
                else:
                    return fc

    def dropout(self, X, keepPro, name = None):
        return tf.nn.dropout(X, keepPro, name)

    def net_build(self, isTrain=False):
        input = tf.pad(self.input_x, [[0, 0], [2, 2], [2, 2], [0, 0]], 'constant')
        conv1 = self.convolutional_layer(input, 5, 5, 1, 6, self.mu, self.sigma, "Layer_1_Convolutional")
        pool2 = self.pooling_layer(conv1, 2, 2, 2, 2, "Layer_2_Pooling")
        conv3 = self.convolutional_layer(pool2, 5, 5, 6, 16, self.mu, self.sigma, "Layer_3_Convolutional")
        pool4 = self.pooling_layer(conv3, 2, 2, 2, 2, "Layer_4_Pooling")
        # SOLUTION: Flatten. Input = 5x5x16. Output = 400.
        fcIn = flatten(pool4)
        fc5 = self.fullyConnected_layer(fcIn, 400, 120, self.mu, self.sigma, True, "Layer_5_FullyConnected")
        #dropout1 = self.dropout(fc5, self.dropout) #0.5?
        fc6 = self.fullyConnected_layer(fc5, 120, 84, self.mu, self.sigma, True, "Layer_6_FullyConnected")
        #dropout2 = self.dropout(fc6, self.dropout)  # 0.5?
        logits = self.fullyConnected_layer(fc6, 84, 10, self.mu, self.sigma, False, "Layer_7_FullyConnected")
        return logits

