from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from lenet import Lenet

mnist = input_data.read_data_sets('/home/zpp/download/file/MNIST/', one_hot=True)
# 配置神经网络的参数
Batch_size = 100
LR_base = 0.01
LR_decay = 0.99
Regular_RATE = 0.0001
STEPS = 20000
MOVING_AVERAGE_DECAY = 0.99

if __name__ == '__main__':
    X_train = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='trainX')
    Y_train_label = tf.placeholder(tf.float32, shape=[None, 10], name='trainY')
    X_test = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='testX')
    Y_test_label = tf.placeholder(tf.float32, shape=[None, 10], name='testY')

    lenet = Lenet(0, 0.1, 0.5, X_train)
    Y_train = lenet.train_digits
    Y_test = lenet.test_digits

    global_step = tf.Variable(0, trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = Y_train, labels = tf.argmax(Y_train_label, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    loss = cross_entropy_mean
    learning_rate = tf.train.exponential_decay(LR_base, global_step, mnist.train.num_examples/Batch_size, LR_decay)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_opt = tf.no_op(name='train')

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(STEPS):
            xs, ys = mnist.train.next_batch(Batch_size)
            input_x, ys = shuffle(xs, ys)
            input_x = np.reshape(input_x, [-1, 28, 28, 1])
            _, loss_value, step, y = sess.run([train_opt, loss, global_step, Y_train], feed_dict={X_train: input_x, Y_train_label: ys})

            if (i+1) % 2000 == 0:
                x_val, y_val = mnist.validation.images, mnist.validation.labels
                x_val2 = np.reshape(x_val, [-1, 28, 28, 1])
                y_pred, loss_val = sess.run([Y_train, loss], feed_dict={X_train: x_val2, Y_train_label: y_val})
                y_pred2 = tf.nn.softmax(y_pred)
                correct_prediction = tf.equal(tf.arg_max(y_pred2, 1), tf.arg_max(y_val, 1))
                acc = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                print("step: {}, loss: {}, acc: {}".format(step, loss_val, acc.eval()))

        # 测试集acc
        print('-----------------------testing-----------------------')
        print("test acc:")
        x_test, y_test_label = mnist.test.images, mnist.test.labels
        x_test_input = np.reshape(x_test, [-1, 28, 28, 1])
        y_pred,lossvalue = sess.run([Y_train, loss], feed_dict={X_train: x_test_input, Y_train_label: y_test_label})
        y_test_pred = tf.nn.softmax(y_pred)
        correct_prediction = tf.equal(tf.argmax(y_test_pred, 1), tf.argmax(y_test_label, 1))
        acc = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(format(acc.eval()))

