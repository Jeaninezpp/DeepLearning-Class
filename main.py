# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn.metrics as metrics
from tensorflow.examples.tutorials.mnist import input_data
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)

mnist = input_data.read_data_sets("C:/Users/ASUS/PycharmProjects/DL/W7/data/", one_hot=True)

def preprocess(images,labels,size):
    len_data = len(images)
    from_0_to_9 = [[],[],[],[],[],[],[],[],[],[]]
    for i in range(len_data):
        for j in range(10):
            if labels[i][j] == 1:
                from_0_to_9[j].append(images[i])
    image1 = []  # first img
    image2 = []  # second img
    label = []
    count_list_right = {}
    count_list_nega = {}
    # positive examples,each digital 1/10
    num_each_right = int(size/2/10)
    num_each_nega = int(size / 2 / 45)
    for i in range(10):
        isbreak = 0
        count =0
        len_i = len(from_0_to_9[i])
        for j in  range(len_i):
            if isbreak:
                break
            for k in range(len_i):
                if j==k :
                    continue
                image1.append(from_0_to_9[i][j])
                image2.append(from_0_to_9[i][k])
                label.append([0])
                count+=1
                if count>=num_each_right:
                    count_list_right["%d,%d"%(i,i)]=count
                    isbreak = 1
                    break

    # negative examples
    for i in range(10):
        for j in range(i+1,10):
            for count in range(num_each_nega):
                image1.append(from_0_to_9[i][count])
                image2.append(from_0_to_9[j][count])
                label.append([1])
            count_list_nega["%d,%d"%(i,j)]=num_each_nega

    data = []
    data.append(image1)
    data.append(image2)
    data.append(label)
    return data


def fully_connected_layer(scope_name,x,W_name,b_name,W_shape,reuse = False):
    with tf.variable_scope(scope_name) as scope:
        if reuse:
            scope.reuse_variables()
        fc_W = tf.get_variable(W_name, initializer=tf.truncated_normal(W_shape, stddev=0.1))
        fc_b = tf.get_variable(b_name, initializer=tf.zeros(W_shape[1]))
        tf.summary.histogram("weights", fc_W)
        tf.summary.histogram("biases", fc_b)
        fc = tf.matmul(x, fc_W) + fc_b
        return fc

def net(x,scope_name='net'):
    with tf.variable_scope(scope_name,reuse=tf.AUTO_REUSE):
        fc0 = fully_connected_layer("fc1",x,"fc1_w","fc1_b",[784,500])
        fc0 = tf.nn.sigmoid(fc0)
        fc1 = fully_connected_layer("fc2",fc0,"fc2_w","fc2_b",[500,10])
        fc1 = tf.nn.relu(fc1)
        return fc1

def model():
    Q = tf.constant([5.0])
    thresh = 2.5
    iterations = 1000
    lr = 0.01
    batch_size = 900

    x1 = tf.placeholder(tf.float32, [None, 784], name="x1")
    x2 = tf.placeholder(tf.float32, [None, 784], name="x2")
    y = tf.placeholder(tf.float32, [None,1], name="y")
    net1 = net(x1)
    net2 = net(x2)
    y_ = tf.reshape(y,[-1])

    Ew = tf.sqrt(tf.reduce_sum(tf.square(net1 - net2), 1))
    tf.summary.histogram("Ew", Ew)
    L1 = 2 * (1 - y_) * tf.square(Ew) / Q
    L2 = 2 * y_ * tf.exp(-2.77 * Ew / Q) * Q
    tf.summary.histogram("L1", L1)
    tf.summary.histogram("L2", L2)
    Loss = tf.reduce_mean(L1 + L2)
    tf.summary.scalar("loss",Loss)

    prediction = tf.greater(Ew, thresh)
    tf.summary.histogram("prediction", tf.cast(prediction,tf.int32))

    correct_prediction = tf.equal(prediction, tf.cast(y_, tf.bool))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accuracy", accuracy)

    train_step = tf.train.AdamOptimizer(lr).minimize(Loss)

    test_images = mnist.test.images
    test_labels = mnist.test.labels
    test_data = preprocess(test_images,test_labels,9000)

    image_x = []
    image_y_acc = []
    image_y_loss = []

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # tf.summary.image("input", lenet.x, 3)
        merged_summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter("LOGDIR/", sess.graph)

        for i in range(iterations):
            images,labels = mnist.train.next_batch(batch_size)
            data = preprocess(images,labels,batch_size)
            sess.run(train_step,feed_dict={x1:data[0],x2:data[1],y:data[2]})
            if i%10 == 0:
                loss,acc,s = sess.run([Loss,accuracy,merged_summary],feed_dict={x1:test_data[0],x2:test_data[1],y:test_data[2]})
                writer.add_summary(s, i)
                print("%4d iteration: acc: %4f , loss: %4f" % (i, acc, loss))
                image_x.append(i)
                image_y_acc.append(acc)
                image_y_loss.append(loss)
        e_w = sess.run(Ew, feed_dict={x1:test_data[0],x2:test_data[1],y:test_data[2]})

        fig = plt.figure()
        threshs = [0.5, 1., 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
        for thresh in threshs:
            predictions = (e_w > thresh).astype(np.int8)
            labels = sess.run(y, feed_dict={y:test_data[2]})
            precision, recall, th = metrics.precision_recall_curve(labels, predictions)
            plt.plot(precision, recall, linewidth=1.0,label='thresh='+str(thresh))
        print( '[accuracy,loss]:', sess.run([accuracy,Loss],feed_dict={x1:test_data[0],x2:test_data[1],y:test_data[2]}))

        plt.plot([0.5,1], [0.5,1], linewidth=1.0,label='equal')
        plt.title("P-R")
        plt.legend(loc='lower center')
        plt.xlabel("precision")
        plt.ylabel('recall')
        plt.show()

if __name__ =="__main__":
    model()