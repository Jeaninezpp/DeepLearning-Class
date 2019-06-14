import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from tensorflow.python.framework import graph_util


# 相同步长采样2000个点
sample = 2000
Z = np.linspace(0, 2*np.pi, sample, endpoint=True)
x = (Z-33)/18
C = np.cos(Z)
a = random.uniform(0, 2*np.pi)
zero_to_a = a
a_to_2pi = 2*np.pi - a
sample_left = math.ceil(zero_to_a / (2*np.pi) * sample)
sample_right = sample - sample_left
data1 = np.linspace(0, a, sample_left, endpoint=False)
data2 = np.linspace(a, 2*np.pi, sample_right, endpoint=True)
dataz = np.concatenate((data1, data2), axis=0)
datay = np.cos(dataz)
datax = (dataz-33)/18

datax = datax.reshape(-1, 1)
datay = datay.reshape(-1, 1)


# 定义占位符
data = tf.placeholder(tf.float32, [None, 1], name='inputx')
real_y = tf.placeholder(tf.float32, [None, 1], name='inputy')

# 定义变量
w1 = tf.Variable(tf.random_normal([1, 1]), dtype=tf.float32, name='w1')
w2 = tf.Variable(tf.random_normal([1, 1]), dtype=tf.float32, name='w2')
w3 = tf.Variable(tf.random_normal([1, 1]), dtype=tf.float32, name='w3')
bias = tf.Variable(tf.ones([1]), dtype=tf.float32, name='bias')

# 定义图
y_lable = tf.add(tf.add(tf.add(tf.matmul(data, w1), tf.matmul(data**2, w2)),
                    tf.matmul(data**3, w3)),
            bias, name='op_to_store')
# 定义loss和优化器
loss = tf.reduce_mean(tf.square(real_y - y_lable))
learning_rate = 0.2
# train = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
train = tf.train.AdamOptimizer(learning_rate).minimize(loss)


# ckpt:只保存四个最新的模型
saver = tf.train.Saver(max_to_keep=4)



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(30000):
        sess.run(train, feed_dict={data: datax, real_y: datay})
        if (i % 1000 == 0):
            #print('----', i, 'epoch ----')
            #print('y_lab:', sess.run(y_lable, feed_dict={data: datax}))
            print(i, ' loss :', sess.run(loss, feed_dict={data: datax, real_y: datay}))
    print('w1:', sess.run(w1), 'w2:', sess.run(w2),
          'w3:', sess.run(w3), 'bias:', sess.run(bias))
    test_set = sess.run(y_lable, feed_dict={data: datax})
    '''---------------------------------------------存储模型---------------------------------------------'''
    #　ckpt:设置存储步长，每1000个迭代保存一次，过程中可以不更新meta文件
    # save_path = saver.save(sess, './store_ckpt/tf_fit.ckpt', global_step=1000, write_meta_graph=False)
    save_path = saver.save(sess, './store_ckpt/tf_fit.ckpt')

    # pb
    graph_def = tf.get_default_graph().as_graph_def()
    constan_graph = graph_util.convert_variables_to_constants(sess, graph_def, ['op_to_store'])
    # 写入序列化的pb文件
    with tf.gfile.FastGFile('./store_pb/tf_fit.pb', mode='wb') as f:
        f.write(constan_graph.SerializeToString())
    '''---------------------------------------------存储模型---------------------------------------------'''


'''--------------------------------画图--------------------------------'''
plt.scatter(datax, datay, marker='+', color='c', label='sample points')
plot1 = plt.plot(datax, datay, '-', color='r', label='original')
plot2 = plt.plot(datax, test_set, 'r', color='b', label='poly_fit')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc=4)  # 指定legend的位置
plt.title('cos(18x+33) and it\' 3-times poly fitting curve by tf')
plt.savefig('poly_fitting.png')
plt.show()
