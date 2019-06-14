import tensorflow as tf
from tensorflow.contrib.layers import flatten


class lenet():
    def __init__(self):
        self.x = tf.placeholder(tf.float32, shape=[None, 32, 32, 1], name="input_x")
        self.y = tf.placeholder(tf.float32, shape=[None, 10], name="input_y")
        self.loss, self.accuracy, self.fc1 = self.network()
        # self.optimizer = tf.train.AdamOptimizer(1e-4)
        self.optimizer = tf.train.GradientDescentOptimizer(1e-4)
        self.train_step = self.optimizer.minimize(self.loss)
        self._create_gradient_op()

    def conv_layer(self, input, channels_in, channels_out, name_):
        with tf.name_scope(name_):
            filter_shape = [5, 5, channels_in, channels_out]
            #initer = tf.truncated_normal_initializer(stddev=0.1)
            initer = tf.contrib.layers.xavier_initializer()
            with tf.variable_scope('conv_layer'):
                W = tf.get_variable(name_ + '_w', dtype=tf.float32, shape=filter_shape, initializer=initer)
                b = tf.get_variable(name_ + '_b', dtype=tf.float32, shape=[channels_out], initializer=initer)

            conv = tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding="VALID")
            act = tf.nn.relu(conv + b)
            #act = tf.sigmoid(conv + b)
            tf.summary.histogram("weights", W)
            tf.summary.histogram("biases", b)
            tf.summary.histogram("activations", act)
        return act, W, filter_shape

    def fc_layer(self,input, channels_in, channels_out, name_):
        with tf.name_scope(name_):
            w = tf.Variable(tf.truncated_normal([channels_in, channels_out], stddev=0.1), name=name_+"_w")
            b = tf.Variable(tf.constant(0.1, shape=[channels_out]), name=name_+"_b")
            act = tf.add(tf.matmul(input, w), b, name=name_+"act")
            tf.summary.histogram("weights", w)
            tf.summary.histogram("biases", b)
            tf.summary.histogram("activations", act)
            return act


    def pool(self, pool_input, k_size, pool_strides, scope_name):
        with tf.name_scope(scope_name):
            out = tf.nn.max_pool(pool_input, ksize=k_size, strides=pool_strides, padding='VALID', name='max_pool')
        return out

    # Create the network
    def network(self):
        # Setup placeholders, and reshape the data
        x_image = tf.reshape(self.x, [-1, 32, 32, 1])
        conv1, con_W1, filter_shape1 = self.conv_layer(x_image, 1, 6, "conv1")
        pool1 = self.pool(conv1, [1, 2, 2, 1], [1, 2, 2, 1], "Pool1")
        conv2, con_W2, filter_shape2 = self.conv_layer(pool1, 6, 16, "conv2")

        self.visual_kernel(con_W1, conv1, filter_shape1)
        self.visual_kernel(con_W2, conv2, filter_shape2)

        pool2 = self.pool(conv2, [1, 2, 2, 1], [1, 2, 2, 1], "Pool2")
        # conv3, _, _ = self.conv_layer(pool2, 16, 120, "conv3")
        # flattened = tf.reshape(conv3, [-1, 120])
        flattened = flatten(pool2)
        fc0 = self.fc_layer(flattened, 400, 120, "fc0")
        act_fc0 = tf.nn.relu(fc0, name= "act_fc0")
        #act_fc0 = tf.nn.sigmoid(fc0, name = "act_fc0")
        # fc1 = self.fc_layer(flattened, 120, 84, "fc1")
        fc1 = self.fc_layer(act_fc0, 120, 84, "fc1")
        act_fc1 = tf.nn.relu(fc1, name="act_fc1")
        #act_fc1 = tf.nn.sigmoid(fc1, name="act_fc1")


        fc2 = self.fc_layer(act_fc1, 84, 10, "fc2")

        y_predicted = tf.nn.softmax(fc2)
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y*tf.log(y_predicted)))
        # compute the accuracy
        correct_prediction = tf.equal(tf.argmax(y_predicted, 1), tf.argmax(self.y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.image('input', x_image, 10)
        tf.summary.scalar('cross_entropy', cross_entropy)
        tf.summary.scalar('accuracy', accuracy)
        return cross_entropy,accuracy, fc1

    def visual_kernel(self,conv_W, conv, filter_shape):
        with tf.name_scope('visual') as v_s:
            # scale weights to [0 1], type is still float
            x_min = tf.reduce_min(conv_W)
            x_max = tf.reduce_max(conv_W)
            kernel_0_to_1 = (conv_W - x_min) / (x_max - x_min)

            kernel_transposed = tf.transpose(kernel_0_to_1, [3, 2, 0, 1])
            conv_W_img = tf.reshape(kernel_transposed, [-1, filter_shape[0], filter_shape[1], 1])
            tf.summary.image('conv_w', conv_W_img, max_outputs=filter_shape[3])
            feature_img = conv[0:1, :, :, 0:filter_shape[3]]
            feature_img = tf.transpose(feature_img, perm=[3, 1, 2, 0])
            tf.summary.image('feature', feature_img, max_outputs=filter_shape[3])

    def _create_gradient_op(self):
        with tf.name_scope("gradient"):
            wparas = ['conv1_w', 'conv2_w']
            # draw gradient map
            with tf.variable_scope('conv_layer', reuse=True):
                for para in wparas:
                    w = tf.get_variable(para)
                    grad = self.optimizer.compute_gradients(self.loss, [w])
                    g = grad[0][0]
                    tf.summary.histogram("grad_" + para, g)

                    with tf.name_scope('visual') as v_sg:
                        # scale weights to [0 1], type is still float
                        x_min = tf.reduce_min(g)
                        x_max = tf.reduce_max(g)
                        g_0_to_1 = (g - x_min) / (x_max - x_min)
                        g_transposed = tf.transpose(g_0_to_1, [3, 2, 0, 1])

                        g_img = tf.reshape(g_transposed, [-1, g.get_shape()[0], g.get_shape()[1], 1])
                        tf.summary.image('g_conv_w', g_img, max_outputs=g.get_shape()[3] * g.get_shape()[2])

