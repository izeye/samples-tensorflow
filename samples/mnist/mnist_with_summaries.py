from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import input_data

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data for unit testing.')
flags.DEFINE_integer('max_steps', 1000, 'Number of steps to run trainer.')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')

def main(_):
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True, fake_data=FLAGS.fake_data)
    
    sess = tf.InteractiveSession()
    
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    W = tf.Variable(tf.zeros([784, 10]), name='weights')
    b = tf.Variable(tf.zeros([10]), name='bias')
    
    with tf.name_scope('Wx_b'):
        y = tf.nn.softmax(tf.matmul(x, W) + b)
    
    tf.histogram_summary('weights', W)
    tf.histogram_summary('biases', b)
    tf.histogram_summary('y', y)
    
    y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')
    
    with tf.name_scope('xent'):
        cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
        tf.scalar_summary('cross entropy', cross_entropy)
        
    with tf.name_scope('train'):
        train_step = tf.train.GradientDescentOptimizer(
            FLAGS.learning_rate).minimize(cross_entropy)
    
    with tf.name_scope('test'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.scalar_summary('accuracy', accuracy)
    
    merged = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter('MNIST_logs', sess.graph_def)
    tf.initialize_all_variables().run()
    
    for i in range(FLAGS.max_steps):
        if i % 10 == 0:
            feed = {x: mnist.test.images, y_: mnist.test.labels}
            summary_str, acc = sess.run([merged, accuracy], feed_dict=feed)
            writer.add_summary(summary_str, i)
            print('Accuracy at step %s: %s' % (i, acc))
        else:
            batch_xs, batch_ys = mnist.train.next_batch(100, fake_data=FLAGS.fake_data)
            feed = {x: batch_xs, y_: batch_ys}
            sess.run(train_step, feed_dict=feed)

if __name__ == '__main__':
    tf.app.run()
