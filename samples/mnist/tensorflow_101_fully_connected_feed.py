from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import tensorflow as tf

import input_data
import tensorflow_101_mnist as mnist

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 2000, 'Number of steps to run trainer.')
flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('batch_size', 100, 'Batch size. Must divide evenly into the data set sizes.')
flags.DEFINE_string('train_dir', 'MNIST_logs', 'Directory to put the training data.')
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data for unit testing.')

def placeholder_inputs(batch_size):
    images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, mnist.IMAGE_PIXELS))
    labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
    return images_placeholder, labels_placeholder

def fill_feed_dict(data_set, images_pl, labels_pl):
    images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size, FLAGS.fake_data)
    feed_dict = {
        images_pl: images_feed,
        labels_pl: labels_feed
    }
    return feed_dict

def do_eval(sess, eval_correct, images_placeholder, labels_placeholder, data_set):
    true_count = 0
    steps_pr_epoch = data_set.num_examples // FLAGS.batch_size
    num_examples = steps_pr_epoch * FLAGS.batch_size
    for step in xrange(steps_pr_epoch):
        feed_dict = fill_feed_dict(data_set, images_placeholder, labels_placeholder)
        true_count += sess.run(eval_correct, feed_dict=feed_dict)
    precision = true_count / num_examples
    print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' % (
        num_examples, true_count, precision))

def run_training():
    data_sets = input_data.read_data_sets('MNIST_data', FLAGS.fake_data)
    
    with tf.Graph().as_default():
        images_placeholder, labels_placeholder = placeholder_inputs(FLAGS.batch_size)
        
        logits = mnist.inference(images_placeholder, FLAGS.hidden1, FLAGS.hidden2)
        
        loss = mnist.loss(logits, labels_placeholder)
        
        train_op = mnist.training(loss, FLAGS.learning_rate)
        
        eval_correct = mnist.evaluation(logits, labels_placeholder)
        
        summary_op = tf.merge_all_summaries()
        
        saver = tf.train.Saver()
        
        sess = tf.Session()
        
        init = tf.initialize_all_variables()
        sess.run(init)
        
        summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, graph_def=sess.graph_def)
        
        for step in xrange(FLAGS.max_steps):
            start_time = time.time()
            
            feed_dict = fill_feed_dict(data_sets.train, images_placeholder, labels_placeholder)
            
            _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
            
            duration = time.time() - start_time
            
            if step % 100 == 0:
                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                summary_str = sess.run(summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
            
            if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                saver.save(sess, FLAGS.train_dir, global_step=step)
                print('Training Data Eval:')
                do_eval(sess, eval_correct, images_placeholder, labels_placeholder, data_sets.train)
                print('Validation Data Eval:')
                do_eval(sess, eval_correct, images_placeholder, labels_placeholder, data_sets.validation)
                print('Test Data Eval:')
                do_eval(sess, eval_correct, images_placeholder, labels_placeholder, data_sets.test)

def main(_):
    run_training()

if __name__ == '__main__':
    tf.app.run()