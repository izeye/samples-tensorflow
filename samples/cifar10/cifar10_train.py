from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from datetime import datetime
import os.path

import numpy as np
from six.moves import xrange
import tensorflow as tf

import cifar10

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', 'cifar10_train', 'Directory where to write event logs and checkpoint.')
tf.app.flags.DEFINE_boolean('log_device_placement', False, 'Whether to log device placement.')
# tf.app.flags.DEFINE_integer('max_steps', 1000000, 'Number of batches to run.')
tf.app.flags.DEFINE_integer('max_steps', 2000, 'Number of batches to run.')

def train():
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)
        
        images, labels = cifar10.distorted_inputs()
        
        logits = cifar10.inference(images)
        
        loss = cifar10.loss(logits, labels)
        
        train_op = cifar10.train(loss, global_step)
        
        saver = tf.train.Saver(tf.all_variables())
        
        summary_op = tf.merge_all_summaries()
        
        init = tf.initialize_all_variables()
        
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement))
        sess.run(init)
        
        tf.train.start_queue_runners(sess=sess)
        
        summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, graph_def=sess.graph_def)
        
        for step in xrange(FLAGS.max_steps):
            start_time = time.time()
            _, loss_value = sess.run([train_op, loss])
            duration = time.time() - start_time
            
            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
            
            if step % 10 == 0:
                num_examples_per_step = FLAGS.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)
                
                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)')
                print(format_str % (datetime.now(), step, loss_value, examples_per_sec, sec_per_batch))
            
            if step % 100 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)
            
            if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

def main(argv=None):
    cifar10.maybe_download_and_extract()
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()

if __name__ == '__main__':
    tf.app.run()
