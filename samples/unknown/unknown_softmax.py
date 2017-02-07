import time

import tensorflow as tf

import input_data

label_size = 2
learning_rate = 0.01

def main(_):
    start_time = time.time()

    data_sets = input_data.read_data_sets()

    with tf.Graph().as_default(), tf.Session() as session:
        dictionary_size = len(data_sets.dictionary)

        x = tf.placeholder(tf.float32, [None, dictionary_size])
        W = tf.Variable(tf.zeros([dictionary_size, label_size]))
        b = tf.Variable(tf.zeros([label_size]))
        y = tf.nn.softmax(tf.matmul(x, W) + b)

        y_ = tf.placeholder(tf.float32, [None, label_size])
        cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

        tf.initialize_all_variables().run()

        batch_count = 1000
        batch_size = 100
        # batch_size = 50
        for i in range(batch_count):
            batch_xs, batch_ys = data_sets.train.next_batch(batch_size)
            train_step.run({x: batch_xs, y_: batch_ys})

        correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print(accuracy.eval({x: data_sets.validation.inputs, y_: data_sets.validation.labels}))

    print("Elapsed time:", time.time() - start_time)

if __name__ == "__main__":
    tf.app.run()