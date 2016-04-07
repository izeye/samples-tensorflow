from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import threading
import time

from six.moves import xrange

import numpy as np
import tensorflow as tf

from tensorflow.models.embedding import gen_word2vec as word2vec

flags = tf.app.flags

flags.DEFINE_string("save_path", None, "Directory to write the model and training summaries.")
flags.DEFINE_string("train_data", None, "Training text file. "
                                        "E.g., unzipped file http://mattmahoney.net/dc/text8.zip.")
flags.DEFINE_string("eval_data", None, "File consisting of analogies of four tokens. "
                                       "embedding 2 - embedding 1 + embedding 3 should be close to embedding 4."
                                       "E.g. https://word2vec.googlecode.com/svn/trunk/questions-words.txt.")
flags.DEFINE_integer("embedding_size", 200, "The embedding dimension size.")

# epochs_to_train = 15
epochs_to_train = 2
flags.DEFINE_integer("epochs_to_train", epochs_to_train, "Number of epochs to train. "
                                                         "Each epoch processes the training data once completely.")
flags.DEFINE_float("learning_rate", 0.2, "Initial learning rate.")
flags.DEFINE_integer("num_neg_samples", 100, "Negative samples per training example.")
flags.DEFINE_integer("batch_size", 16, "Number of training examples processed per step (size of mini-batch)")
flags.DEFINE_integer("concurrent_steps", 12, "The number of words to predict to the left and right of the target word.")
flags.DEFINE_integer("window_size", 5, "The number of words to predict to the left and right of the target word.")
flags.DEFINE_integer("min_count", 5, "The minimum number of word occurrences for it to be included in the vocabulary.")
flags.DEFINE_float("subsample", 1e-3, "Subsample threshold for word occurrence. "
                                      "Words that appear with higher frequency will be randomly down-sampled. "
                                      "Set to 0 to disable.")
flags.DEFINE_integer("statistics_interval", 5, "Print statistics every n seconds.")
flags.DEFINE_integer("summary_interval", 5, "Save training summary to file every n seconds "
                                            "(rounded up to statistics interval).")
flags.DEFINE_integer("checkpoint_interval", 600, "Checkpoint the model (i.e. save the parameters) every n seconds "
                                                 "(rounded up to statistics interval).")
flags.DEFINE_boolean(
    "interactive", True, "If true, enters an IPython interactive session to play with the trained model. "
                         "E.g., try model.analogy('france', 'paris', 'russia') "
                         "and model.nearby(['proton', 'elephant', 'maxwell'])")
FLAGS = flags.FLAGS

class Options(object):
    def __init__(self):
        self.save_path = FLAGS.save_path
        self.train_data = FLAGS.train_data
        self.eval_data = FLAGS.eval_data
        self.emb_dim = FLAGS.embedding_size
        self.epochs_to_train = FLAGS.epochs_to_train
        self.learning_rate = FLAGS.learning_rate
        self.num_samples = FLAGS.num_neg_samples
        self.batch_size = FLAGS.batch_size
        self.concurrent_steps = FLAGS.concurrent_steps
        self.window_size = FLAGS.window_size
        self.min_count = FLAGS.min_count
        self.subsample = FLAGS.subsample
        self.statistics_interval = FLAGS.statistics_interval
        self.summary_interval = FLAGS.summary_interval
        self.checkpoint_interval = FLAGS.checkpoint_interval

class Word2Vec(object):
    def __init__(self, options, session):
        self._options = options
        self._session = session
        self._word2id = {}
        self._id2word = []
        self.build_graph()
        self.build_eval_graph()
        self.save_vocab()
        self._read_analogies()
        
    def build_graph(self):
        opts = self._options
        (words, counts, words_per_epoch, self._epoch, self._words, examples, labels)\
            = word2vec.skipgram(filename=opts.train_data,
                                batch_size=opts.batch_size,
                                window_size=opts.window_size,
                                min_count=opts.min_count,
                                subsample=opts.subsample)
        (opts.vocab_words, opts.vocab_counts, opts.words_per_epoch)\
            = self._session.run([words, counts, words_per_epoch])
        opts.vocab_size = len(opts.vocab_words)
        print("Data file:", opts.train_data)
        print("Vocabulary size:", opts.vocab_size - 1, "+ UNK")
        print("Words per epoch:", opts.words_per_epoch)
        self._examples = examples
        self._labels = labels
        self._id2word = opts.vocab_words
        for i, w in enumerate(self._id2word):
            self._word2id[w] = i
        true_logits, sampled_logits = self.forward(examples, labels)
        
        loss = self.nce_loss(true_logits, sampled_logits)
        
        tf.scalar_summary("NCE loss", loss)
        self._loss = loss
        self.optimize(loss)
        
        tf.initialize_all_variables().run()
        
        self.saver = tf.train.Saver()
    
    def forward(self, examples, labels):
        opts = self._options
        init_width = 0.5 / opts.emb_dim
        emb = tf.Variable(tf.random_uniform([opts.vocab_size, opts.emb_dim], -init_width, init_width), name="emb")
        self._emb = emb
        
        sm_w_t = tf.Variable(tf.zeros([opts.vocab_size, opts.emb_dim]), name="sm_w_t")
        sm_b = tf.Variable(tf.zeros([opts.vocab_size]), name="sm_b")
        
        self.global_step = tf.Variable(0, name="global_step")
        
        labels_matrix = tf.reshape(tf.cast(labels, dtype=tf.int64), [opts.batch_size, 1])
        
        sampled_ids, _, _ = tf.nn.fixed_unigram_candidate_sampler(
            true_classes=labels_matrix,
            num_true=1,
            num_sampled=opts.num_samples,
            unique=True,
            range_max=opts.vocab_size,
            distortion=0.75,
            unigrams=opts.vocab_counts.tolist())
        
        example_emb = tf.nn.embedding_lookup(emb, examples)
        
        true_w = tf.nn.embedding_lookup(sm_w_t, labels)
        true_b = tf.nn.embedding_lookup(sm_b, labels)
        
        sampled_w = tf.nn.embedding_lookup(sm_w_t, sampled_ids)
        sampled_b = tf.nn.embedding_lookup(sm_b, sampled_ids)
        
        true_logits = tf.reduce_sum(tf.mul(example_emb, true_w), 1) + true_b
        
        sampled_b_vec = tf.reshape(sampled_b, [opts.num_samples])
        sampled_logits = tf.matmul(example_emb, sampled_w, transpose_b=True) + sampled_b_vec
        return true_logits, sampled_logits
    
    def nce_loss(self, true_logits, sampled_logits):
        opts = self._options
        true_xent = tf.nn.sigmoid_cross_entropy_with_logits(true_logits, tf.ones_like(true_logits))
        sampled_xent = tf.nn.sigmoid_cross_entropy_with_logits(sampled_logits, tf.zeros_like(sampled_logits))
        nce_loss_tensor = (tf.reduce_sum(true_xent) + tf.reduce_sum(sampled_xent)) / opts.batch_size
        return nce_loss_tensor
    
    def optimize(self, loss):
        opts = self._options
        words_to_train = float(opts.words_per_epoch * opts.epochs_to_train)
        lr = opts.learning_rate * tf.maximum(0.0001, 1.0 - tf.cast(self._words, tf.float32) / words_to_train)
        self._lr = lr
        optimizer = tf.train.GradientDescentOptimizer(lr)
        train = optimizer.minimize(loss, global_step=self.global_step, gate_gradients=optimizer.GATE_NONE)
        self._train = train
    
    def build_eval_graph(self):
        analogy_a = tf.placeholder(dtype=tf.int32)
        analogy_b = tf.placeholder(dtype=tf.int32)
        analogy_c = tf.placeholder(dtype=tf.int32)
        
        nemb = tf.nn.l2_normalize(self._emb, 1)
        
        a_emb = tf.gather(nemb, analogy_a)
        b_emb = tf.gather(nemb, analogy_b)
        c_emb = tf.gather(nemb, analogy_c)
        
        target = c_emb + (b_emb - a_emb)
        
        dist = tf.matmul(target, nemb, transpose_b=True)
        
        _, pred_idx = tf.nn.top_k(dist, 4)
        
        nearby_word = tf.placeholder(dtype=tf.int32)
        nearby_emb = tf.gather(nemb, nearby_word)
        nearby_dist = tf.matmul(nearby_emb, nemb, transpose_b=True)
        nearby_val, nearby_idx = tf.nn.top_k(nearby_dist, min(1000, self._options.vocab_size))
        
        self._analogy_a = analogy_a
        self._analogy_b = analogy_b
        self._analogy_c = analogy_c
        self._analogy_pred_idx = pred_idx
        self._nearby_word = nearby_word
        self._nearby_val = nearby_val
        self._nearby_idx = nearby_idx
    
    def save_vocab(self):
        opts = self._options
        with open(os.path.join(opts.save_path, "vocab.txt"), "w") as f:
            for i in xrange(opts.vocab_size):
                f.write("%s %d\n" % (tf.compat.as_text(opts.vocab_words[i]), opts.vocab_counts[i]))
    
    def _read_analogies(self):
        questions = []
        questions_skipped = 0
        with open(self._options.eval_data, "rb") as analogy_f:
            for line in analogy_f:
                if line.startswith(b":"):
                    continue
                words = line.strip().lower().split(b" ")
                ids = [self._word2id.get(w.strip()) for w in words]
                if None in ids or len(ids) != 4:
                    questions_skipped += 1
                else:
                    questions.append(np.array(ids))
        print("Eval analogy file:", self._options.eval_data)
        print("Questions:", len(questions))
        print("Skipped:", questions_skipped)
        self._analogy_questions = np.array(questions, dtype=np.int32)
    
    def train(self):
        opts = self._options
        
        initial_epoch, initial_words = self._session.run([self._epoch, self._words])
        
        summary_op = tf.merge_all_summaries()
        summary_writer = tf.train.SummaryWriter(opts.save_path, self._session.graph)
        workers = []
        for _ in xrange(opts.concurrent_steps):
            t = threading.Thread(target=self._train_thread_body)
            t.start()
            workers.append(t)
        
        last_words, last_time, last_summary_time = initial_words, time.time(), 0
        last_checkpoint_time = 0
        while True:
            time.sleep(opts.statistics_interval)
            (epoch, step, loss, words, lr)\
                = self._session.run([self._epoch, self.global_step, self._loss, self._words, self._lr])
            now = time.time()
            last_words, last_time, rate = words, now, (words - last_words) / (now - last_time)
            print("Epoch %4d Step %8d: lr = %5.3f loss = %6.2f words/sec = %8.0f\r" % (
                epoch, step, lr, loss, rate), end="")
            sys.stdout.flush()
            if now - last_summary_time > opts.summary_interval:
                summary_str = self._session.run(summary_op)
                summary_writer.add_summary(summary_str, step)
                last_summary_time = now
            if now - last_checkpoint_time > opts.checkpoint_interval:
                self.saver.save(self._session, opts.save_path + "model", global_step=step.astype(int))
                last_checkpoint_time = now
            if epoch != initial_epoch:
                break
        for t in workers:
            t.join()
        return epoch
    
    def _train_thread_body(self):
        initial_epoch, = self._session.run([self._epoch])
        while True:
            _, epoch = self._session.run([self._train, self._epoch])
            if epoch != initial_epoch:
                break
    
    def eval(self):
        correct = 0
        
        total = self._analogy_questions.shape[0]
        start = 0
        while start < total:
            limit = start + 2500
            sub = self._analogy_questions[start:limit, :]
            idx = self._predict(sub)
            start = limit
            for question in xrange(sub.shape[0]):
                for j in xrange(4):
                    if idx[question, j] == sub[question, 3]:
                        correct += 1
                    elif idx[question, j] in sub[question, :3]:
                        continue
                    else:
                        break
        print()
        print("Eval %4d/%d accuracy = %4.1f%%" % (correct, total, correct * 100.0 / total))
    
    def _predict(self, analogy):
        idx, = self._session.run([self._analogy_pred_idx], {
            self._analogy_a: analogy[:, 0],
            self._analogy_b: analogy[:, 1],
            self._analogy_c: analogy[:, 2]
        })
        return idx
    
    def analogy(self, w0, w1, w2):
        wid = np.array([[self._word2id.get(w, 0) for w in [w0, w1, w2]]])
        idx = self._predict(wid)
        for c in [self._id2word[i] for i in idx[0, :]]:
            if c not in [w0, w1, w2]:
                return c
        return "unknown"
    
    def nearby(self, words, num=20):
        ids = np.array([self._word2id.get(x, 0) for x in words])
        vals, idx = self._session.run([self._nearby_val, self._nearby_idx], {self._nearby_word: ids})
        for i in xrange(len(words)):
            print("\n%s\n=====" % words[i])
            for (neighbor, distance) in zip(idx[i, :num], vals[i, :num]):
                print("%-20s %6.4f" % (self._id2word[neighbor], distance))

def _start_shell(local_ns=None):
    import IPython
    user_ns = {}
    if local_ns:
        user_ns.update(local_ns)
    user_ns.update(globals())
    IPython.start_ipython(argv=[], user_ns=user_ns)

def main(_):
    # print(FLAGS.train_data)
    
    if not FLAGS.train_data or not FLAGS.eval_data or not FLAGS.save_path:
        print("--train_data, --eval_data and --save_path must be specified.")
        sys.exit(1)
    opts = Options()
    with tf.Graph().as_default(), tf.Session() as session:
        with tf.device("/cpu:0"):
            model = Word2Vec(opts, session)
        for _ in xrange(opts.epochs_to_train):
            model.train()
            model.eval()
        model.saver.save(session, os.path.join(opts.save_path, "model.ckpt"), global_step=model.global_step)
        if FLAGS.interactive:
            _start_shell(locals())

if __name__ == "__main__":
    tf.app.run()