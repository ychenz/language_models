# Created by Yuchen on 3/19/17.
import gensim
import nltk
import itertools
import tensorflow as tf
import gzip
import os
import csv
import multiprocessing as mp
import numpy as np
import time
import math

from RedditModelReader import freeze_graph, load_vocab

# Download nltk corpus and tagger
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
logging = tf.logging
flags = tf.flags

flags.DEFINE_string(
    "mode", None,
    "Train or test the model. Options are: train, generate")

FLAGS = flags.FLAGS


class Config(object):
    vocabulary_size = 8000
    batch_size = 50
    embedding_size = hidden_size = 200  # size of word's embedding vector
    num_layers = 2  # not used, single layer for now
    lr_decay = 0.5
    keep_prob = 0.5  # rate of keeping a neuron for dropout operation

    init_scale = 0.1  # random uniform initializer

    learning_rate = 0.5
    max_grad_norm = 5  # prevent exploding gradient issue
    max_epoch = 5

    def get_std(self):
        return math.sqrt(2 / self.vocabulary_size)


class GenConfig(Config):
    batch_size = 1


class RedditParser(object):
    vocabulary_size = 8000
    unknown_token = "UNK"
    sentence_start_token = "SEN_START"
    sentence_end_token = "SEN_END"

    def __init__(self, config, data_dir='data/reddit_data/'):
        self.data_dir = data_dir
        self.config = config
        self.vocabulary_size = config.vocabulary_size
        self.batch_size = config.batch_size
        self.index_to_word = None
        self.word_to_index = None
        self.X = None
        self.Y = None
        self.max_len = 0

        self.cursor = 0
        self.epoch = 0

        self.create_vocab()

    def load_word2vector(self):
        self.word2vector_model = gensim.models.KeyedVectors.load_word2vec_format(
            'data/GoogleNews-vectors-negative300.bin', binary=True)  # take 3GB of memory

    def tokenization(self, sentences_list):
        '''
        Break sentences into words
        :param sentences_list:
        :return: list of words
        '''
        # print("Tokenizing all sentences using all CPUs ...")
        pool = mp.Pool(processes=12)
        word_tokens = pool.map(nltk.tokenize.word_tokenize, sentences_list)
        pool.close()
        pool.join()
        return word_tokens

    def _sent_tokenization(self, body):
        '''
        Process method
        Separate comment to individual sentences
        :param body:
        :return: list of sentences
        '''
        # print("Tokenize comment to sentences")
        sentences = itertools.chain(*[nltk.sent_tokenize(body)])
        sentences = ["%s %s %s" % (self.sentence_start_token, x, self.sentence_end_token) for x in sentences if
                     len(x) < 300]  # 0.5% of sentence has length >300
        return sentences

    def create_vocab(self):
        vocabulary_size = self.vocabulary_size
        unknown_token = self.unknown_token
        sentence_start_token = self.sentence_start_token
        sentence_end_token = self.sentence_end_token

        all_sentences = []
        for path, subdir, files in os.walk(self.data_dir):
            # Parsing files
            for file in files:
                if file.endswith(".gz"):
                    file_path = os.path.join(path, file)
                    with gzip.open(file_path, 'rt') as f:
                        raw_sententces = []
                        reader = csv.reader(f, skipinitialspace=True)
                        i = 0
                        for row in reader:
                            body = row[0]
                            raw_sententces.append(body.lower())
                            i += 1
                            if i >= 50000:  # for testing the graph with small data
                                break
                        pool = mp.Pool(processes=12)
                        # split comments to sentences
                        all_sentences = pool.map(self._sent_tokenization, raw_sententces)
                        pool.close()
                        pool.join()

                    break  # only read 1 file for now
            break
        # tokenize words in every sentence
        # example: ['SENTENCE_START', 'get', 'one', 'of', 'the', 'cheap', 'mustang', 'packages', '.', 'SENTENCE_END']
        # tokenized_sentences = [nltk.tokenize.word_tokenize(sent) for sent in all_sentences]
        all_sentences = list(itertools.chain.from_iterable(
            all_sentences[1:]))  # expand inner lists to single list,exclude 1st elemtent（tag）
        tokenized_sentences = self.tokenization(all_sentences)

        # building dictionary for unique words and their count
        word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
        print("Found %d unique words tokens." % len(word_freq.items()))
        print("Using vocabulary size %d." % vocabulary_size)

        # only use the most frequent $vocabulary_size words
        vocab = word_freq.most_common(vocabulary_size - 1)
        index_to_word = [unknown_token]  # 'UNK' token at index[0]
        tmp_index_to_word = [x[0] for x in vocab]
        index_to_word += tmp_index_to_word
        word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])

        # Replace all words not in our vocabulary with the unknown token
        for i, sent in enumerate(tokenized_sentences):
            tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]

        print("Total training sentences: %d" % len(all_sentences))
        print("\nExample sentence: '%s'" % all_sentences[1])
        print("\nExample sentence after Pre-processing: '%s'" % tokenized_sentences[1])

        # Create the training data
        X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
        y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])

        print("\nExample training item: '%s'" % X_train[1])
        print("\nExample label: '%s'" % y_train[1])
        print("Generating words lookup table ...")
        if os.path.exists("models/word_labels.csv"):
            os.remove("models/word_labels.csv")
        with open("models/word_labels.csv", 'w+') as f:
            writer = csv.writer(f)
            for id_word in enumerate(index_to_word):
                writer.writerow(id_word)
            f.flush()

        self.index_to_word = index_to_word
        self.word_to_index = word_to_index
        self.max_len = len(max(X_train, key=len))
        self.X = X_train
        self.Y = y_train
        print("Max length: %d" % self.max_len)

    def shuffle(self):
        shuffled_indices = np.arange(len(self.X))
        np.random.shuffle(shuffled_indices)
        new_X = [self.X[i] for i in shuffled_indices]
        new_Y = [self.Y[i] for i in shuffled_indices]
        self.X = new_X
        self.Y = new_Y

    def next_batch(self):
        '''
        Return a batch of training data
        :return: training set , label set
        '''
        batch_size = self.batch_size
        if self.cursor + batch_size - 1 > len(self.X):
            self.epoch += 1
            self.shuffle()
            self.cursor = 0
            return None

        unk_idx = self.word_to_index[self.unknown_token]
        batch_x = [self.X[i] for i in range(self.cursor, self.cursor + batch_size)]
        batch_y = [self.Y[i] for i in range(self.cursor, self.cursor + batch_size)]
        self.cursor += batch_size

        # padding
        seq_len = []
        maxlen = self.max_len
        for sent_x, sent_y in zip(batch_x, batch_y):
            seq_len.append(len(sent_x))
            diff = maxlen - len(sent_x)
            sent_x += [unk_idx] * diff

            # encode padding token to -1 for one-hot value of [0,0,0,0 ... 0], which means 0 possibility
            # which have the same value as the output of the dynamic RNN after the finish of the seq_len calculation
            sent_y += [unk_idx - 1] * diff

        return batch_x, batch_y, seq_len

    @property
    def get_word(self, id):
        return self.index_to_word[id]

    def featureSerializer(self):
        pass


class RedditModel(object):
    def __init__(self, config, is_training, max_len):
        self.config = config
        state_size = self.config.hidden_size
        keep_prob = self.config.keep_prob
        vocabulary_size = self.config.vocabulary_size
        batch_size = self.config.batch_size

        self._seq_len = tf.placeholder(tf.int32, shape=(batch_size,),
                                       name="seq_len")  # holds length of each input sentence of the batch
        self._inputs = tf.placeholder(tf.int32, shape=(batch_size, max_len), name="inputs")
        self._labels = tf.placeholder(tf.int32, shape=(batch_size, max_len), name="labels")
        # these lines are causing memory fragmentation, variable shape is causing lots of tensor reallocation
        # self._inputs = tf.placeholder(tf.int32, shape=(batch_size, None), name="inputs")
        # self._labels = tf.placeholder(tf.int32, shape=(batch_size, None), name="labels")

        # Building RNN graph
        with tf.variable_scope("RNN"):
            GRUCell = tf.contrib.rnn.GRUCell(state_size)

        if is_training and keep_prob < 1:
            attn_cell = tf.contrib.rnn.DropoutWrapper(
                GRUCell, output_keep_prob=keep_prob)
            cell = attn_cell
        else:
            cell = GRUCell

        self._initial_state = cell.zero_state(batch_size,
                                              tf.float16)  # This created initial state across whole batch size,
        # equals op:  init_state = tf.get_variable('init_state', [1, hidden_size], initializer=tf.constant_initializer(0.0))
        # init_state = tf.tile(init_state, [batch_size, 1])

        # build word embedding layer
        with tf.device("/cpu:0"):
            embedding = tf.get_variable(
                "embedding", [vocabulary_size, state_size], dtype=tf.float16)
            inputs = tf.nn.embedding_lookup(embedding, self._inputs)

        if is_training and keep_prob < 1:
            inputs = tf.nn.dropout(inputs, keep_prob)

        # forward pass
        # rnn_outputs shape: [batch_size x max(seq_len) x state_size]  State shape: [batch_size x state_size]
        with tf.variable_scope("RNN"):
            rnn_outputs, state = tf.nn.dynamic_rnn(cell, inputs, sequence_length=self._seq_len,
                                                   initial_state=self._initial_state)
        rnn_outputs = tf.reshape(tf.concat(axis=1, values=rnn_outputs),
                                 [-1, state_size])  # shape: [（batch_size*max(seq_len)）x state_size]
        softmax_w = tf.get_variable(
            "softmax_w", [state_size, vocabulary_size], dtype=tf.float16)
        softmax_b = tf.get_variable("softmax_b", [vocabulary_size], dtype=tf.float16)
        logits = tf.matmul(rnn_outputs, softmax_w) + softmax_b  # logits shape: [batch_size*max(seq_len)）x state_size]

        # predict
        with tf.variable_scope("Output"):
            self._predictions = tf.nn.softmax(logits, name="Prediction")

        # convert labels to one-hot vectors
        labels = tf.one_hot(tf.reshape(self._labels, [-1]), depth=vocabulary_size, axis=-1)
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels, name=None)
        self._cost = cost = tf.reduce_sum(loss) / tf.reduce_sum(
            tf.cast(self._seq_len, tf.float16))  # average loss across all valid predictions
        self._final_state = state

        if not is_training:
            return

        self._lr = tf.Variable(config.learning_rate, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(
            self._lr)  # slightly larger epsilon to avoid numerical instability with zero moments with float16
        self._train_op = optimizer.apply_gradients(
            zip(grads, tvars))

        tf.summary.scalar("Training Loss", self._cost)
        tf.summary.scalar("Learning Rate", self._lr)
        self._summary_op = tf.summary.merge_all()

        self._new_lr = tf.placeholder(
            tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    @property
    def seq_len(self):
        return self._seq_len

    @property
    def input(self):
        return self._inputs

    @property
    def labels(self):
        return self._labels

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def predictions(self):
        return self._predictions

    @property
    def train_op(self):
        return self._train_op

    @property
    def summary_op(self):
        return self._summary_op

    @property
    def lr(self):
        return self._lr


def main(_):
    if not FLAGS.mode:
        raise ValueError("Must set --mode to specify a mode")
    mode = FLAGS.mode

    config = Config()
    gen_config = GenConfig()
    with tf.Graph().as_default():
        config.batch_size = 45  # This config takes about 1500M memory with max 300 sentence
        config.learning_rate = 0.5
        initializer = tf.truncated_normal_initializer(stddev=config.get_std())

        if mode == 'generate':
            with tf.name_scope("Generate"):
                with tf.variable_scope("Model", reuse=None, initializer=initializer):
                    gen_model = RedditModel(gen_config, False, 1)
            generating_text(gen_model)
            return

        with tf.name_scope("Train"):
            reddit_parser = RedditParser(config)
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                model = RedditModel(config, True, reddit_parser.max_len)

        # train the model
        start_time = time.time()
        batch_cnt = 0
        loss_value = 0
        prev_cost = 0

        gpu_config = tf.ConfigProto()
        gpu_config.gpu_options.allow_growth = True
        save_path = '/home/tina/Scripts/python/RNN/checkpoints/'
        sv = tf.train.Supervisor(logdir='checkpoints', summary_op=None)
        with sv.managed_session(config=gpu_config) as session:

            while reddit_parser.epoch < config.max_epoch:
                # if sv.should_stop():
                #     break

                batch_data = reddit_parser.next_batch()  # generate batch of training data
                if batch_data is None:
                    print("Cost at epoch %d: %f" % (reddit_parser.epoch, loss_value))
                    print("Training time for epoch %d: %f seconds" % (reddit_parser.epoch, time.time() - start_time))
                    lr_rate = config.learning_rate * config.lr_decay
                    print("Decreasing learning rate from %f to %f" % (config.learning_rate, lr_rate))
                    model.assign_lr(session, lr_rate)
                    config.learning_rate = lr_rate
                    continue
                else:
                    x, y, seq_len = batch_data

                session.run(model.initial_state)
                feed_dict = {
                    model.seq_len: seq_len,
                    model.input: x,
                    model.labels: y,
                }

                _, loss_value = session.run([model.train_op, model.cost], feed_dict=feed_dict)
                # print("Cost at batch %d: %f" % (batch_cnt, loss_value))
                batch_cnt += 1

                # adjusting learning rate if overshoot. This creates discrimination!
                # if prev_cost != 0 and loss_value - prev_cost > 2:
                #     print("Cost at batch %d: %f increased from %f" % (batch_cnt, loss_value, prev_cost))
                #     lr_rate = config.learning_rate * config.lr_decay
                #     print("Decreasing learning rate from %f to %f" % (config.learning_rate, lr_rate))
                #     model.assign_lr(session, lr_rate)
                #     config.learning_rate = lr_rate
                # prev_cost = loss_value

                if batch_cnt % 100 == 0 and batch_cnt != 0:
                    summaries = session.run(model.summary_op, feed_dict=feed_dict)
                    sv.summary_computed(session, summaries)
                    print("Cost at batch %d: %f" % (batch_cnt, loss_value))
                    print("Saving model to %s" % save_path)
                    sv.saver.save(session, save_path, global_step=sv.global_step)

            print("Saving model to %s" % save_path)
            sv.saver.save(session, save_path, global_step=sv.global_step)


def generating_text(model, n_sent=10, model_path='checkpoints'):
    checkpoint = tf.train.get_checkpoint_state(model_path)
    input_checkpoint = checkpoint.model_checkpoint_path
    saver = tf.train.Saver()

    def random_state():
        return np.random.uniform(-3, 3, (1, 200))

    with tf.Session() as session:
        saver.restore(session, input_checkpoint)
        index_to_word, word_to_index = load_vocab()
        state = random_state()
        mInput = np.matrix([[2]])  # 2 is the start token of a sentence

        text = ""
        count = 0
        while count < n_sent:
            output_prob, state = session.run([model.predictions, model.final_state],
                                             {model.input: mInput,
                                              model.initial_state: state,
                                              model.labels: np.matrix([[-1]]),
                                              model.seq_len: [1]})
            word_idx = np.argmax(output_prob[0])
            if word_idx == 0:
                # word_idx = np.argmax(output_prob[0][1:])
                word_idx = 245

            if word_idx != 1:  # end of a sentence
                word = index_to_word[word_idx]
                text += " %s" % word
                mInput = np.matrix([[word_idx]])
            else:
                text += "\n"
                mInput = np.matrix([[2]])
                state = random_state()
                count += 1

        print(text)

        # print("Generating model to model/reddit_model.pb")
        # freeze_graph("checkpoints")


def test_parser():
    config = Config()
    reddit_parser = RedditParser(config)
    print(len(reddit_parser.index_to_word))
    print(reddit_parser.index_to_word[config.vocabulary_size])

    print(reddit_parser.X[1])
    print(reddit_parser.Y[1])
    print("Shuffling ...")
    reddit_parser.shuffle()
    print(reddit_parser.X[1])
    print(reddit_parser.Y[1])

    x, y, seq_len = reddit_parser.next_batch()
    print(x)
    print(y)
    print(seq_len)
    print(reddit_parser.cursor)


if __name__ == "__main__":
    # test_parser()
    tf.app.run()
