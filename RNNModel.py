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

# Download nltk corpus and tagger
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
logging = tf.logging


class Config(object):
    vocabulary_size = 8000
    batch_size = 20
    embedding_size = hidden_size = 200  # size of word's embedding vector
    num_layers = 2  # not used, single layer for now
    lr_decay = 0.8
    keep_prob = 0.5  # rate of keeping a neuron for dropout operation
    init_scale = 0.1
    learning_rate = 0.005
    max_grad_norm = 10  #??
    max_epoch = 5


class RedditParser(object):
    vocabulary_size = 8000
    unknown_token = "UNK"
    sentence_start_token = "SEN_START"
    sentence_end_token = "SEN_END"

    def __init__(self,config,data_dir='data/reddit_data/'):
        self.data_dir = data_dir
        self.config = config
        self.vocabulary_size = config.vocabulary_size
        self.batch_size = config.batch_size
        self.index_to_word = None
        self.word_to_index = None
        self.X = None
        self.Y = None

        self.cursor = 0
        self.epoch = 0

        self.create_vocab()

    def load_word2vector(self):
        self.word2vector_model = gensim.models.KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True) # take 3GB of memory

    def tokenization(self,sentences_list):
        '''
        Break sentences into words
        :param sentences_list:
        :return: list of words
        '''
        # print("Tokenizing all sentences using all CPUs ...")
        pool = mp.Pool(processes=8)
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
        sentences = ["%s %s %s" % (self.sentence_start_token, x, self.sentence_end_token) for x in sentences]
        return sentences

    def create_vocab(self):
        vocabulary_size = self.vocabulary_size
        unknown_token = self.unknown_token
        sentence_start_token = self.sentence_start_token
        sentence_end_token = self.sentence_end_token

        all_sentences = []
        for path,subdir,files in os.walk(self.data_dir):
            # Parsing files
            for file in files:
                if file.endswith(".gz"):
                    file_path = os.path.join(path,file)
                    with gzip.open(file_path,'rt') as f:
                        raw_sententces = []
                        reader = csv.reader(f,skipinitialspace=True)
                        i = 0
                        for row in reader:
                            body = row[0]
                            raw_sententces.append(body.lower())
                            # i += 1
                            # if i >= 1000: # for testing the graph with small data
                            #     break
                        pool = mp.Pool(processes=8)
                        # split comments to sentences
                        all_sentences = pool.map(self._sent_tokenization, raw_sententces)
                        pool.close()
                        pool.join()

                    break # only read 1 file for now
            break
        # tokenize words in every sentence
        # example: ['SENTENCE_START', 'get', 'one', 'of', 'the', 'cheap', 'mustang', 'packages', '.', 'SENTENCE_END']
        # tokenized_sentences = [nltk.tokenize.word_tokenize(sent) for sent in all_sentences]
        all_sentences = list(itertools.chain.from_iterable(all_sentences[1:])) # expand inner lists to single list,exclude 1st elemtent（tag）
        tokenized_sentences = self.tokenization(all_sentences)

        # building dictionary for unique words and their count
        word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
        print("Found %d unique words tokens." % len(word_freq.items()))
        print("Using vocabulary size %d." % vocabulary_size)

        # only use the most frequent $vocabulary_size words
        vocab = word_freq.most_common(vocabulary_size-1)
        index_to_word = [unknown_token] # 'UNK' token at index[0]
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
        with open("models/word_labels.csv",'w+') as f:
            writer = csv.writer(f)
            for id_word in enumerate(index_to_word):
                writer.writerow(id_word)
            f.flush()

        self.index_to_word = index_to_word
        self.word_to_index = word_to_index
        self.X = X_train
        self.Y = y_train

    def shuffle(self):
        shuffled_indices = np.arange(len(self.X))
        np.random.shuffle(shuffled_indices)
        new_X = [ self.X[i] for i in shuffled_indices ]
        new_Y = [ self.Y[i] for i in shuffled_indices ]
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
        batch_x = [ self.X[i] for i in range(self.cursor,self.cursor + batch_size)]
        batch_y = [ self.Y[i] for i in range(self.cursor,self.cursor + batch_size)]
        self.cursor += batch_size

        # padding
        seq_len = []
        maxlen = max([len(sent) for sent in batch_x])
        for sent_x,sent_y in zip(batch_x,batch_y):
            seq_len.append(len(sent_x))
            diff = maxlen - len(sent_x)
            sent_x += [unk_idx] * diff

            # encode padding token to -1 for one-hot value of [0,0,0,0 ... 0], which means 0 possibility
            # which have the same value as the output of the dynamic RNN after the finish of the seq_len calculation
            sent_y += [unk_idx - 1] * diff

        return batch_x,batch_y,seq_len

    @property
    def get_word(self,id):
        return self.index_to_word[id]

    def featureSerializer(self):
        pass


class GenerativeGRUModel(object):

    def __init__(self,config,is_training):
        self.config = config
        state_size = self.config.hidden_size
        keep_prob = self.config.keep_prob
        vocabulary_size = self.config.vocabulary_size
        batch_size = self.config.batch_size

        self.seq_len = seq_len = tf.placeholder(tf.int32, shape=(batch_size,), name="seq_len")
        self.inputs = inputs = tf.placeholder(tf.int32, shape=(batch_size, None), name="inputs")
        self.labels = labels = tf.placeholder(tf.int32, shape=(batch_size, None), name="labels")

        # Building RNN graph
        GRUCell = tf.contrib.rnn.GRUCell(state_size)

        if is_training and keep_prob < 1:
            attn_cell = tf.contrib.rnn.DropoutWrapper(
                GRUCell, output_keep_prob=keep_prob)
            cell = attn_cell
        else:
            cell = GRUCell

        self._initial_state = cell.zero_state(batch_size, tf.float32)  # This created initial state across whole batch size,
        # equals op:  init_state = tf.get_variable('init_state', [1, hidden_size], initializer=tf.constant_initializer(0.0))
        # init_state = tf.tile(init_state, [batch_size, 1])

        # build word embedding layer
        with tf.device("/cpu:0"):
            embedding = tf.get_variable(
                "embedding", [vocabulary_size, state_size], dtype=tf.float32)
            inputs = tf.nn.embedding_lookup(embedding, inputs)

        if is_training and keep_prob < 1:
            inputs = tf.nn.dropout(inputs, keep_prob)

        #  forward pass
        # rnn_outputs shape: [batch_size x max(seq_len) x state_size]  State shape: [batch_size x state_size]
        with tf.variable_scope("RNN"):
            rnn_outputs, state = tf.nn.dynamic_rnn(cell, inputs, sequence_length=seq_len,
                                                         initial_state=self._initial_state)
        rnn_outputs = tf.reshape(tf.concat(axis=1, values=rnn_outputs), [-1, state_size])  # shape: [（batch_size*max(seq_len)）x state_size]
        softmax_w = tf.get_variable(
            "softmax_w", [state_size, vocabulary_size], dtype=tf.float32)
        softmax_b = tf.get_variable("softmax_b", [vocabulary_size], dtype=tf.float32)
        logits = tf.matmul(rnn_outputs, softmax_w) + softmax_b # logits shape: [batch_size*max(seq_len)）x state_size]

        # predict
        self._predictions = tf.nn.softmax(logits)

        # convert labels to one-hot vectors
        labels = tf.one_hot(tf.reshape(labels, [-1]),depth=vocabulary_size,axis=-1)
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels,name=None)
        self._cost = cost = tf.reduce_sum(loss) / tf.reduce_sum(tf.cast(seq_len, tf.float32))  # average loss across all valid predictions
        self._final_state = state

        if not is_training:
            return

        self._lr = tf.Variable(config.learning_rate, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), config.max_grad_norm)
        optimizer = tf.train.AdamOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(
            zip(grads, tvars))

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
    def lr(self):
        return self._lr


def main(_):
    config = Config()
    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-config.init_scale,config.init_scale)
        config.batch_size = 5
        config.vocabulary_size = 8000
        config.max_epoch = 5
        config.learning_rate = 0.007

        with tf.name_scope("Train"):
            reddit_parser = RedditParser(config)
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                model = GenerativeGRUModel(config, True)
            tf.summary.scalar("Training Loss", model.cost)

        # train the model
        start_time = time.time()
        loss_value = 0
        batch_cnt = 0
        sv = tf.train.Supervisor(logdir='checkpoints')
        feed_dict={}
        with sv.managed_session() as session:
            while reddit_parser.epoch < config.max_epoch:
                batch_data = reddit_parser.next_batch()  # generate batch of training data
                if batch_data is None:
                    print("Cost at epoch %d: %f" % (reddit_parser.epoch, loss_value))
                    print("Training time for epoch %d: %f seconds" % (reddit_parser.epoch, time.time() - start_time))
                    continue
                else:
                    x, y, seq_len = batch_data

                feed_dict = {
                    model.seq_len: seq_len,
                    model.inputs: x,
                    model.labels: y,
                }
                _, loss_value = session.run([model.train_op, model.cost], feed_dict=feed_dict)
                print("Cost at batch %d: %f" % (batch_cnt, loss_value))
                batch_cnt += 1

            loss_value = session.run(model.cost, feed_dict=feed_dict)
            save_path = './models/rnn-model'
            print("Saving model to %s" % save_path)
            sv.saver.save(session, save_path, global_step=sv.global_step)
        # state = session.run(model.initial_state)
        # pred = session.run(model.predictions)
        # cost = session.run(model.cost)
        # print("Cost: %f" % cost)
        # print(pred)

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
