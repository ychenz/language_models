import numpy as np
import random
import tensorflow as tf
import math
import time

flags = tf.flags

flags.DEFINE_string(
    "mode", None,
    "Train or test the model. Options are: train, generate")

FLAGS = flags.FLAGS

class Config(object):
    len_per_section = 50
    skip = 3
    hidden_size = 256
    num_layers = 2
    batch_size = 1024
    max_steps = 500
    log_every = 5
    save_every = 2
    keep_prob = 0.5
    learning_rate = 0.001
    lr_decay = 0.95
    max_grad_norm = 5
    checkpoint_directory = 'checkpoints'


class CPUConfig(Config):
    batch_size = 70000


class GenConfig(Config):
    batch_size = 1


class WikiParser(object):
    cursor = 0

    def __init__(self, config):
        self.config = config
        data = open('data/wiki_data/wiki.test.small.tokens').read()
        # self.valid = open('data/wiki_data/wiki.valid.tokens').read()
        # self.test = open('data/wiki_data/wiki.test.tokens').read()
        chars = sorted(list(set(data)))
        self.char_size = len(chars)
        print(self.char_size)
        print(chars)

        self.char2id = dict((c, i) for i, c in enumerate(chars))
        self.id2char = chars

        # generating training data
        self.sections = []
        self.next_chars = []
        for i in range(0, len(data) - config.len_per_section, config.skip):
            self.sections.append(data[i:i + config.len_per_section])
            self.next_chars.append(data[i+1:i + config.len_per_section+1])

        print("Example X: " + str(self.sections[100]))
        print("Example Y: " + str(self.next_chars[100]))
        print("Training data size: ", len(self.sections))
        print("Steps per epoch: ", int(len(self.sections) / config.batch_size))

    def sample(self, prediction):
        r = random.uniform(0, 1)
        # store the prediction character
        s = 0
        char_id = len(prediction) - 1
        # for each char prediction probability
        for i in range(len(prediction)):
            s += prediction[i]
            if s >= r:
                char_id = i
                break
        char_one_hot = np.zeros(shape=[len(self.id2char)])
        char_one_hot[char_id] = 1.0
        return char_one_hot

    def next_batch(self):
        sections = self.sections
        next_chars = self.next_chars
        batch_size = self.config.batch_size
        if self.cursor + batch_size - 1 > len(sections):
            self.cursor = (self.cursor + batch_size - 1) - len(sections)
            return False

        # Vectorize input and output
        # Matrix of section length by num of characters
        x = np.zeros((batch_size, self.config.len_per_section, self.char_size))
        y = np.zeros((batch_size, self.config.len_per_section, self.char_size))

        # for each char in each section, convert each char to an ID
        # for each section convert the labels to ids
        for i, section in enumerate(sections[self.cursor:self.cursor + batch_size]):
            for j, char in enumerate(section):
                x[i, j, self.char2id[char]] = 1

        for i, next_char in enumerate(next_chars[self.cursor:self.cursor + batch_size]):
            for j, char in enumerate(next_char):
                y[i, j, self.char2id[char]] = 1

        self.cursor += self.config.batch_size
        return x, y


class WikiModel(object):
    '''
    Character level model that generates wikipedia articles
    '''

    def __init__(self, config, len_per_section, char_size, is_training):
        batch_size = config.batch_size
        hidden_size = config.hidden_size
        num_layers = config.num_layers

        with tf.variable_scope("LSTM"):
            with tf.variable_scope("Input"):
                self._inputs = tf.placeholder(tf.float16, shape=(batch_size, len_per_section, char_size), name="inputs")
                self._labels = tf.placeholder(tf.float16, shape=(batch_size, len_per_section, char_size), name="labels")
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, state_is_tuple=True)
            if is_training and config.keep_prob < 1:
                attn_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=config.keep_prob)
            else:
                attn_cell = lstm_cell
            cell = tf.contrib.rnn.MultiRNNCell([attn_cell for _ in range(num_layers)], state_is_tuple=True)

            self._initial_state = cell.zero_state(batch_size, tf.float16)
            outputs, state = tf.nn.dynamic_rnn(cell, self._inputs, initial_state=self._initial_state,)
            outputs = tf.reshape(outputs, [-1, hidden_size])
            softmax_w = tf.get_variable("softmax_w", [hidden_size, char_size], dtype=tf.float16)
            softmax_b = tf.get_variable("softmax_b", [char_size], dtype=tf.float16)
            logits = tf.matmul(outputs, softmax_w) + softmax_b

            with tf.variable_scope("Output"):
                self._predictions = tf.nn.softmax(logits, name="prediction")

            labels = tf.reshape(self._labels, [-1, char_size])
            self._cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                                labels=labels,
                                                                                ), name="cost")
            self._final_state = state

        if not is_training:
            return

        self._lr = tf.Variable(config.learning_rate, trainable=False)
        optimizer = tf.train.AdamOptimizer(epsilon=1e-4)  # slightly larger epsilon to avoid numerical instability with zero moments with float16
        self._train_op = optimizer.minimize(self._cost)
        tf.summary.scalar("Training Loss", self._cost)
        tf.summary.scalar("Learning Rate", self._lr)
        self._summary_op = tf.summary.merge_all()
        self._new_lr = tf.placeholder(
            tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    @property
    def inputs(self):
        return self._inputs

    @property
    def labels(self):
        return self._labels

    @property
    def predictions(self):
        return self._predictions

    @property
    def cost(self):
        return self._cost

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def summary_op(self):
        return self._summary_op

    @property
    def train_op(self):
        return self._train_op


def generating_text(model, parser, model_path='models/wiki'):
    checkpoint = tf.train.get_checkpoint_state(model_path)
    input_checkpoint = checkpoint.model_checkpoint_path
    saver = tf.train.Saver()

    test_start = 'I plan to make the world a better place '
    with tf.Session() as session:
        saver.restore(session, input_checkpoint)
        state = session.run(model.initial_state)

        for i in range(len(test_start) - 1):
            test_X = np.zeros((1, parser.char_size))
            # store it in id from
            test_X[0, parser.char2id[test_start[i]]] = 1.
            inputs = [[test_X[0]], ]
            output_prob, state = session.run([model.predictions, model.final_state],
                                       {model.inputs: inputs,
                                        model.initial_state: state})
        # where we store encoded char predictions
        test_X = np.zeros((1, parser.char_size))
        test_X[0, parser.char2id[test_start[-1]]] = 1.

        test_generated = ''
        test_generated += test_start
        # lets generate 500 characters
        for i in range(500):
            inputs = [[test_X[0]], ]
            # get each prediction probability
            output_prob, state = session.run([model.predictions, model.final_state],
                                             {model.inputs: inputs,
                                              model.initial_state: state})
            # one hot encode it
            # next_char_one_hot = model.sample(output_prob)
            # get the indices of the max values (highest probability)  and convert to char
            next_char = parser.id2char[np.argmax(output_prob[0])]
            # add each char to the output text iteratively
            test_generated += next_char
            # update the next char
            test_X = np.zeros((1, parser.char_size))
            test_X[0, parser.char2id[next_char]] = 1.

        print(test_generated)


def train():
    if not FLAGS.mode:
        raise ValueError("Must set --mode to specify a mode")
    mode = FLAGS.mode

    graph = tf.Graph()
    config = Config()
    gen_config = GenConfig()
    parser = WikiParser(config)
    parser.next_batch()
    with graph.as_default():
        initializer = tf.truncated_normal_initializer(stddev=math.sqrt(2 / parser.char_size))
        if mode == 'generate':
            with tf.name_scope("Generate"):
                with tf.variable_scope("Model", reuse=None, initializer=initializer):
                    gen_model = WikiModel(gen_config, 1, parser.char_size, False)
            generating_text(gen_model, parser)
            return

        with tf.name_scope("Train"):
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                model = WikiModel(config, config.len_per_section, parser.char_size, True)

        gpu_config = tf.ConfigProto()
        gpu_config.gpu_options.allow_growth = True
        save_path = '/home/tina/Scripts/python/RNN/checkpoints/model.ckpt'
        sv = tf.train.Supervisor(logdir='checkpoints', summary_op=None)
        with sv.managed_session(config=gpu_config) as session:
            session.run(model.initial_state)
            lr = config.learning_rate
            feed_dict = {}
            for step in range(config.max_steps):
                batch_cnt = 0
                start_time = time.time()
                costs = 0.0
                iters = 0

                while True:
                    session.run(model.initial_state)
                    data = parser.next_batch()
                    if not data:
                        break
                    else:
                        inputs, labels = data
                    feed_dict = {
                        model.inputs: inputs,
                        model.labels: labels
                    }
                    _, loss_value = session.run([model.train_op, model.cost], feed_dict)
                    batch_cnt += 1
                    # print("Cost at batch %d: %f" % (batch_cnt, loss_value))
                    costs += loss_value
                    iters += config.len_per_section

                # lr = lr * config.lr_decay
                # model.assign_lr(session, lr)

                print('training perplexity at step %d: %.2f speed: %.0f bpm cost: %.3f' %
                      (step, np.exp(costs/batch_cnt), batch_cnt/float((time.time() - start_time)/60.0), costs/batch_cnt))
                summaries = session.run(model.summary_op, feed_dict=feed_dict)
                sv.summary_computed(session, summaries)


                print("Saving model to %s" % save_path)
                sv.saver.save(session, save_path, global_step=step)
            print("Saving final model to %s" % save_path)
            sv.saver.save(session, save_path, global_step=sv.global_step)


def main(_):
    train()

if __name__ == '__main__':
    tf.app.run()
