# Created by Yuchen on 4/20/17.
import nltk
import numpy as np
from keras.models import Sequential
from keras.layers import Input, Embedding, LSTM, Dense, Dropout
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import reader

class VectorReader(object):

    def __init__(self, glove_file):
        '''
        Assumes a glove word vector file with 200 d vectors
        :param glove_file: 
        '''
        with open(glove_file) as f:
            all_words = f.read().split('\n')

            self._word_to_vec = {}
            for line in all_words:
                values = line.split()
                if len(values) < 200:
                    continue
                word = values[0]
                vector = values[1:]
                self._word_to_vec[word] = vector

            print("Successfully built dict with vocabulary: %d" % len(self._word_to_vec))

    def id_to_vec(self, word_to_id):
        pass

    @property
    def word_to_vec(self):
        return self._word_to_vec


class Config(object):
    """Medium config."""
    init_scale = 0.05
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 20
    sample_steps = 10
    hidden_size = 650
    max_epoch = 20
    max_max_epoch = 39
    keep_prob = 0.5
    lr_decay = 0.8
    batch_size = 128


# class WikiInput(object):
#   """The input data."""
#
#   def __init__(self, config, data, name=None):
#     self.batch_size = batch_size = config.batch_size
#     self.num_steps = num_steps = config.num_steps
#     self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
#     self.input_data, self.targets = reader.ptb_producer(
#         data, batch_size, num_steps, name=name)

class WikiReader(object):
    _unknown_token = "unk"
    _end_token = "eos"

    def __init__(self, file_path):
        with open(file_path) as f:
            data = f.read().replace("<unk>", self._unknown_token).replace("\n", self._end_token).lower().split()
            self.data = data

    def build_vocab(self, word_to_vec):
        for i in range(0, len(self.data)):
            if not self.data[i] in word_to_vec:
                self.data[i] = self._unknown_token

        word_freq = nltk.FreqDist(self.data)
        print("Found %d unique words tokens." % len(word_freq.items()))
        vocab = word_freq.most_common()  # using all words
        index_to_word = [(x[0], word_to_vec[x[0]]) for x in vocab]
        index_to_word, embedding_matrix = zip(*index_to_word)
        return index_to_word, embedding_matrix

    def train_data(self, config, index_to_word):
        word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])
        step_length = config.num_steps

        #  One fucking sub list has length 19 instead of 20 wasted 3 hours of my life!!!!!
        train_x = [[word_to_index[word] for word in self.data[i:i + step_length]]
                   for i in range(0, len(self.data) - config.sample_steps + 1, config.sample_steps)]

        train_y = [[word_to_index[word] for word in self.data[i:i + step_length]]
                   for i in range(1, len(self.data) - config.sample_steps + 1, config.sample_steps)]

        example_x = [index_to_word[wid] for wid in train_x[10]]
        example_y = [index_to_word[wid] for wid in train_y[10]]
        print("Example x: " + str(example_x))
        print("Example y: " + str(example_y))
        print("Traing sample: %d" % len(train_x))

        # deleting lists with wrong length
        for i in range(0, len(train_x)):
            if len(train_x[i]) != 20:
                del train_x[i]
                del train_y[i]

        train_y = [np_utils.to_categorical(l, num_classes=len(word_to_index)) for l in train_y]  # out of memory
        return np.array(train_x), np.array(train_y)

class WikiLanguageModel(object):

    def __init__(self, config, embedding_matrix):
        embedding_matrix = np.array(embedding_matrix)
        embedding_layer = Embedding(len(embedding_matrix),
                                    200,
                                    weights=[embedding_matrix],
                                    input_length=config.num_steps,
                                    trainable=False)

        model = Sequential()
        embedding_layer(Input(shape=(config.num_steps,), dtype='float16'))
        model.add(embedding_layer)
        model.add(LSTM(128, return_sequences=True))
        model.add(LSTM(128))
        model.add(Dropout(0.2))
        model.add(Dense(len(embedding_matrix), activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        self.model = model

    def fit(self, x, y, epochs=20, batch_size=128, verbose=2, callbacks=None):
        self.model.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=callbacks)

    def save(self):
        model_json = self.model.to_json()
        with open("models/wiki_word_model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights("models/wiki_word_model.h5")
        print("Saved model to disk")

    def generate(self, save_path):
        pass


def train():
    vector_reader = VectorReader("models/glove.6B.200d.txt")
    # train_data, word_to_id = reader.wiki_raw_data('data/wiki_data/wiki.test.tokens')
    # train_input = WikiInput(config=Config(), data=train_data, name="TrainInput")
    wiki_reader = WikiReader('data/wiki_data/wiki.test.tokens')
    index_to_word, embedding_matrix = wiki_reader.build_vocab(vector_reader.word_to_vec)
    config = Config()

    # define the checkpoint
    filepath = "checkpoints/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    x, y = wiki_reader.train_data(config, index_to_word)
    print(x.shape)
    print(y.shape)
    model = WikiLanguageModel(config=config, embedding_matrix=embedding_matrix)
    model.fit(x, y, epochs=config.max_epoch, batch_size=config.batch_size, callbacks=callbacks_list)
    model.save()


if __name__ == '__main__':
    train()
