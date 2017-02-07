import numpy as np

train_small_data_set_filename = "train-data-set-small.txt"
test_small_data_set_filename = "test-data-set-small.txt"

train_large_data_set_filename = "train-data-set-large.txt"
test_large_data_set_filename = "test-data-set-large.txt"

label_size = 2

def features_to_matrix(features_set, dictionary):
    features_set_size = len(features_set)
    dictionary_size = len(dictionary)
    matrix = np.zeros(shape=(features_set_size, dictionary_size))
    for i in range(features_set_size):
        features = features_set[i]
        for feature in features:
            feature_id = dictionary[feature]
            matrix[i, feature_id] = 1
    return matrix


def labels_to_matrix(labels_set):
    labels_set_size = len(labels_set)
    matrix = np.zeros(shape=(labels_set_size, label_size))
    for i in range(labels_set_size):
        label = labels_set[i]
        matrix[i, label] = 1
    return matrix


class DataSet(object):
    def __init__(self, inputs, labels):
        self._inputs = inputs
        self._labels = labels
        self._num_examples = inputs.shape[0]
        self._index_in_epoch = 0
        self._epochs_completed = 0

    @property
    def inputs(self):
        return self._inputs

    @property
    def labels(self):
        return self._labels

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            self._epochs_completed += 1
            print("Completed epoch(s):", self._epochs_completed)
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._inputs = self._inputs[perm]
            self._labels = self._labels[perm]
            start = 0
            self._index_in_epoch = batch_size
        end = self._index_in_epoch
        return self._inputs[start:end], self._labels[start:end]


def read_data_sets():
    class DataSets(object):
        pass
    data_sets = DataSets()

    features_set = []
    labels_set = []

    count = 0
    # max_count = 100000
    max_count = 200000
    # FIXME: You will get `Killed: 9` with 300,000 samples.
    # max_count = 300000
    # train_data_set_filename = train_small_data_set_filename
    train_data_set_filename = train_large_data_set_filename
    with open(train_data_set_filename, "r") as f:
        for line in f:
            label_and_features = line.split("\t")
            label = label_and_features[0]
            features = label_and_features[1].split(",")
            features_set.append(features)
            labels_set.append(label)
            count += 1
            if count >= max_count:
                break
    train_data_set_size = len(features_set)
    print("Train data set size:", train_data_set_size)

    id = 0
    dictionary = {}
    for features in features_set:
        for feature in features:
            if not (feature in dictionary):
                dictionary[feature] = id
                id += 1
    dictionary_size = len(dictionary)
    print("Dictionary size:", dictionary_size)

    x = features_to_matrix(features_set, dictionary)
    print("x:", x)
    y_ = labels_to_matrix(labels_set)
    print("y_:", y_)

    validation_data_set_size = int(train_data_set_size / 5)
    train_x = x[validation_data_set_size:]
    train_y_ = y_[validation_data_set_size:]
    validation_x = x[:validation_data_set_size]
    validation_y_ = y_[:validation_data_set_size]
    print("Designated train data set size:", len(train_x))
    print("Designated validation data set size:", len(validation_x))

    data_sets.train = DataSet(train_x, train_y_)
    data_sets.validation = DataSet(validation_x, validation_y_)
    data_sets.dictionary = dictionary

    # FIXME: Implement me.
    data_sets.test = DataSets()
    return data_sets
