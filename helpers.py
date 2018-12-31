import numpy as np
from itertools import product
import re


def accuracy(y_pred, y_true):
    """
    Simple metric to measure the accuracy between two arrays.
    :param y_pred: Numpy array.
    :param y_true: Numpy array.
    :return:
    """
    if y_pred.shape != y_true.shape:
        raise ValueError('Shapes must match')

    return np.sum(y_pred == y_true)/len(y_pred)


def pad(data, end=True):
    """
    Helper method to pad an array with a ones column.
    :param data: data to pand.
    :param end: boolean to determine if pad is added to end or on the front.
    :return: padded data array.
    """
    pad = np.ones((len(data), 1))
    if end:
        return np.hstack([data, pad])
    else:
        return np.hstack([pad, data])


def shuffle(x, y):
    """
    Takes in an x and y numpy array. Assuming they are of the same height dimension.
    Combines then shuffles then and seperates them.
    This ensure the correct labels stay together.
    """
    stacked_data = np.hstack([x, y.reshape(y.shape[0], 1)])
    np.random.shuffle(stacked_data)
    x, y = stacked_data[:, :-1], stacked_data[:, -1]
    return x, y


def read_lib_svm(filename, attribute_count, label_regex=r'\+\d|\-\d'):
    """
    Reads the lib_svm format and blases the values into a numpy array.
    This function makes heavy use of regex to parse the data.
    :return: Numpy array with information from file.
    """
    # Regex's needed
    value_regex = r'(\d+):(\d*\.?\d*)'

    # Open the file and read through each line.
    with open(filename) as f:

        # Need to count datapoints to see how big to make numpy array.
        row_count = 0
        for line in enumerate(f):
            if line != '':
                row_count += 1

        # Creating numpy arrays to store data.
        labels = np.ndarray(row_count)
        # Using zero so all values not set below are set to zero.
        data = np.zeros((row_count, attribute_count))

        print('There are {} data points in this file. Please verify manually.'.format(row_count))
        # Have to reset file to be able to read lines again.
        f.seek(0)
        for col_ix, line in enumerate(f):

            # There should only be one of these:
            label_matches = re.findall(label_regex, line)
            if len(label_matches) != 1:
                raise ValueError("Invalid Format. More than one label givn to single data point..")
            labels[col_ix] = label_matches[0]

            # Set all given data points. All other should be zero.
            value_matches = re.findall(value_regex, line)
            for att_index, value in value_matches:
                # att_index is 1-indexed need to transform to 0-indexed.
                data[col_ix][int(att_index) - 1] = float(value)

    return data, labels.reshape(labels.shape[0],1)


def enumerate_cross_val_params(possible_params):
    """
    This method takes in a dictionary with key name as cross validation parameter, and values as a list of possible
    values that paramerter can take and it enumerates all possible combinations.
    :param possible_params:
    :return:
    """
    enumerations = []
    for cross_val, params in possible_params.items():
        enumerations.append([v for v in params])
    enumerations = list(product(*enumerations))

    cvp = []
    for v in enumerations:
        cvp.append(dict(zip(possible_params.keys(), v)))
    return cvp


class CrossValidation:
    """
    This class represents an instance of a CrossValidation run.
    """

    def __init__(self, classifier, train_method, test_method, cv_splits, cross_val_params=None):
        """
        Set up methods to use when running your cross validation.
        :param classifier: Classifier class to use for cross validation. Assumes classifier take cross_val params
        that should also be provided.
        :param train_method: Method call on the untrained classifier and train split given to it.
        :param test_method: Method to call on trained classifier and test split.
        :param cv_splits: Splits to use. Should be tupe with train in 0 index and test at 1.
        :param cross_val_params: Kwargs to pass into the classifier class given.
        """
        self.clf = classifier
        self.train_method = train_method
        self.test_method = test_method
        self.cv_splits = cv_splits
        self.cross_val_params = cross_val_params

        self.cv_results = None

    def run(self):
        """
        Makes use of all the methods, and parameters given to the cross validation instance to actually
        perform the validation.
        :return: Returns a dictionary with the validations corresponding params as keys and the accuracy
        values for each different split and the classifier object.
        """
        self.cv_results = {}

        print("-------------------- Starting Cross Validation Run: {}-----------------".format(self.clf.__name__))

        for cv_param in self.cross_val_params:
            print('--------------------------------------------------------')
            print("Starting split with params {}".format(str(cv_param)))
            cv_clf = self.clf(**cv_param)

            cv_accuracy = np.zeros((len(self.cv_splits)))
            for ix, (train_split, test_split) in enumerate(self.cv_splits):
                print('\r' + 'On split: {} out of {}'.format(ix+1, len(self.cv_splits)), end="")
                self.train_method(cv_clf, train_split)
                cv_accuracy[ix] = self.test_method(cv_clf, test_split)

            self.cv_results[str(cv_param)] = {'accuracy_list': cv_accuracy, 'classifier': cv_clf}
            print('')
