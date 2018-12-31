import numpy as np
from itertools import product
from sklearn.utils import shuffle as sklearn_shuffle
from helpers import accuracy, pad, shuffle


class BasePerceptron:
    """
    This class represent the basic functionality of a perceptron class.

    If you extend this you must implement the train method.
    """

    def __init__(self, learning_rate=.01):
        self.learning_rate = learning_rate
        self.weights = None
        self.update_count = 0
        self.epoch_records = {}

    def train(self):
        raise NotImplementedError("IMPLEMENT THIS")

    def __repr__(self):
        return "{}(learning_rate={})".format(self.__class__.__name__, self.learning_rate)

    def __str__(self):
        return "{}".format(self.__class__.__name__)

    def predict(self, data):
        """
        This method takes in
        :param data: numpy array of each data point to make a prediction for. This array should already be padded with
        a 1's column in order to ensure a bias weight is taken into consideration. The number of the columns should
        also correspond with the number of weights.
        :return: numpy array containing predictions made for each data point given in the data array.
        """
        # Pad the data with an all ones vector.
        data = pad(data)

        # For each data point make a prediction and store it.
        preds = np.ndarray((len(data), 1))
        for ix, x in enumerate(data):
            preds[ix] = np.sign(np.dot(self.weights, x))

        return preds

class SimplePerceptron(BasePerceptron):
    """
    Implementation of the basic simple perceptron algorithm.
    """

    suggested_hyperparameters = {'learning_rate': [1, 0.1, 0.01]}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def train(self, data, labels, epochs=1, record_epochs=False, validation_set=None, epoch_steps=1):
        """
        This method runs a simple version of the perceptron algorithm on the data given.
        :param data: numpy array of each data point to be used for training. This array should already be padded with
        a 1's column in order to ensure a bias weight is included.
        :param labels: numpy array specify the labels {-1, 1}
        :param epochs: number of epochs to run.
        :record_epochs: If set to true will record weights, and accuracy after each epoch.
        :return: None
        """
        # Pad the data with an all ones vector.
        p_data = pad(data)

        # Initialize the weights.
        self.weights = np.random.uniform(low=-0.01, high=0.01, size=p_data.shape[1])
        self.update_count = 0

        # Go through epochs.
        for epoch in range(epochs):

            # Go through each data point.
            for x, y in zip(*shuffle(p_data, labels)):

                # If (w^t*x + b)*y < 0 make an update.
                if np.dot(self.weights, x) * y < 0:
                    # Update the weights
                    self.weights = self.weights + self.learning_rate * y * x

                    # Record update count.
                    self.update_count += 1

            # record epoch specific information if specified
            if record_epochs and epochs % 2 == 0:
                if validation_set is None:
                    val_x, val_y = data, labels
                else:
                    val_x, val_y = validation_set[0], validation_set[1]
                self.epoch_records[epoch + 1] = {'accuracy': accuracy(self.predict(val_x), val_y),
                                                 'weights': self.weights}


class DecayingPerceptron(BasePerceptron):
    """
    Implementation of the simple perceptron with decaying learning rate.
    """

    suggested_hyperparameters = {'learning_rate': [1, 0.1, 0.01]}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def train(self, data, labels, epochs=1, record_epochs=False, validation_set=None):
        """
        This method runs a simple version of the perceptron algorithm on the data given.
        :param data: numpy array of each data point to be used for training. This array should already be padded with
        a 1's column in order to ensure a bias weight is included.
        :param labels: numpy array specify the labels {-1, 1}
        :param epochs: number of epochs to run.
        :record_epochs: If set to true will record weights, and accuracy after each epoch.
        :return: None
        """
        # Pad the data with an all ones vector.
        p_data = pad(data)

        # Initialize the weights.
        self.weights = np.random.uniform(low=-0.01, high=0.01, size=p_data.shape[1])

        # Decaying Learning Rate
        t = 0

        for epoch in range(epochs):

            # Go through each data point.
            for x, y in zip(*shuffle(p_data, labels)):

                # If (w^t*x + b)*y < 0 make an update.
                if np.dot(self.weights, x) * y < 0:
                    # Calculate the decayed learning rate.
                    decayed_learning_rate = self.learning_rate / (1 + t)

                    # Update the weights
                    self.weights = self.weights + (self.decayed_learning_rate * y * x)

                    # Record update count.
                    self.update_count += 1

                # Increment t after each example not just mispredictions.
                t += 1

            # record epoch specific information if specified
            if record_epochs:
                val_x, val_y = validation_set[0], validation_set[1]
                self.epoch_records[epoch + 1] = {'accuracy': accuracy(self.predict(val_x), val_y),
                                                 'weights': self.weights}


class MarginPerceptron(BasePerceptron):
    """
    Implementation of the simple perceptron with margin update.
    """

    suggested_hyperparameters = {'learning_rate': [1, 0.1, 0.01],
                                 'margin': [1, 0.1, 0.01]}

    def __init__(self, margin=.01, **kwargs):
        super().__init__(**kwargs)
        self.margin = margin

    def train(self, data, labels, epochs=1, record_epochs=False, validation_set=None):
        """
        This method runs a simple version of the perceptron algorithm on the data given.
        :param data: numpy array of each data point to be used for training. This array should already be padded with
        a 1's column in order to ensure a bias weight is included.
        :param labels: numpy array specify the labels {-1, 1}
        :return: None
        """
        # Pad the data with an all ones vector.
        p_data = pad(data)

        # Initialize the weights.
        self.weights = np.random.uniform(low=-0.01, high=0.01, size=p_data.shape[1])

        # Decaying Learning Rate
        t = 0

        for epoch in range(epochs):

            # Go through each data point.
            for x, y in zip(*shuffle(p_data, labels)):

                # If (w^t*x + b)*y < margin make an update.
                if np.dot(self.weights, x) * y < self.margin:
                    # Calculate the decayed learning rate.
                    decayed_learning_rate = self.learning_rate / (1 + t)

                    # Update the weights
                    self.weights = self.weights + (decayed_learning_rate * y * x)

                    # Record update count.
                    self.update_count += 1

                # Increment t after each example not just mispredictions.
                t += 1

            # record epoch specific information if specified
            if record_epochs:
                val_x, val_y = validation_set[0], validation_set[1]
                self.epoch_records[epoch + 1] = {'accuracy': accuracy(self.predict(val_x), val_y),
                                                 'weights': self.weights}


class AveragedPerceptron(BasePerceptron):
    """
    Implementation of the simple perceptron with weight averaging.
    """

    suggested_hyperparameters = {'learning_rate': [1, 0.1, 0.01]}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.average_weights = None

    def train(self, data, labels, epochs=1, record_epochs=False, validation_set=None):
        """
        This method runs a simple version of the perceptron algorithm on the data given.
        :param data: numpy array of each data point to be used for training. This array should already be padded with
        a 1's column in order to ensure a bias weight is included.
        :param labels: numpy array specify the labels {-1, 1}
        :return: None
        """
        # Pad the data with an all ones vector.
        p_data = pad(data)

        # Initialize the weights and average weigths.
        self.weights = np.random.uniform(low=-0.01, high=0.01, size=p_data.shape[1])
        self.average_weights = self.weights

        for epoch in range(epochs):

            # Go through each data point.
            for x, y in zip(*shuffle(p_data, labels)):

                # If (w^t*x + b)*y < 0 make an update.
                if np.dot(self.weights, x) * y < 0:
                    # Update the weights
                    self.weights = self.weights + self.learning_rate * y * x

                    # Record update count.
                    self.update_count += 1

                # Increment the average weights even if no misprediction happens.
                self.average_weights = self.average_weights + self.weights

            # record epoch specific information if specified
            if record_epochs:
                val_x, val_y = validation_set[0], validation_set[1]
                # Set current weights to the averaged weights so predict will use them.
                temp_weights = self.weights
                self.weights = self.average_weights / (len(data) * epoch + 1)
                self.epoch_records[epoch + 1] = {'accuracy': accuracy(self.predict(val_x), val_y),
                                                 'weights': self.weights}
                # Set them back to resume normal algorithm operation.
                self.weights = temp_weights

        # Divide by the total number of examples it has seen.
        self.average_weights = self.average_weights / (len(data) * epochs)
        # Finally set the final weights to the average weights so they will be used for predictions.
        self.weights = self.average_weights


class AggressiveMarginPerceptron(BasePerceptron):
    """
    Implementation of the simple perceptron with aggresive margin update.
    """

    suggested_hyperparameters = {'margin': [1, 0.1, 0.01]}

    def __init__(self, margin=.01, **kwargs):
        super().__init__(**kwargs)
        self.margin = margin

    def train(self, data, labels, epochs=1, record_epochs=False, validation_set=None):
        """
        This method runs a simple version of the perceptron algorithm on the data given.
        :param data: numpy array of each data point to be used for training. This array should already be padded with
        a 1's column in order to ensure a bias weight is included.
        :param labels: numpy array specify the labels {-1, 1}
        :return: None
        """
        # Pad the data with an all ones vector.
        p_data = pad(data)

        # Initialize the weights.
        self.weights = np.random.uniform(low=-0.01, high=0.01, size=p_data.shape[1])

        for epoch in range(epochs):

            # Go through each data point.
            for x, y in zip(*shuffle(p_data, labels)):

                # If (w^t*x + b)*y < margin make an update.
                if np.dot(self.weights, x) * y < self.margin:
                    # Calculate the aggressive learning rate.
                    aggressive_learning_rate = (self.margin - (y * np.dot(self.weights, x))) / (np.dot(x, x) + 1)

                    # Update the weights
                    self.weights = self.weights + (aggressive_learning_rate * y * x)

                    # Record update count.
                    self.update_count += 1

            # record epoch specific information if specified
            if record_epochs:
                val_x, val_y = validation_set[0], validation_set[1]
                self.epoch_records[epoch + 1] = {'accuracy': accuracy(self.predict(val_x), val_y),
                                                 'weights': self.weights}
