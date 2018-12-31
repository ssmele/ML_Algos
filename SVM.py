from sklearn.utils.validation import check_X_y, check_array, check_is_fitted, check_array
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

import numpy as np

class BaseSGD:
    """
    This class represent the basic functionality of a SGD class.

    If you extend this you must implement the fit method.
    """

    def __init__(self, verbose=0, weights='uniform'):
        self.verbose = verbose

    def fit(self):
        raise NotImplementedError("IMPLEMENT THIS")
        
    def get_params(self, deep=False):
        raise NotImplementedError("IMPLEMENT THIS")
        
    def set_params(self, **params):
        raise NotImplementedError("IMPLEMENT THIS")

    def __repr__(self):
        return "{}(learning_rate={})".format(self.__class__.__name__, self.learning_rate)

    def __str__(self):
        return "{}".format(self.__class__.__name__)

    def v_print(self, *args, **kwargs):
        """
        Method to help with debugging.
        :param args: Args to print out.
        :param kwargs: Kwargs to print out
        """
        if self.verbose == 0:
            pass
        if self.verbose == 1:
            print(*args, **kwargs)

    def predict(self, X, weights_to_use=None):
        """
        Make predictions from current weight vector or of given weights.
        :param X: data to make predictions on
        :param weights_to_use: Weights to use for prediction.
        :return: predictions made.
        """
        # Need to perform checks so sklearn and I dont get frustrated.
        X = check_array(X)
        check_is_fitted(self, 'is_fitted_')
        
        if weights_to_use is not None:
            weights = weights_to_use
        else:
            weights = self.weights_

        preds = np.ndarray(len(X))
        for ix, x in enumerate(X):
            preds[ix] = np.sign(np.dot(weights, x))
        return preds


class StochasticGDSVM(BaseSGD):
    
    suggested_hyperparameters = {
        'learning_rate': (10**0, 10**-1, 10**-2, 10**-3, 10**-4),
        'C' : (10**1, 10**0, 10**-1, 10**-2, 10**-3, 10**-4),
        'epochs': (10, 30, 50, 75, 100)
        #'verbose': (1,)
    }

    def __init__(self, learning_rate=.01, C=.01, epochs=10, **kwargs):
        super().__init__(**kwargs)
        self.C = C
        self.learning_rate = learning_rate
        self.epochs = epochs
        
    def get_params(self, deep=True):
        return {'learning_rate': self.learning_rate, 'C': self.C}
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
    def fit(self, X, y, validation_sets=None):
        """
        This method performs the training for the Stochastic Gradient Descent with SVM sub gradient loss function.
        :param X: Training data.
        :param y: Labels for the training data.
        :param epochs: Number of epochs to run .
        :return: Newly trained model.
        """
        X, Y = check_X_y(X, y)
        
        # Initialize the weights.
        self.weights_ = np.random.uniform(low=-0.01, high=0.01, size=X.shape[1])
        
        # Dict for storing epoch info. 
        self.epoch_records = {}

        # example counter used for learning rate calculation.
        s = 0
        
        for epoch in range(self.epochs):
            self.v_print("Epoch: {}/{}".format(epoch+1, self.epochs))

            # Go through each data point.
            for x, y in zip(*shuffle(X, Y)):

                # Calculate learning rate based on step
                learning_rate = self.learning_rate/(1+s)

                # Change weights based on subgradient calculation
                if y*np.dot(self.weights_, x) <= 1:
                    self.weights_ = self.weights_ - learning_rate*self.weights_ + learning_rate*self.C*y*x
                else:
                    self.weights_ = self.weights_ - learning_rate*self.weights_

            s += 1

            # If we were given a validation set calculate current models accuracy on it.
            if validation_sets is not None:
                self.is_fitted_ = True
                self.epoch_records[epoch] = {'weights': self.weights_.copy()}
                for ix, validation_set in enumerate(validation_sets):
                    preds = self.predict(validation_set[0])
                    self.epoch_records[epoch]['val_set_{}_acc'.format(ix)] = accuracy_score(validation_set[1], preds)
                self.is_fitted_ = False

        # Let the model know its been fitted.
        self.is_fitted_ = True
        return self
