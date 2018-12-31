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
        
        # If we want to use weights from a prior epoch.
        if weights_to_use is not None:
            weights = weights_to_use
        else:
            weights = self.weights_

        # For each data point go through and make a prediction.
        preds = np.ndarray(len(X))
        for ix, x in enumerate(X):
            preds[ix] = np.sign(np.dot(weights, x))
        return preds

class StochasticGDLogisticRegression(BaseSGD):
    """
    This class represents the classic logistic regression machine learning algorithm. Follows the sklearn estimator model
    spec in order to make use of the GridSearchCV module. 
    """
    
    suggested_hyperparameters = {
        'learning_rate':  (10**0, 10**-1, 10**-2, 10**-3, 10**-4, 10**-5),
        'sigma' :  (10**-1, 10**0, 10**1, 10**2, 10**3, 10**4),
        'epochs': (10, 30, 50, 75, 100),
        'verbose': (1,)
    }

    def __init__(self, learning_rate=.01, sigma=.01, epochs=10, **kwargs):
        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.sigma = sigma
        self.epochs = epochs
        
    def get_params(self, deep=True):
        return {'learning_rate': self.learning_rate, 
                'sigma': self.sigma}
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, X, y, validation_sets=None, epochs=10):
        """
        This method performs the training for the Stochastic Gradient Descent with SVM sub gradient loss function.
        :param X: Training data.
        :param Y: Labels for the training data.
        :param epochs: Number of epochs to run .
        :return: Newly trained model.
        """
        X, Y = check_X_y(X, y)
        
        # Dict for storing epoch info. 
        self.epoch_records = {}
        
        # Initialize the weights.
        self.weights_ = np.random.uniform(low=-0.01, high=0.01, size=X.shape[1])
        
        for epoch in range(self.epochs):

            # Calculate learning rate based on epoch.
            learning_rate = self.learning_rate/(1+epoch)

            # Go through each data point.
            for x, y in zip(*shuffle(X, Y)):
                # Calculate regularizer term.
                reg = np.divide(self.weights_, self.sigma)
                
                # This is to stop potential overflow.
                exp_term = y*np.dot(self.weights_, x)
                if exp_term > 6:
                    sig_term = 1
                elif exp_term < -6:
                    sig_term = 0
                else:
                    sig_term = np.divide(1, 1+ np.exp(exp_term))
                
                # Perform gradient update step.
                self.weights_ = self.weights_ - learning_rate*(reg - (y*x*sig_term))
                
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
