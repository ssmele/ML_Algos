import numpy as np
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class NaiveBayesBC:
    """
    This class is a basic implementation of the Naive Bayes Binary Classifier. This classifier only assumes two types of 
    classes positive and negative which you can specify the value for.
    """

    suggested_hyperparameters = {
        's': [2, 1.5, 1.0, 0.5]
    }

    def __init__(self, s=0.0):
        """
        Initializer for NaiveBayes classifier.
        :param s: smoothing hyperparameter.
        """
        self.s = s

    def get_params(self, deep=True):
        return {'s': self.s}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, X, y, pos_label_val=1, neg_label_val=-1):
        """
        Method to perform counts for logistic regression.
        :param X: attribute dataset.
        :param Y: labels
        :param pos_label_val: value of positive labels.
        :param neg_label_val: value of neagtive labels.
        :return: model
        """
        X, Y = check_X_y(X, y)
        
        # Calculate probabilities and counts for label values.
        self.num_p_ = np.sum(Y == pos_label_val)
        self.num_n_ = len(Y) - self.num_p_
        self.prob_p_ = self.num_p_ / len(Y)
        self.prob_n_ = 1- self.prob_p_

        # Calculate the negative values.
        pos_ix = (Y == pos_label_val).flatten()
        neg_ix = (Y == neg_label_val).flatten()

        # Calculate the probabilities for each attribute value for each different y-value.
        self.pos_attir_probs_ = np.array([np.array([(np.sum(x[pos_ix] == 0) + self.s) / (self.num_p_ + (2 * self.s)),
                                                    (np.sum(x[pos_ix] == 1) + self.s) / (self.num_p_ + (2 * self.s))])
                                          for x in X.T])
        self.neg_attir_probs_ = np.array([np.array([(np.sum(x[neg_ix] == 0) + self.s) / (self.num_n_ + (2 * self.s)),
                                                    (np.sum(x[neg_ix] == 1) + self.s) / (self.num_n_ + (2 * self.s))])
                                          for x in X.T])

        # Let the model know its been fitted.
        self.is_fitted_ = True
        return self

    def predict(self, X):
        """
        Calculate the arg max of P(Y=1|X), P(Y=0|X) assuming pos, neg labels are 0, 1. Applies log to all the probabilties
        so we no longer need to take the product of the probabilities and can take the sum.
        :param X: data to make predictions on
        :return: predictions made.
        """
        # Need to perform checks so sklearn and I dont get frustrated.
        X = check_array(X)
        check_is_fitted(self, 'is_fitted_')

        # Go through and caluclate possiblity of positive, and negative.
        preds = np.ndarray(len(X))
        for ix, x in enumerate(X):
            pop = np.log(self.prob_p_) + np.sum(np.log(
                np.array([possible_probs[int(x[ip])] for ip, possible_probs in enumerate(self.pos_attir_probs_)])))
            pon = np.log(self.prob_n_) + np.sum(np.log(
                np.array([possible_probs[int(x[ip])] for ip, possible_probs in enumerate(self.neg_attir_probs_)])))

            preds[ix] = [-1, 1][np.argmax([pon, pop])]
        return preds