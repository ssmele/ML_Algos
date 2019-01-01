import numpy as np
from collections import defaultdict
from graphviz import Digraph
from data import Data


class Node:
    """
    Class that represents the Node for constructing the DecisionTree.

    v should be an :string that matches the name of an attribute in the tree.

    Makes use of a dictionary to store the references to its children nodes.
    """

    def __init__(self, v, children=None, most_frequent=None):
        self.v = v
        self.children = children
        self.most_frequent = most_frequent

    def is_leaf(self):
        """
        If children is None then we know this is a leaf node
        :return: Boolean.
        """
        return self.children is None

    def __repr__(self):
        return "Node(v={})".format(self.v)

    def __str__(self):
        return "Node: v={}".format(self.v)


class DecisionTreeBinaryClassifier:
    """
    Implementation of the ID3 Binary Classifier algorithm. To use this class the data is assumed to be within a Data object.
    """

    def __init__(self, max_depth=np.inf):
        self.max_depth = max_depth
        self.reached_depth = None
        self.tree = None
        self.most_frequent_item = None

    def __repr__(self):
        return "DecisionTreeBinaryClassifier(max_depth={})".format(self.max_depth)

    def __str__(self):
        if self.tree is None and self.reached_depth is None:
            return "Untrained DecisionTreeBinaryClassifier"
        else:
            return "DecisionTreeBinaryClassifier: Reached Depth={}".format(self.reached_depth)

    def predict(self, data):
        """
        Predicts class for data rows given.

        :param data: Must be provided Data datatype.
        :return: Value representing the prediction.
        """
        # If the tree is not trained yet raise an error.
        if self.tree is None:
            raise ValueError("Model Not Trained Yet!")

        preds = np.ndarray(len(data), dtype='object')
        for ix in range(len(data)):

            # For each data row traverse the tree tell we find a leaf node.
            cur_node = self.tree
            while not cur_node.is_leaf():
                data_val = str(data.raw_data[ix][data.get_column_index(cur_node.v)])

                if data_val not in cur_node.children:
                    # If the datavalue has not already been seen set most frequent item as value.
                    cur_node = Node(self.most_frequent_item, None, None)
                else:
                    # If the value has been seen set corresponding child ro
                    cur_node = cur_node.children[data_val]

            # Once we reach a leaf then we have found our prediction.
            preds[ix] = cur_node.v

        return preds

    def train(self, data, label, pos_val):
        """
        Simple wrapper for the ID3 algorithm if more algorithms are implemented this might get more
        complicated. This is assuming a binary classifier
        :param data: Data object provided by professor.
        :param label: column name of the label.
        :param pos_val: label representing positive values.
        :return: generated decision tree starting at root node.
        """
        # Calculate all availabe columns to make splits on.
        cols = np.array(list(data.column_index_dict.keys()))
        possible_attributes = cols[(cols != label) & (cols != '')]

        self.most_frequent_item = self.most_frequent(data.get_column(label))

        # Calling ID3 to set up initial root node and build tree recursively.
        self.tree, self.reached_depth = self.ID3(data, 0, possible_attributes, label, pos_val)
        return self.tree

    def best_attribute(self, data, attributes, label, pos_val):
        """
        This method picks out the best attribute to split on based on the max of the information gain.
        :param data: data to use when calculating information gain. Must be provided Data datatype.
        :param attributes: Attributes to calculate information for.
        :param label: column name of the label.
        :param pos_val: label representing positive value.s
        :return: attribute with maximum information gain.
        """
        info_gains = [self.information_gain(str(a), data, label, pos_val) for a in attributes]
        return attributes[np.argmax(info_gains)]

    def information_gain(self, attribute, data, label, pos_val):
        """
        This method is an implementation of the information gain equation.
        :param attribute: Attribute to evaluate entropy for.
        :param data: Provided Data datatype.
        :param label: column name of the label.
        :param pos_val: label representing positive value.
        :return:
        """
        # Calculating entropy of whole dataset.
        data_entropy = self.entropy(data, label, pos_val)

        # Getting possible attribute values.
        possible_attribute_values = data.attributes[attribute].possible_vals

        # Setting up value count dict.
        val_counts = defaultdict(lambda: 0, keys=possible_attribute_values)
        unique_items, counts = np.unique(data.get_column(attribute), return_counts=True)
        for u, c in zip(unique_items, counts):
            val_counts[u] = c

        # Calculating proportion of all possible values in attribute.
        attribute_val_proportions = np.array([val_counts[val]/len(data) for val in possible_attribute_values])

        # Calculating entropies with respect to different attribute values.
        attribute_entropies = np.array([self.entropy(data, label, pos_val, attribute, val)
                                        for val in possible_attribute_values])

        # Calculate information gain and return.
        return data_entropy - np.sum(attribute_entropies * attribute_val_proportions)

    def information_gain_t(self, attribute, data, label, pos_val, cost_dict):
        # Form of information gain that considers cost.
        gain = self.information_gain(attribute, data, label, pos_val)
        return (gain ** 2)/cost_dict[attribute]

    def information_gain_s(self, attribute, data, label, pos_val, cost_dict):
        # Form of information gain that considers cost.
        gain = self.information_gain(attribute, data, label, pos_val)
        return (np.power(2, gain) - 1)/np.sqrt(cost_dict[attribute] + 1)

    @classmethod
    def entropy(self, data, label, pos_val, attribute=None, attribute_value=None):
        """
        This function calculates the entropy within the data. If not attribute, and attribute_value are given
        then its entropy of the full dataset. If attribute, and attribute_value are given then its the entropy
        with respect to that attribute.

        :param data: Provided Data datatype.
        :param label: column name of the label.
        :param pos_val: label representing positive value.
        :param attribute: attribute name to find entropy for.
        :param attribute_value: attribute value to find entropy for
        :return: entropy.
        """
        # Get mask of positive labels.
        label_mask = data.get_column(label) == pos_val

        # Getting appropriate values to use if attribute value was selected.
        if attribute and attribute_value:
            attribute_mask = data.get_column(attribute) == attribute_value
            total_count = np.sum(attribute_mask)
            pos_count = np.sum(attribute_mask & label_mask)
            neg_count = total_count - pos_count
        else:
            pos_count = np.sum(label_mask)
            total_count = len(data)
            neg_count = total_count - pos_count

        if total_count == 0:
            return 0

        # Get probabilities for positive and negative case.
        p_positive, p_negative = pos_count / total_count, neg_count / total_count

        # If we see any of the probabilties are 0 then end early to avoid undefined behavior in np.log2 when given 0.
        if p_positive == 0 or p_negative == 0:
            return 0

        # Perform entropy calculation and return
        return -(p_positive * np.log2(p_positive)) - (p_negative * np.log2(p_negative))

    @classmethod
    def most_frequent(self, column):
        """
        Find the most used value within numpy array.
        :param column: Numpy array.
        :return: Most frequent item in numpy array.
        """
        u, c = np.unique(column, return_counts=True)
        return u[np.argmax(c)]

    def ID3(self, data, depth, pos_atts, label='label', pos_val='e'):
        """
        :param data: Provided Data datatype.
        :param depth: variable to keep track of how far the tree gets.
        :param pos_atts: variable to keep track of what columns are available to choose from.
        :param label: Name of the column that specifies the labels.
        :param pos_val: value of positive examples in the dataset.
        :return: A Decision Tree
        """

        # If all labels same return leaf node and current depth.
        if len(set(data.get_column(label))) == 1:
            return Node(data.get_column(label)[0], None), depth

        # Check to see if going any deeper will result in exceeding depth limit. If so stop the recursion.
        if depth >= self.max_depth:
            # Returning mode of label.
            return Node(self.most_frequent(data.get_column(label)), None), depth

        # Find best attribute for the root node. Disregarding the label attribute.
        best = str(self.best_attribute(data, pos_atts, label, pos_val))
        # Get all the values possible for the best attribute.
        possible_vals = data.attributes[best].possible_vals

        # Make a node with the best attribute.
        tree = Node(best, defaultdict(dict), None)

        # Fill in the newly created best node with its children.
        depths = []
        for posi_val in possible_vals:
            r_depth = depth
            
            # Get subset of data where best attribute equals possible value.
            reduced_data = data.get_row_subset(best, posi_val)
            if len(reduced_data) == 0:
                # If their is no data to work with set child to leaf node with most frequent label.
                tree.children[posi_val] = Node(self.most_frequent(data.get_column(label)), None)
            else:
                # If their is data to work with induce a new tree with remaining data and attributes.
                tree.children[posi_val], r_depth = self.ID3(reduced_data, depth+1,
                                                            pos_atts[pos_atts != best].copy(),
                                                            label, pos_val)

            depths.append(r_depth)

        # return tree and maximum depth reached by children.
        return tree, max(depths)

    def make_tree(self, filename='test.gv'):
        """
        This method starts off the process for visualizing the tree.
        Should save a .gv file and pdf representing the tree.
        :return: None
        """
        g = Digraph('tree', filename=filename)
        self._make_tree(self.tree, g, None, None, 0)
        g.save(filename=filename)
        g.render(filename=filename)

    def _make_tree(self, t, g, p_name, p_att, depth):
        """
        Recursive method for making the tree.
        :param t: cur_node in the tree.
        :param g: the graph object
        :param p_name: Name of the parent node.
        :param p_att: Attribute used to get there
        :param depth: current depth of the tree.
        :return: None
        """
        # Anytime you get to a node draw one.
        node_name = str(t.v) + '_' + str(depth) + '_' + str(p_att) + '_' + str(np.random.randint(0, 1000000))
        g.node(node_name, label=str(t.v))

        # If name is not None we need to add parent nodes.
        if p_name is not None and p_att is not None:
            g.edge(p_name, node_name, label=p_att)

        # If we are not working with leaf node add all children to the graph.
        if not t.is_leaf():
            for k in t.children.keys():
                self._make_tree(t.children[k], g, node_name, k, depth+1)
