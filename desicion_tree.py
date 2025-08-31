import numpy as np

class DecisionTreeClassifier:
    """
    A decision tree classifier implemented from scratch.

    Parameters:
        min_samples_split (int): The minimum number of samples required to split an internal node.
        max_depth (int): The maximum depth of the tree.
    """
    class Node:
        """
        Helper class representing a single node in the decision tree.
        A node is either a decision node (internal) or a leaf node (terminal).
        """
        def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
            # --- For a Decision Node ---
            self.feature_index = feature_index  # Index of the feature to split on.
            self.threshold = threshold          # Threshold value for the split.
            self.left = left                    # Left child node.
            self.right = right                  # Right child node.
            self.info_gain = info_gain          # Information gain achieved by this split.
            
            # --- For a Leaf Node ---
            self.value = value                  # The predicted class label.

        def is_leaf_node(self):
            """Checks if the node is a leaf node."""
            return self.value is not None

    def __init__(self, min_samples_split=2, max_depth=100):
        # The root node of the decision tree.
        self.root = None
        
        # Stopping conditions for the tree building process.
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

    # --- Public Methods (API) ---
    def print_tree(self, tree=None, indent=" "):
        """
        Recursively prints the decision tree structure to standard output.
        """
        # If no tree is specified, start from the root of the instance's tree.
        if tree is None:
            tree = self.root
        
        # Base case: If the node is a leaf, print its value.
        if tree.value is not None:
            print(tree.value)
            return
            
        # Recursive step: Print the split condition and then traverse children.
        else:
            # Print the current node's split rule and information gain.
            print(f"X_{tree.feature_index} <= {tree.threshold:.2f} ? (Gain: {tree.info_gain:.4f})")

            # Print the left subtree.
            print(f"{indent}├─ Left: ", end="")
            self.print_tree(tree.left, indent + "│  ")

            # Print the right subtree.
            print(f"{indent}└─ Right: ", end="")
            self.print_tree(tree.right, indent + "   ")

    def fit(self, X, y, impurity_calculation="entropy"):
        """
        Builds and trains the decision tree classifier.

        Parameters:
            X (np.ndarray): The training input samples (features).
            y (np.ndarray): The target values (labels).
        """
        # Ensure y is a 2D column vector for concatenation.
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
            
        dataset = np.concatenate((X, y), axis=1)
        # Start the recursive tree building process.
        self.root = self._build_tree(dataset, impurity_calculation)

    def predict(self, X):
        """
        Predicts class labels for samples in X.

        Parameters:
            X (np.ndarray): The input samples to classify.

        Returns:
            list: A list of predicted class labels for each sample in X.
        """
        # Use a list comprehension to make a prediction for each row in X.
        predictions = np.array([self._make_prediction(row, self.root) for row in X])
        return predictions

    # --- Private Helper Methods ---
    
    def _build_tree(self, dataset, impurity_calculation, current_depth=0 ):
        """
        Recursively builds the decision tree in a greedy fashion.

        This function determines whether to create a leaf node (an answer) or a
        decision node (a question) based on a set of stopping criteria. If a
        decision node is created, it calls itself to build the subtrees.

        Parameters:
            dataset (np.ndarray): The data for the current node, with the last
                                column being the target variable.
            current_depth (int): The depth of the current node in the tree.

        Returns:
            Node: The root node of the constructed tree or subtree.
        """
        # Split the dependent and independent variables of the dataset.
        X, y = dataset[:, :-1], dataset[:, -1]
        num_samples, num_features = X.shape

        # --- Base Case: Check for stopping conditions ---
        # 1. If min_samples or max_depth is reached.
        # 2. If all samples in the node belong to one class (node is pure).
        if (num_samples < self.min_samples_split or
            current_depth >= self.max_depth or
            len(np.unique(y)) == 1):
            
            leaf_value = self._calculate_leaf_value(y)
            return DecisionTreeClassifier.Node(value=leaf_value)

        # --- Recursive Step: Find the best split and continue building ---
        best_split = self._get_best_split(dataset, impurity_calculation)

        # Check if a valid split that provides positive information gain was found.
        if best_split and best_split.get("info_gain", 0) > 0:
            # Recurse on the left child.
            left_subtree = self._build_tree(
                best_split["dataset_left"], current_depth + 1
            )
            # Recurse on the right child.
            right_subtree = self._build_tree(
                best_split["dataset_right"], current_depth + 1
            )

            # Return a decision node that links to the subtrees.
            return DecisionTreeClassifier.Node(feature_index=best_split["feature_index"],
                        threshold=best_split["threshold"],
                        left=left_subtree,
                        right=right_subtree,
                        info_gain=best_split["info_gain"])

        else:
            # Base Case: No split provided improvement 
            # If no split could increase purity, create a leaf node.
            leaf_value = self._calculate_leaf_value(y)
            return DecisionTreeClassifier.Node(value=leaf_value)
    

    def _get_best_split(self, dataset, impurity_calculation):
        """
        gets the best split that maximizes the information gain and thus reducing the entropy of the dataset

        parameters: 
        returns:
            a dictionary containing the dataset split into left and right.
            note: dictionary also includes the index of the feature used to split the data and the information gain

        """

        x, y = dataset[:, :-1], dataset[:, -1]
        num_samples, num_features = np.shape(x)

        # dictionary to store the best split
        best_split = {}
        max_info_gain = -float("inf")

        for feature_index in range(num_features):
            # the current feature we are looking at
            feature_values = dataset[:, feature_index]

            # since there are infinite possible ways to split the data set we are going to cap this by the number of unique values in the feture vector
            possible_threshold = np.unique(feature_values)
            for threshold in possible_threshold:
                # get current split
                dataset_left, dataset_right = self._split(dataset, feature_index, threshold)

                # check if children are not null
                if ((len(dataset_left) > 0) and (len(dataset_right) > 0) ):
                    left_y, right_y = dataset_left[:, -1], dataset_right[:, -1]

                    # compute information gain
                    information_gain = self._information_gain(y, left_y, right_y, impurity_calculation)

                    # update the currentsplit gains more information than the previos best
                    if (information_gain > max_info_gain):
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["info_gain"] = information_gain

                        max_info_gain = information_gain

        return best_split
    
    def _split(self, dataset, feature_index, threshold):
        """
        Splits a dataset into two subsets based on a threshold and feature.

        Parameters:
            dataset (np.ndarray): The dataset to split.
            feature_index (int): The index of the feature to split on.
            threshold (float): The threshold value to split by.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing the left and right
                                        dataset subsets.
        """
        # Create a boolean mask by applying the condition to the entire feature column at once.
        left_mask = dataset[:, feature_index] <= threshold
        
        # The right dataset is everything that doesn't match the left mask.
        # The '~' operator inverts the boolean mask.
        right_mask = ~left_mask
        
        # Use the boolean masks to select the rows for each subset.
        left_dataset = dataset[left_mask]
        right_dataset = dataset[right_mask]
        
        return left_dataset, right_dataset

    def _information_gain(self, parent, left_child, right_child, mode="entropy"):
        """
        Helper function to compute the information gain of a split.

        Information gain is the parent's impurity minus the weighted average
        of the children's impurity.

        Parameters:
            parent (np.ndarray): The dependent variable of the parent dataset.
            left_child (np.ndarray): The dependent variable of the left child dataset.
            right_child (np.ndarray): The dependent variable of the right child dataset.
            mode (str): The impurity measure to use ('gini' or 'entropy').

        Returns:
            float: The calculated information gain.
        """
        # Calculate the weight of each child node based on the number of samples.
        weight_left = len(left_child) / len(parent)
        weight_right = len(right_child) / len(parent)

        # Calculate information gain based on the chosen impurity mode.
        if mode == "gini":
            parent_impurity = self._gini_impurity(parent)
            left_impurity = self._gini_impurity(left_child)
            right_impurity = self._gini_impurity(right_child)
        else:  # Default to entropy
            parent_impurity = self._entropy(parent)
            left_impurity = self._entropy(left_child)
            right_impurity = self._entropy(right_child)

        # Formula: Gain = Parent Impurity - (Weight_Left * Impurity_Left + Weight_Right * Impurity_Right)
        weighted_child_impurity = (weight_left * left_impurity) + (weight_right * right_impurity)
        information_gain = parent_impurity - weighted_child_impurity
        
        return information_gain

    # --- Impurity Measures ---

    def _entropy(self, y):
        """
        Calculates the entropy of a vector 'y'.

        Entropy is a measure of randomness or uncertainty.
        Formula: H(X) = - Σ (p_i * log2(p_i))
        """
        if len(y) == 0:
            return 0  # Entropy of an empty set is 0.

        # Count occurrences of each class label.
        _, counts = np.unique(y, return_counts=True)
        # Calculate the probability of each class.
        probabilities = counts / len(y)
        
        # Calculate and sum the entropy for each class.
        entropy = -np.sum([p * np.log2(p) for p in probabilities if p > 0])
        
        return entropy

    def _gini_impurity(self, y):
        """
        Calculates the Gini impurity of a vector 'y'.

        Gini impurity measures the likelihood of an incorrect classification
        of a new instance of a random variable.
        Formula: Gini(E) = 1 - Σ (p_i^2)
        """
        if len(y) == 0:
            return 0  # Gini impurity of an empty set is 0.
            
        # Count occurrences of each class label.
        _, counts = np.unique(y, return_counts=True)
        # Calculate the probability of each class.
        probabilities = counts / len(y)
        
        # The Gini impurity is 1 minus the sum of squared probabilities.
        gini = 1 - np.sum(probabilities**2)
        
        return gini
    
        
    def _calculate_leaf_value(self, y):
        """
        Determines the value for a leaf node by finding the most frequent label.
        This is a manual implementation without using collections.Counter.
        """
        # Handle the edge case of an empty node.
        if y is None:
            return None

        # Create a dictionary to store the frequency of each label.
        frequency_map = {}
        for element in y:
            # .get(element, 0) fetches the current count or defaults to 0 if the element is new.
            frequency_map[element] = frequency_map.get(element, 0) + 1

        most_frequent_label = max(frequency_map, key=frequency_map.get)
        
        return most_frequent_label

    def _make_prediction(self, x, tree):
        """
        Recursively traverses the tree to classify a single data instance.

        Parameters:
            x (np.ndarray): A single data instance (a row of features).
            tree (Node): The current node in the decision tree.

        Returns:
            The predicted class label from a leaf node.
        """
        # Base case: If we have reached a leaf node, return its value.
        if tree.value is not None:
            return tree.value

        # Recursive step: Decide whether to go left or right.
        feature_value = x[tree.feature_index]
        
        if feature_value <= tree.threshold:
            # Traverse the left subtree.
            return self._make_prediction(x, tree.left)
        else:
            # Traverse the right subtree.
            return self._make_prediction(x, tree.right)
