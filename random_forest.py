import numpy as np
from desicion_tree import DecisionTreeClassifier
from joblib import Parallel, delayed                # Used to build all the trees in parallel because it can be done asynchronously

class RandomForestClassifier:
    def __init__(self, num_tree=10, min_samples_split=2, max_depth=100, num_features=None, n_jobs=-1):
        self.num_tree = num_tree
        self.min_samples_split=min_samples_split
        self.max_depth = max_depth
        self.num_features = num_features

        # List to store all the trees we have made in this forest
        self.trees = []
        
        # Specify how many CPU cores to use in building the trees in parallel. Default is all the CPU cores
        self.n_jobs=n_jobs
        
    
    # --- Public Methods (API) ---
    def fit(self, X, y):
        """
        Fits the random forest model based on Training data asynchronously

        Parameter:
            X: Values for the independant varaibles
            y: Values for the dependant variable
        
        Return Value:
            None
        """
        
        # Build each trees in parallel because one can be done independantly from another
        self.trees = Parallel(self.n_jobs)(
            delayed(self._train_single_tree)(X, y) for _ in range(self.num_tree)
        )

    def predict(self, X):
        """
        Predicts class labels for a set of input samples.

        This method operates by aggregating the predictions from every decision
        tree in the forest and determining the final class label via a majority
        vote.

        The process involves three main steps:
        1.  Gathering predictions from each individual tree for all samples.
        2.  Transposing the resulting array so that each row contains all the
            predictions (votes) for a single sample.
        3.  For each sample, finding the most frequently occurring prediction,
            which becomes the final output for that sample.

        Parameters:
            X (np.ndarray): The input samples to classify, with a shape of
                            (n_samples, n_features).

        Returns:
            np.ndarray: An array of predicted class labels for each input sample.
        """
        # Get predictions from every tree in the forest.
        all_tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        
        # Transpose the predictions to align votes for each sample.
        predictions_per_sample = all_tree_predictions.T
        
        # Calculate the majority vote for each sample.
        final_predictions = np.array([self._most_common_label(votes) for votes in predictions_per_sample])
        
        return final_predictions


    # Private helper methods
    def _train_single_tree(self, X, y):
        """
        Creates and trains a single decision tree on a bootstrapped sample of the data.

        This helper function is designed to be called in parallel to build the forest.

        Parameters:
            X (np.ndarray): The full feature dataset.
            y (np.ndarray): The full target labels.

        Returns:
            DecisionTreeClassifier: A single, trained decision tree object.
        """
        # Create a bootstrapped sample from the original data.
        X_sample, y_sample = self._bootstrap_sample(X, y)

        # Initialize a new Decision Tree with the forest's hyperparameters.
        tree = DecisionTreeClassifier(
            min_samples_split=self.min_samples_split,
            max_depth=self.max_depth
        )

        # Train the tree on the bootstrapped sample, not the original data.
        tree.fit(X_sample, y_sample)
        
        # Return the fully trained tree.
        return tree

    import numpy as np

    def _bootstrap_sample(self, X, y):
        """
        Creates a sample by bootstrapping both rows (data points) and features (columns).

        This method generates a unique training set for a single tree by randomly
        selecting rows with replacement and features with replacement.
        """
        n_samples, n_features = X.shape
        
        # Select random row indices with replacement.
        sample_idxs = np.random.choice(n_samples, size=n_samples, replace=True)
        
        # Select random feature indices with replacement.
        # The number of features selected is controlled by self.num_feats
        # Use the square root of the number of features by default
        num_feats_to_select = self.num_features if self.num_features else int(np.sqrt(n_features))
        feature_idxs = np.random.choice(n_features, size=num_feats_to_select, replace=True)
        
        # Create the new sample by first selecting the rows, then the features.
        X_sample = X[sample_idxs][:, feature_idxs]
        
        # The labels (y) are only indexed by rows to maintain their link to the data points.
        y_sample = y[sample_idxs]
        
        return X_sample, y_sample


    
    def _most_common_label(self, y):
        """
        Finds the most frequently occurring label in a list by manual iteration.
        """
        # Return None immediately if the input list is empty.
        if y is None:
            return None

        frequency = {}
        # Use a tuple to track the most frequent item and its count.
        most_frequent = (None, 0) # (label, count)

        # Iterate through each predicted label in the list.
        for label in y:
            # Increment the count for the current label.
            frequency[label] = frequency.get(label, 0) + 1

            # If the current label's count is higher than the max seen so far then
            # update the most frequent tracker.
            if frequency[label] > most_frequent[1]:
                most_frequent = (label, frequency[label])

        return most_frequent[0]

        

    


        
        



