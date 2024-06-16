from cnst import INFINITY
import numpy as np


class DecisionTree:
    def __init__(self, depth=0, max_depth=6):
        self.depth = depth
        self.max_depth = max_depth
        self.left = None
        self.right = None
        self.value = None
        self.feature_index = None
        self.threshold = None

    def fit(self, data):
        X = []
        y = []
        for row in data:
            X.append(row[:-1])
            y.append(row[-1])
        if self.depth >= self.max_depth or len(set(y)) == 1:
            self.value = max(set(y), key=list(y).count)
        else:
            self.feature_index, self.threshold = self.best_split(data)
            if self.feature_index is not None:
                left_data, right_data = self.split(data, self.feature_index, self.threshold)
                self.left = DecisionTree(self.depth + 1, self.max_depth)
                self.left.fit(left_data)
                self.right = DecisionTree(self.depth + 1, self.max_depth)
                self.right.fit(right_data)
            else:
                self.value = max(set(y), key=list(y).count)

    def best_split(self, data):
        best_feature, best_threshold, best_gain = None, None, 0
        current_impurity = self.gini([row[-1] for row in data])
        n_features = len(data[0]) - 1

        for feature_index in range(n_features):
            thresholds = set(row[feature_index] for row in data)
            for threshold in thresholds:
                left_data, right_data = self.split(data, feature_index, threshold)
                if not left_data or not right_data:
                    continue

                gain = self.information_gain(left_data, right_data, current_impurity)
                if gain > best_gain:
                    best_gain, best_feature, best_threshold = gain, feature_index, threshold

        return best_feature, best_threshold

    def split(self, data, feature_index, threshold):
        left = [row for row in data if row[feature_index <= threshold]]
        right = [row for row in data if row[feature_index > threshold]]
        return left, right

    def gini(self, y):
        counts = np.bincount(y)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)

    def information_gain(self, left, right, current_impurity):
        p = float(len(left) / (len(left) + len(right)))
        return (current_impurity - p * self.gini([row[-1] for row in left])
                - (1 - p) * self.gini([row[-1] for row in right]))

    def predict(self, X):
        if self.value is not None:
            return self.value
        feature_value = X[self.feature_index]
        if feature_value <= self.threshold:
            return self.left.predict(X)
        else:
            return self.right.predict(X)


class RandomForest:
    def __init__(self, n_tress=10, max_depth=3):
        self.n_trees = n_tress
        self.max_depth = max_depth
        self.trees = []

    def fit(self, data):
        self.trees = []
        for _ in range(self.n_trees):
            sample = self.bootstrap_sample(data)
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(sample)
            self.trees.append(tree)

    def bootstrap_sample(self, data):
        n_samples = len(data)
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return [data[i] for i in indices]

    def predict(self, X):
        tree_predictions = list(map(lambda tree: tree.predict(X), self.trees))
        return max(set(tree_predictions), key=tree_predictions.count)
