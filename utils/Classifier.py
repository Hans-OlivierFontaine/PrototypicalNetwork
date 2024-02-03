from abc import ABC, abstractmethod
import numpy as np


class Classifier(ABC):
    @abstractmethod
    def train(self, features, labels):
        pass

    @abstractmethod
    def predict(self, features):
        pass


class CentroidClassifier(Classifier):
    def __init__(self):
        self.centroids = {}

    def train(self, features, labels):
        # Calculate the centroid for each class
        for label in np.unique(labels):
            self.centroids[label] = np.mean(features[labels == label], axis=0)

    def predict(self, features):
        # Predict the class by finding the nearest centroid
        predictions = []
        for feature in features:
            predictions.append(
                min(self.centroids.keys(), key=lambda label: np.linalg.norm(feature - self.centroids[label])))
        return predictions


class CosineSimilarityClassifier(Classifier):
    def __init__(self):
        self.centroids = {}

    def train(self, features, labels):
        for label in np.unique(labels):
            self.centroids[label] = np.mean(features[labels == label], axis=0)

    def predict(self, features):
        predictions = []
        for feature in features:
            predictions.append(max(self.centroids.keys(), key=lambda label: np.dot(feature, self.centroids[label]) / (
                        np.linalg.norm(feature) * np.linalg.norm(self.centroids[label]))))
        return predictions


class MahalanobisClassifier(Classifier):
    def __init__(self):
        self.means = {}
        self.cov_inv = {}

    def train(self, features, labels):
        overall_cov = np.cov(features, rowvar=False)
        self.cov_inv = np.linalg.inv(overall_cov)
        for label in np.unique(labels):
            self.means[label] = np.mean(features[labels == label], axis=0)

    def predict(self, features):
        predictions = []
        for feature in features:
            mahalanobis_distances = {
                label: np.sqrt((feature - self.means[label]).T @ self.cov_inv @ (feature - self.means[label])) for label
                in self.means}
            predictions.append(min(mahalanobis_distances, key=mahalanobis_distances.get))
        return predictions


def classifier_factory(classifier_type):
    if classifier_type == 'centroid':
        return CentroidClassifier()
    elif classifier_type == 'cosine_similarity':
        return CosineSimilarityClassifier()
    elif classifier_type == 'mahalanobis':
        return MahalanobisClassifier()
    else:
        raise ValueError(f"Classifier type '{classifier_type}' is not supported.")

