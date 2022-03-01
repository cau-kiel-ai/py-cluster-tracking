import numpy as np
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors


class GridTracker:
    """
    This class allows to track clusters by using a grid as a hashing data structure.

    :param value_range: A list with d (dimension) entries containing 2 values for upper and lower bounds for each axis.
    :param bins: The number of bins to use for each axis. This value must be passed as list, which d elements.
        d depicting the number of dimensions.
    :param threshold: A lower bound expressed as ratio for a cluster to be considered a match.
    """

    def __init__(
        self,
        value_range,
        bins,
        threshold,
    ):
        self.threshold = threshold
        self.value_range = value_range
        self.steps = []
        self.bins = bins
        for idx, r in enumerate(self.value_range):
            range_min, range_max = r
            _, step = np.linspace(range_min, range_max, bins[idx], retstep=True)
            self.steps.append(step)
        self.previous = None

    def next(self, data, labels):
        """
        Calculate the matches for the clusters of the given data based on previous data instances.

        :param data: A list of coordinates for all data points.
        :param labels: A list of labels depicting to which cluster each data point belongs.
        :return: A list of tuples showing the link between the clusters of the current and previous time instance.
        """
        matches = []
        temp = dict()
        for idx, value in enumerate(data):
            label = labels[idx]
            if label == -1:
                continue
            label_set = temp.get(label, set())
            bins = tuple()
            for jdx, coord in enumerate(value):
                dim_min = self.value_range[jdx][0]
                dim_max = self.value_range[jdx][1]
                step = self.steps[jdx]
                if coord != dim_max:
                    bins = (*bins, int((coord - dim_min) / step))
                else:
                    bins = (*bins, self.bins[jdx] - 2)
            label_set.add(bins)
            temp[label] = label_set

        if self.previous is not None:
            for label, label_set in temp.items():
                potential = []
                for p_label, p_label_set in self.previous.items():
                    ratio = len(label_set.intersection(p_label_set)) / len(label_set)
                    if ratio > self.threshold:
                        potential.append(((label, p_label), ratio))
                if len(potential) > 0:
                    matches.append(max(potential, key=lambda x: x[1])[0])

        self.previous = temp
        return matches


class CentroidTracker:
    """
    This class allows to track clusters by using a grid as a hashing data structure.

    :param eps: A float depicting an upper threshold for the travelled distance.
    :param metric: see :func:`~sklearn.metrics.pairwise.pairwise_distances`
    """

    def __init__(
        self,
        eps,
        metric='euclidean',
    ):
        self.eps = eps
        self.previous = None
        self.metric = metric

    def next(self, data, labels):
        """
        Calculate the matches for the clusters of the given data based on previous data instances.

        :param data: A list of coordinates for all data points.
        :param labels: A list of labels depicting to which cluster each data point belongs.
        :return: A list of tuples showing the link between the clusters of the current and previous time instance.
        """
        matches = []
        data = np.asarray(data)
        labels = np.asarray(labels)
        data = data[labels != -1]
        labels = labels[labels != -1]
        clf = NearestCentroid()
        clf.fit(data, labels)
        centroids = clf.centroids_
        if self.previous is not None:
            for idx, cent in enumerate(centroids):
                nearest = self.previous.predict([cent])
                x = [cent, self.previous.centroids_[nearest][0]]
                distance = pairwise_distances(x, metric=self.metric)[0][1]
                if distance < self.eps:
                    matches.append((clf.classes_[idx], nearest[0]))
        self.previous = clf
        return matches


class PairwiseTracker:
    """
    This class allows to track clusters by comparing all data points in a pairwise manner.

    :param eps: A float depicting an upper threshold for the travelled distance.
    :param threshold: A float depicting lower bound for a ratio of matching
        data points to consider two clusters a match.
    :param metric: see :func:`~sklearn.neighbors.NearestNeighbors`
    :param algorithm: see :func:`~sklearn.neighbors.NearestNeighbors`
    """
    def __init__(
        self,
        eps,
        threshold,
        metric='euclidean',
        algorithm='brute'
    ):
        self.eps = eps
        self.threshold = threshold
        self.previous = None
        self.metric = metric
        self.algorithm = algorithm

    def next(self, data, labels):
        """
        Calculate the matches for the clusters of the given data based on previous data instances.

        :param data: A list of coordinates for all data points.
        :param labels: A list of labels depicting to which cluster each data point belongs.
        :return: A list of tuples showing the link between the clusters of the current and previous time instance.
        """
        matches = []
        data = np.asarray(data)
        labels = np.asarray(labels)
        data = data[labels != -1]
        labels = labels[labels != -1]
        label_set = dict()
        for idx, e in enumerate(data):
            label_list = label_set.get(labels[idx], list())
            label_list.append(e)
            label_set[labels[idx]] = label_list

        if self.previous is not None:
            for idx, current_cluster in label_set.items():
                neigh = NearestNeighbors(n_neighbors=1, algorithm=self.algorithm, metric=self.metric)
                neigh.fit(current_cluster)
                ratios = []
                for jdx, prev_cluster in self.previous.items():
                    neighbors_dist, neighbors_idx = neigh.kneighbors(prev_cluster)
                    num_matches = len(list(filter(lambda x: x[0] < self.eps, neighbors_dist)))
                    ratio = num_matches/len(current_cluster)
                    ratios.append((jdx, ratio))
                cluster_match = max(ratios, key=lambda x: x[1], default=None)
                if cluster_match is not None:
                    matches.append((idx, cluster_match[0]))

        self.previous = label_set
        return matches
