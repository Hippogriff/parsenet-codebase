"""
This implements differentiable mean shift clustering
algorithm for the use in deep learning.
"""
import numpy as np
import torch

from src.guard import guard_exp, guard_sqrt


class MeanShift:
    def __init__(self, ):
        """
        Differentiable mean shift clustering inspired from
        https://arxiv.org/pdf/1712.08273.pdf
        """
        pass

    def mean_shift(self, X, num_samples, quantile, iterations, kernel_type="gaussian", bw=None, nms=True):
        """
        Complete function to do mean shift clutering on the input X
        :param num_samples: number of samples to consider for band width
        calculation
        :param X: input, N x d
        :param quantile: to be used for computing number of nearest
        neighbors, 0.05 works fine.
        :param iterations: 
        """
        if bw == None:
            with torch.no_grad():
                bw = self.compute_bandwidth(X, num_samples, quantile)

                # avoid numerical issues.
                bw = torch.clamp(bw, min=0.003)
        new_X, _ = self.mean_shift_(X, b=bw, iterations=iterations, kernel_type=kernel_type)
        if not nms:
            return new_X, bw

        with torch.no_grad():
            _, indices, new_labels = self.nms(new_X, X, b=bw)
        center = new_X[indices]

        return new_X, center, bw, new_labels

    def mean_shift_(self, X, b, iterations=10, kernel_type="gaussian"):
        """
        Differentiable mean shift clustering.
        X are assumed to lie on the hyper shphere, and thus are normalized
        to have unit norm. This is done for computational
        efficiency and will not work if the assumptions are voilated.
        :param X: N x d, points to be clustered
        :param b: bandwidth
        :param iterations: number of iterations
        """
        # initialize all the points as the seed points
        new_X = X.clone()
        delta = 1
        for i in range(iterations):
            if kernel_type == "gaussian":
                dist = 2.0 - 2.0 * new_X @ torch.transpose(X, 1, 0)

                # TODO Normalization is still remaining.
                K = guard_exp(- dist / (b ** 2) / 2)
            else:
                # epanechnikov
                dist = 2.0 - 2.0 * new_X @ torch.transpose(X, 1, 0)
                dist = 3 / 4 * (1 - dist / (b ** 2))
                K = torch.nn.functional.relu(dist)

            D = 1 / (torch.sum(K, 1, keepdim=True))

            # K: N x N, X: N x d, D: N x 1
            M = (K @ X) * D - new_X
            new_X = new_X + delta * M

            # re-normalize it to lie on unit hyper-sphere.
            new_X = new_X / torch.norm(new_X, dim=1, p=2, keepdim=True)
        # new_X: center of the clusters
        return new_X, X

    def guard_mean_shift(self, embedding, quantile, iterations, kernel_type="gaussian"):
        """
        Some times if band width is small, number of cluster can be larger than 50, that
        but we would like to keep max clusters 50 as it is the max number in our dataset.
        in that case you increase the quantile to increase the band width to decrease
        the number of clusters.
        """
        while True:
            _, center, bandwidth, cluster_ids = self.mean_shift(
                embedding, 5000, quantile, iterations, kernel_type=kernel_type
            )
            if torch.unique(cluster_ids).shape[0] > 49:
                quantile *= 2
            else:
                break
        return center, bandwidth, cluster_ids

    def kernel(self, X, kernel_type, bw):
        """
        Assuing that the feature vector in X are normalized.
        """
        if kernel_type == "gaussian":
            # gaussian
            dist = 2.0 - 2.0 * X @ torch.transpose(X, 1, 0)
            # TODO not considering the normalization factor
            K = guard_exp(- dist / (bw ** 2) / 2)

        elif kernel_type == "epa":
            # epanechnikov
            dist = 2.0 - 2.0 * X @ torch.transpose(X, 1, 0)
            dist = 3 / 4 * (1 - dist / (bw ** 2))
            K = torch.nn.functional.relu(dist)
        return K

    def compute_bandwidth(self, X, num_samples, quantile):
        """
        Compute the bandwidth for mean shift clustering.
        Assuming the X is normalized to lie on hypersphere.
        :param X: input data, N x d
        :param num_samples: number of samples to be used
        for computing distance, <= N
        :param quantile: nearest neighbors used for computing
        the bandwidth.
        """
        N = X.shape[0]
        L = np.arange(N)
        np.random.shuffle(L)
        X = X[L[0:num_samples]]
        # dist = (torch.unsqueeze(X, 1) - torch.unsqueeze(X, 0)) ** 2
        dist = 2 - 2 * X @ torch.transpose(X, 1, 0)
        # dist = torch.sum(dist, 1)
        K = int(quantile * num_samples)
        top_k = torch.topk(dist, k=K, dim=1, largest=False)[0]

        max_top_k = guard_sqrt(top_k[:, -1], 1e-6)

        return torch.mean(max_top_k)

    def nms(self, centers, X, b):
        """
        Non max suprression.
        :param centers: center of clusters
        :param X: points to be clustered
        :param b: band width used to get the centers
        """
        membership = 2.0 - 2.0 * centers @ torch.transpose(X, 1, 0)

        # which cluster center is closer to the points
        membership = torch.min(membership, 0)[1]

        # Find the unique clusters which is closer to at least one point
        uniques, counts_ = np.unique(membership.data.cpu().numpy(), return_counts=True)

        # count of the number of points belonging to unique cluster ids above
        counts = torch.from_numpy(counts_.astype(np.float32)).cuda(torch.get_device(centers))

        num_mem_cluster = torch.zeros((X.shape[0])).cuda(torch.get_device(centers))

        # Contains the count of number of points belonging to a
        # unique cluster
        num_mem_cluster[uniques] = counts

        # distance of clusters from each other
        dist = 2.0 - 2.0 * centers @ torch.transpose(centers, 1, 0)

        # find the nearest neighbors to each cluster based on some threshold
        # TODO this could be b ** 2
        cluster_nbrs = dist < b
        cluster_nbrs = cluster_nbrs.float()

        cluster_center_ids = torch.unique(torch.max(cluster_nbrs[uniques] * num_mem_cluster.reshape((1, -1)), 1)[1])
        # pruned centers
        centers = centers[cluster_center_ids]

        # assign labels to the input points
        # It is assumed that the embeddings lie on the hypershphere and are normalized
        temp = centers @ torch.transpose(X, 1, 0)
        labels = torch.max(temp, 0)[1]
        return centers, cluster_center_ids, labels

    def pdist(self, x, y):
        x = torch.unsqueeze(x, 1)
        y = torch.unsqueeze(y, 0)
        dist = torch.sum((x - y) ** 2, 2)
        return dist

        # return torch.unique(torch.max(cluster_nbrs[uniques] * num_mem_cluster.reshape((1, -1)), 0)[1])

# def nms(new_X, X, b):
#     membership = new_X @ torch.transpose(X, 1, 0)

#     # which cluster center is closer to the points
#     membership = torch.max(membership, 0)[1]

#     # Find the unique clusters which is closer to at least one point
#     uniques, counts_ = np.unique(membership.data.cpu().numpy(), return_counts=True)

#     # count of the number of points belonging to unique cluster ids above
#     counts = torch.from_numpy(counts_.astype(np.float32)).cuda()

#     num_mem_cluster = torch.zeros((10000)).cuda().float()

#     # Contains the count of number of points belonging to a
#     # unique cluster
#     num_mem_cluster[uniques] = counts

#     # distance of clusters from each other
#     dist = new_X @ torch.transpose(new_X, 1, 0)

#     # find the nearest neighbors based on some threshold
#     correct = dist > b
#     correct = correct.float()
#     clusters = []
#     for i in uniques:
#         # choose the cluster id which has more number of points closer to itself
#         clusters.append(torch.max(correct[i] * num_mem_cluster[i], 0)[1])

#     return torch.unique(torch.stack(clusters))


# def mean_shift_clustering(X, b, iterations = 10):
#     """
#     :param X: N x d
#     :param b: bandwidth
#     Problems: 
#     """
#     for i in range(iterations):
#         K = torch.exp(X @ torch.transpose(X, 1, 0) * b)
#         D = 1 / (torch.sum(K, 1, keepdim=True) + 1e-7)

#         # K: N x N, X: N x d, D: N x 1
#         M = (K @ X) * D - X
#         X = X + M
#         X = X / torch.norm(X, dim=1, p=2, keepdim=True)
#         print (torch.norm(M))
#         dist = X @ torch.transpose(X, 1, 0)
#     return X, dist


# def mean_shift_clustering_1(X, b, iterations = 10):
#     """
#     :param X: N x d
#     :param b: bandwidth
#     Problems: 
#     """
#     new_X = X.clone()
#     delta = 1
#     for i in range(iterations):
#         K = torch.exp(new_X @ torch.transpose(X, 1, 0) / b)
#         D = 1 / (torch.sum(K, 1, keepdim=True))

#         # K: N x N, X: N x d, D: N x 1
#         M = (K @ X) * D - new_X
#         new_X = new_X + delta * M
#         print (torch.mean(torch.norm(M, dim=1, p=2)))

#         new_X = new_X / torch.norm(new_X, dim=1, p=2, keepdim=True)
#     return new_X, X
