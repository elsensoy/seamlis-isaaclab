
import numpy as np
import skfmm
from random import shuffle

def kmeans(np_obstacle_map, np_frontier_map, k):
    """
    Calculates k-means clustering with the Intersection over Union (IoU) metric.
        :param boxes: numpy array of shape (r, 2), where r is the number of rows
        :param k: number of clusters
        :return: numpy array of shape (k, 2)
    @ note: when the indices returned by numpy, they always return in [y, x] order
            while the goal and robot positions are in a natural [x, y] order
    """

    is_frontier = np_frontier_map == 1 # it's [y,x] order
    frontier_idx = np.where(is_frontier) # [[y1, y2...], [x1, x2...]]
    rows = frontier_idx[0].shape[0] # number of frontier points in the discretized map (exclude duplicates)
    # execption: no frontiers or fewer frontiers than clusters
    if rows == 0:
        return None, None, None, None
    if rows < k:
        k = rows

    distances = np.empty((rows, k)) # assign as the number of frontier points
    last_clusters = np.zeros((rows,))

    # randomly select k frontier points as initial cluster centers
    cluster_idx = np.arange(rows)
    shuffle(cluster_idx)
    cluster_idx = cluster_idx[:k]

    # loop over 5 iterations for convergence
    for count in range(5):

        # For each cluster center, calculates geodesic distances to all frontier points using the Fast Marching Method (FMM).
        for k_i, ci in enumerate(cluster_idx):
            cy, cx = frontier_idx[0][ci], frontier_idx[1][ci] # [y, x] order

            np_obstacle_map_frontierK = np.ma.masked_values(np_obstacle_map, 1)
            np_obstacle_map_frontierK[cy, cx] = 1 # indexing with [y, x] order
            np_obstacle_map_distance = skfmm.distance(1 - np_obstacle_map_frontierK)
            distances[:, k_i] = np_obstacle_map_distance[frontier_idx[0], frontier_idx[1]] # indexing with [y, x] order

        # find the nearest cluster for each frontier point
        nearest_clusters = np.argmin(distances, axis=1) # now it shrink to (n_frontier,)

        if (last_clusters == nearest_clusters).all():
            break

        for k_i in range(k):
            # Assigns each frontier point to the nearest cluster.
            nearest_clusters_k_i_idx = np.where(nearest_clusters == k_i)
            if not nearest_clusters_k_i_idx:
                continue
            
            nearest_clusters_k_i_idx = nearest_clusters_k_i_idx[0]

            if frontier_idx[0][nearest_clusters_k_i_idx].size == 0 or frontier_idx[1][nearest_clusters_k_i_idx].size == 0:
                continue

            # Updates cluster centers to the point with minimum average distance to all points in the cluster.
            np_obstacle_map_frontierK = np.ma.masked_values(np_obstacle_map, 1)
            # indexing with [y, x] order
            np_obstacle_map_frontierK[int(frontier_idx[0][nearest_clusters_k_i_idx].mean()), int(frontier_idx[1][nearest_clusters_k_i_idx].mean())] = 1
            np_obstacle_map_distance = skfmm.distance(1 - np_obstacle_map_frontierK)
            
            # is_frontier is already in [y, x] order
            temp_distance_k_i = np_obstacle_map_distance[is_frontier][nearest_clusters_k_i_idx]
            cluster_idx[k_i] = nearest_clusters_k_i_idx[np.argmin(temp_distance_k_i)]

        last_clusters = nearest_clusters

    return (frontier_idx[0][cluster_idx], frontier_idx[1][cluster_idx]), nearest_clusters, is_frontier, frontier_idx
