import numpy as np
import skfmm
from scipy.optimize import linear_sum_assignment

from algorithms.kmeans import kmeans

# Global Policy - implementation of the vanilla frontier exploration (greedy)

class FrontierPlanner:
    def __init__(self, fov_angle=90, fov_margin=20):
        self.global_goals = None
        self.fov_angle = fov_angle + np.deg2rad(fov_margin) # prioritize frontiers in the front
        
    def get_long_term_goals(self, np_obstacle_map, np_frontier_map, agent_pos, agent_orientations):
        '''
        Compute long-term goals for agents using vanilla frontier exploration (greedy)
        
        @param np_obstacle_map: 2D numpy array representing obstacles (1) and free space (0)
        @param np_frontier_map: 2D numpy array representing frontier cells (1) and non-frontier cells (0)
        @param agent_pos: numpy array of shape (num_agents, 2) representing agent positions in [x, y] order
        @return: numpy array of shape (num_agents, 2) representing goal positions in [x, y] order
        '''
        num_agent = agent_pos.shape[0]
        np_frontier_map = np.copy(np_frontier_map)
        np_obstacle_map_distance = []

        # clip agnet_pos to the map size
        agent_pos = np.clip(agent_pos, [0, 0], [np_obstacle_map.shape[1] - 1, np_obstacle_map.shape[0] - 1])

        # Compute distance maps and mask unreachable areas
        dd_mask = np.ones(np_obstacle_map.shape, dtype=bool)
        for i in range(num_agent):
            np_obstacle_map_frontierK = np.ma.masked_values(np_obstacle_map, 1)
            np_obstacle_map_frontierK[agent_pos[i, 1], agent_pos[i, 0]] = 1
            dd = skfmm.distance(1 - np_obstacle_map_frontierK)
            np_obstacle_map_distance.append(dd)
            if isinstance(dd, np.ndarray):
                dd_mask[:] = False
            else:
                dd_mask &= dd.mask
        np_frontier_map[dd_mask] = 0

        # Perform k-means clustering on frontier points
        cluster_center, nearest_clusters, is_frontier, frontier_idx = kmeans(np_obstacle_map, np_frontier_map, num_agent)
        
        if cluster_center is None:
            self.global_goals = [None] * num_agent
            return self.global_goals

        # Assign clusters to agents using Hungarian algorithm
        nc = cluster_center[0].shape[0]
        cost = np.array([np_obstacle_map_distance[i][cluster_center[0], cluster_center[1]] for i in range(num_agent)])
        cost = np.hstack([cost] * ((num_agent - 1) // nc + 1))
        row_ind, col_ind = linear_sum_assignment(cost)

        # Select goals for each agent
        self.global_goals = np.zeros((num_agent, 2), dtype=np.int32)
        for i in range(num_agent):
            cluster_idx = col_ind[i] % nc
            agent_idx = row_ind[i]
            
            # Find the furthest frontier point in the assigned cluster
            # neareast_clusters: (n_frontier,) -> ex) [0, 1, 2, 0, 1, ...]  if there are three clusters
            cluster_frontiers = (nearest_clusters == cluster_idx)

            if np.any(cluster_frontiers):
                frontier_coords = np.column_stack((frontier_idx[1][cluster_frontiers], frontier_idx[0][cluster_frontiers]))
                agent_pos_2d = agent_pos[agent_idx][:2]
                agent_orientation = agent_orientations[agent_idx]

                # Calculate relative angles and distances
                relative_angles = np.arctan2(frontier_coords[:, 1] - agent_pos_2d[1],
                                             frontier_coords[:, 0] - agent_pos_2d[0]) - agent_orientation
                relative_angles = (relative_angles + np.pi) % (2 * np.pi) - np.pi  #angle normalize
                distances = np_obstacle_map_distance[agent_idx][frontier_idx[0][cluster_frontiers], frontier_idx[1][cluster_frontiers]]

                # Check which frontiers are within the FOV
                in_fov = np.abs(relative_angles) <= (self.fov_angle / 2)

                if np.any(in_fov):
                    # Select the frontier with the smallest relative angle within FOV
                    fov_frontier_idx = np.argmin(np.abs(relative_angles[in_fov]))
                    selected_frontier = frontier_coords[in_fov][fov_frontier_idx]
                else:
                    # If no frontiers in FOV, select the furthest frontier
                    furthest_frontier_idx = np.argmax(distances)
                    selected_frontier = frontier_coords[furthest_frontier_idx]

                # Set goal in [x, y] order
                self.global_goals[agent_idx] = selected_frontier
            else:
                # If no frontiers in the cluster, stay at current position
                self.global_goals[agent_idx] = agent_pos[agent_idx]

        return self.global_goals