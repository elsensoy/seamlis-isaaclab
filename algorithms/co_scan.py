import numpy as np
import skfmm
from scipy.optimize import linear_sum_assignment

from algorithms.kmeans import kmeans

# Global Policy - implementation of "Multi-robot collaborative dense scene reconstruction"

class CoScanPlanner:
    def __init__(self):
        self.global_goals = None
        
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
        np_frontier_map[dd_mask] = 0 # remove unreachable frontiers

        # cluster_center is in [y, x] order
        cluster_center, nearest_clusters, is_frontier, frontier_idx = kmeans(np_obstacle_map, np_frontier_map, num_agent)
        # nearest_clusters: (n_frontier,)
        if cluster_center is None:
            self.global_goals = [None] * num_agent
            return self.global_goals

        # Assign clusters to agents using Hungarian algorithm
        nc = cluster_center[0].shape[0]
        cost = np.array([np_obstacle_map_distance[i][cluster_center[0], cluster_center[1]] for i in range(num_agent)])
        cost = np.hstack([cost] * ((num_agent - 1) // nc + 1))
        row_ind, col_ind = linear_sum_assignment(cost)

        self.global_goals = np.zeros((num_agent, 2), dtype=np.int32)
        for i in range(num_agent):
            cluster_idx = col_ind[i] % nc
            agent_idx = row_ind[i]
            frontier_dist = np_obstacle_map_distance[agent_idx][is_frontier]
            frontier_dist[nearest_clusters != cluster_idx] = np.inf # only consider frontiers in the assigned cluster
            select = np.argmin(frontier_dist)
            # return goal in [x, y] order
            self.global_goals[agent_idx] = [frontier_idx[1][select], frontier_idx[0][select]]

        return self.global_goals

    def replan(self, np_obstacle_map, np_frontier_map, goal, planning_window):
        # goal: [x,y]
        gx1, gx2, gy1, gy2 = planning_window
        np_obstacle_map = np.ma.masked_values(np_obstacle_map, 1)
        np_obstacle_map[goal[1], goal[0]] = 1
        dd = skfmm.distance(1 - np_obstacle_map)[gy1:gy2, gx1:gx2]
        dd[np_frontier_map[gy1:gy2, gx1:gx2] == 0] = np.inf
        goal = np.unravel_index(np.argmin(dd), dd.shape)
        return goal[0], goal[1]

    def check_finish(self, np_frontier_map, stop):
        if len([1 for goal in self.global_goals if goal is not None]) == 1:
            self.global_goals = None
            return True
        for goal, stopped in zip(self.global_goals, stop):
            if goal is None:
                continue
            if stopped:
                stopped = False
                self.global_goals = None
                return True
            elif np_frontier_map[goal[1], goal[0]] == 0:
                self.global_goals = None
                return True
        return False