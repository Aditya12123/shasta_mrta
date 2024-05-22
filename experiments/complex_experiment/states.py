import collections
import time

import numpy as np

from sklearn.cluster import KMeans

from .target import TargetManager


def cluster(vehicles, n_clusters, config):
    """Performs k-means clustering on the given vehicle data.

    Parameters
    ----------
    vehicles : list
        A list of vehicles with all the properties
    n_clusters : int
        Number of clusters to form
    config : yaml
        The configuration file

    Returns
    -------
    cluster_ids, cluster_pos
        Cluster ids and mean cluster position
    """
    # Get unique features
    features = np.unique(get_features(vehicles, config), axis=0)
    if np.all(features == 0):
        cluster_ids, cluster_pos = 0, np.empty(n_clusters)
    else:
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(features)
        cluster_ids = kmeans.labels_
        cluster_pos = []
        for j in range(n_clusters):
            cluster_pos.append(np.mean(features[cluster_ids == j, 0:2], axis=0))
        cluster_pos = np.asarray(cluster_pos)
    return cluster_ids, cluster_pos


def get_features(vehicles, config):
    """Get the features to be used for clustering

    Parameters
    ----------
    vehicles : list
        A list of UGV or UAV class
    config : yaml
        The configuration file

    Returns
    -------
    array
        A numpy array containing 3 features namely x, y position
        and battery or ammon depeding the vehicle type
    """
    w_cluster_battery_pos = config['weights']['w_cluster_battery_pos']
    w_cluster_ammo_pos = config['weights']['w_cluster_ammo_pos']
    features = []
    for vehicle in vehicles:
        temp = [0, 0, 0]  # Three features for clustering
        pos = vehicle.current_pos
        if vehicle.idle:
            temp[0] = pos[0]  # x position
            temp[1] = pos[1]  # y position
            if vehicle.type == 'uav':
                temp[2] = w_cluster_battery_pos * vehicle.battery  # Battery
            else:
                temp[2] = w_cluster_ammo_pos * vehicle.ammo  # Ammo
        features.append(temp)
    features = np.asarray(features)
    return features


def get_least_crowding(importance, n_targets, pareto_list, n_keep_in_pareto):
    """Get the crowding distance

    Parameters
    ----------
    importance : array
        An array with impotance of each node
    n_targets : int
        Number of targets in the mission
    pareto_list : list
        The nodes which are in pareto list
    n_keep_in_pareto : int
        Number of points to keep from pareto list

    Returns
    -------
    list
        A list containing top nodes depending on the crowding distance
    """
    range_f = np.zeros(shape=[n_targets, 1])
    MAX_NUM = 0
    for j in range(n_targets):
        MAX_NUM = max(MAX_NUM, max(importance[:, j]) - min(importance[:, j]))
        range_f[j] = max(importance[:, j]) - min(importance[:, j])

    nich_dist_anchor = MAX_NUM * n_targets + 1
    nich_dist = np.zeros(len(pareto_list))

    for j in range(n_targets):
        arg_s = np.argsort(-1 * importance[pareto_list, j])  # minimization
        nich_dist[arg_s[0]] += nich_dist_anchor
        for i in range(1, len(pareto_list) - 1):
            nich_dist[arg_s[i]] += abs(
                importance[arg_s[i + 1], j] - importance[arg_s[i - 1], j]
            )
        nich_dist[len(pareto_list) - 1] += abs(
            importance[arg_s[len(pareto_list) - 1], j]
            - importance[arg_s[len(pareto_list) - 2], j]
        )

    # Need to talk with amir
    arg_s2 = np.argsort(-1 * nich_dist)  # minimization
    # least_crowded_index = pareto_list[arg_s2[0:n_keep_in_pareto]]
    least_crowded_index = [pareto_list[i] for i in arg_s2[0:n_keep_in_pareto]]
    return least_crowded_index


def pareto_opt(importance, n_nodes, n_targets, n_keep_in_pareto):  # noqa
    """Performs pareto optimisation to get the top nodes to visit

    Parameters
    ----------
    importance : array
        An array with impotance of each node
    n_nodes : int
        Number of nodes in the graph
    n_targets : int
        Number of targets in the mission
    n_keep_in_pareto : int
        Number of points to keep from pareto list

    Returns
    -------
    list
        A list containing top nodes depending on the crowding distance
    """

    arg_s = np.argsort(importance[:, 0])  # minimization
    importance_updated = np.zeros((n_nodes, n_targets))
    for i in range(n_nodes):
        importance_updated[i, :] = importance[arg_s[i], :]

    pareto_list = [arg_s[0]]
    # This is n^2 m which is not optimal. You can get up to n log(n) m
    for i in range(n_nodes):
        point_i_will_be_non_dominated = True
        for j in range(0, i):
            point_i_will_be_dominated_by_point_j = True
            for k in range(n_targets):
                if importance_updated[j, k] > importance_updated[i, k]:
                    point_i_will_be_dominated_by_point_j = False
                    break
            if point_i_will_be_dominated_by_point_j:
                point_i_will_be_non_dominated = False
                break
        if point_i_will_be_non_dominated:
            pareto_list.append(arg_s[i])

    if n_keep_in_pareto == len(pareto_list):
        least_crowded_index = pareto_list
    elif n_keep_in_pareto <= len(pareto_list):
        least_crowded_index = get_least_crowding(
            importance, n_targets, pareto_list, n_keep_in_pareto
        )
    else:
        n = len(pareto_list)
        while n_keep_in_pareto > len(pareto_list):
            pareto_list.append(pareto_list[-n])
        least_crowded_index = pareto_list

    return least_crowded_index


def flatten_state_list(lists):
    """Flattens the state list to form a single list

    Parameters
    ----------
    l : list
        A list contaning the states of all the groups
    """
    for el in lists:
        if isinstance(el, collections.abc.Iterable) and not isinstance(
            el, (str, bytes)
        ):
            yield from flatten_state_list(el)
        else:
            yield el


class StatesExtractor:
    def __init__(self, config, core):
        self.current_time = 0
        self.config = config
        self.target_manager = TargetManager(config, core)
        self.map = core.get_map()
        self.pareto_nodes = None

    def calculate_distance(self, target, node):
        # Convert to cartesian
        cart_node = self.map.convert_to_cartesian([node['y'], node['x']])
        cart_target = self.map.convert_to_cartesian([target['y'], target['x']])

        # Calculate distance
        dist = np.linalg.norm(cart_target - cart_node)
        return dist

    def get_pareto_nodes_online(self):
        """Get pareto nodes to visit running online"""
        n_keep_in_pareto = self.config['state']['n_keep_in_pareto']
        n_nodes = self.config['simulation']['n_nodes']
        n_targets = self.config['simulation']['n_targets']

        # Importance matrix
        importance = np.zeros((n_nodes, n_targets))

        for j in range(n_nodes):
            for k, target in enumerate(self.config['simulation']['target_building_id']):
                target_info = self.target_manager.get_target_info(target)

                node_info = self.map.get_node_info(j)

                dist = self.calculate_distance(
                    self.map.get_node_info(target_info["id"]), node_info
                )
                probability_goals = target_info['probability_goals']

                # Calculate importance
                importance[j, k] = probability_goals * 1 / (dist + 0.001)
        # Top five pareto nodes
        pareto_nodes = pareto_opt(-1 * importance, n_nodes, n_targets, n_keep_in_pareto)
        return pareto_nodes

    def encode_state(self, idx, cluster_ids, cluster_pos, group_type, pareto_node_pos):
        """Encode the states given the cluster id, position and pareto nodes position

        Parameters
        ----------
        idx : int
            An int specifying the group id
        cluster_id : array
            An array from the clustering function
        cluster_pos : array
            An array with position of all the clusters
        group_type : str
            A string speciying the group type i.e., uav or ugv
        pareto_node_pos : array
            An array with position of all the pareto nodes

        Returns
        -------
        list
            A list containing the encoded states
        """
        state = []
        cluseter_pos = cluster_pos[idx]

        # Append distance from cluster to target
        diff = np.asarray(cluseter_pos) - np.asarray(pareto_node_pos)
        dist = np.linalg.norm(diff, axis=1)
        state.append(dist.tolist())

        # Append number of vehicle
        n_vehicles = self.n_vehicles(idx, cluster_ids)
        state.append(n_vehicles)
        return state

    def get_group_info(self, cluster_ids, cluster_pos, group_type, pareto_node_pos):
        """Form a group using cluster_ids and cluster_position.

        Parameters
        ----------
        idx : int
            An int specifying the group id
        cluster_id : array
            An array from the clustering function
        cluster_pos : array
            An array with position of all the clusters
        group_type : str
            A string speciying the group type i.e., uav or ugv
        pareto_node_pos : array
            An array with position of all the pareto nodes

        Returns
        -------
        dict
            A dictionary containing the
            group info where the key is the group id
        """

        if group_type == 'uav':
            n_groups = self.config['simulation']['n_uav_clusters']
        else:
            n_groups = self.config['simulation']['n_ugv_clusters']

        groups = []
        if cluster_pos.all():
            for i in range(n_groups):
                info = {}
                info['cluster_id'] = i
                info['position'] = cluster_pos[i]
                info['vehicle_ids'] = self.vehicles_ids(i, cluster_ids)
                info['group_type'] = group_type
                # Add states
                info['state'] = self.encode_state(
                    i, cluster_ids, cluster_pos, group_type, pareto_node_pos
                )
                groups.append(info)
        else:
            # Return all the states with zeros
            for i in range(n_groups):
                info = {}
                info['cluster_id'] = i
                info['position'] = [0, 0]
                info['vehicle_ids'] = 0
                info['group_type'] = group_type
                # Add states
                info['state'] = [0] * 7
                groups.append(info)

        return groups

    def vehicles_ids(self, idx, cluster_id):
        """Get vehicle id given the group id

        Parameters
        ----------
        idx : int
            An int specifying the group id
        cluster_id : array
            An array from the clustering function

        Returns
        -------
        list
            A list containing which vehicle belongs to a given cluster
        """
        vehicles_ids = cluster_id[cluster_id == idx]
        return vehicles_ids

    def n_vehicles(self, idx, cluster_id):
        """Get number of vehicles in the given cluster

        Parameters
        ----------
        idx : int
            An int specifying the group id
        cluster_id : array
            An array from the clustering function

        Returns
        -------
        int
            Number of vehicles in a given cluster
        """
        n = np.sum(cluster_id == idx)
        return n

    def get_state(self, uav, ugv):
        """Get the state of the mission."""

        """
        1) Remaining time 1 - added
        2) Prob. of the Target to be the Actual Target 3 - added
        3) Size of UxV Clusters 3 × 3 - not used here, size will be constant
        4) Distance of UxV Clusters to each other 3 × 3, not necessary, will leave this
        5) Dist. of UxV Clusters to  Pareto Nodes 3 × 3 × (5 + 3)

        """
        state = []

        # Update the states with time
        remaining_time = self.config['simulation']['total_time'] - self.current_time
        normalized_time = (remaining_time - 0) / (
            self.config['simulation']['total_time'] - 0
        )
        state.append([normalized_time])

        # Perform clustering on UAV and UGV
        n_ugv_clusters = self.config['simulation']['n_ugv_clusters']
        n_uav_clusters = self.config['simulation']['n_uav_clusters']

        cluster_id_ugv, ugv_cluster_pos = cluster(ugv, n_ugv_clusters, self.config)
        cluster_id_uav, uav_cluster_pos = cluster(uav, n_uav_clusters, self.config)

        # Perform pareto optimisation
        self.pareto_nodes = self.get_pareto_nodes_online()

        # Get pareto node position
        pareto_node_pos = []
        for node in self.pareto_nodes:
            node_info = self.map.get_node_info(node)
            position = self.map.convert_to_cartesian([node_info['y'], node_info['x']])
            pareto_node_pos.append(position[0:2])

        ugv_group = self.get_group_info(
            cluster_id_ugv, ugv_cluster_pos, 'ugv', pareto_node_pos
        )
        uav_group = self.get_group_info(
            cluster_id_uav, uav_cluster_pos, 'uav', pareto_node_pos
        )

        # Calculate distance between cluster and pareto node:
        all_distance = list()
        for i in range(n_uav_clusters):
            for j in range(len(pareto_node_pos)):
                distances = self.calculate_distance_pareto_vehicle(
                    uav_cluster_pos[i], pareto_node_pos[j]
                )
                all_distance.append(distances)
        for i in range(n_ugv_clusters):
            for j in range(len(pareto_node_pos)):
                distances = self.calculate_distance_pareto_vehicle(
                    ugv_cluster_pos[i], pareto_node_pos[j]
                )
                all_distance.append(distances)

        normalized_distance = (all_distance - np.min(all_distance)) / (
            np.max(all_distance) - np.min(all_distance)
        )

        # Add normalised distance
        state.append(normalized_distance)

        # Missing probability
        for i in range(3):
            state.append(self.target_manager.targets[i]["probability_goals"])

        # Convert everything into a list
        state = list(flatten_state_list(state))

        return state

    def update_progrees(self, uav, ugv):
        done = self.target_manager.update_progress(uav, ugv)
        return done

    def calculate_distance_pareto_vehicle(self, pareto_pos, vehicle_pos):
        dist = np.linalg.norm(pareto_pos - vehicle_pos)
        return dist
    
    def get_pareto_node(self):
        return self.pareto_nodes
