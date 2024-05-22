import numpy as np
from numpy import genfromtxt


class TargetManager:
    def __init__(self, experiment_config, core):
        super(TargetManager, self).__init__()
        # Need to specify some parameters
        self.config = experiment_config
        self.map = core.get_map()

        # Initial setup
        self._initial_mission_setup()
        self._initial_target_setup()

    def _initial_mission_setup(self):
        self.goal = self.config['simulation']['goal_node']
        self.progress_reward = self.config['reward']['progress_reward']
        self.indoor_reward = 2 * self.progress_reward
        self.n_keep_in_pareto = self.config['state']['n_keep_in_pareto']

    def _initial_target_setup(self):
        """Performs target setup with properties such as goal probability,
        goal progress etc.
        """
        self.buildings = self.map.buildings
        # Targets building
        mask = self.buildings['id'].isin(
            self.config['simulation']['target_building_id']
        )

        # Add more information
        n_targets = self.config['simulation']['n_targets']

        targets_info = self.buildings.loc[mask].copy()
        targets_info['probability_goals'] = 1 / n_targets
        targets_info['progress_goals'] = 0
        targets_info['probability_goals_indoor'] = 1 / n_targets
        targets_info['progress_goals_indoor'] = 0
        targets_info['defence_perimeter'] = 0
        targets_info['n_defence_perimeter'] = self.buildings['perimeter'] / (
            self.config['ugv']['defense_radius'] * 2
        )

        self.targets = targets_info.to_dict('records')
        return None

    def get_target_info(self, id):
        """Get the information about the target.

        Parameters
        ----------
        id : int
            Target ID

        Returns
        -------
        dict
            A dictionary containing all the information about the target.
        """
        for target in self.targets:
            if target['id'] == id:
                return target

    def get_node_info(self, id):
        return self.map.get_node_info(id)

    def check_vehicle(self, vehicle):
        if (not vehicle.idle) and vehicle.type == 'uav':
            return 'uav'
        elif (not vehicle.idle) and vehicle.type == 'ugv':
            return 'ugv'
        else:
            return None

    def check_closeness(self, vehicle, target):
        target_pos = target['position']
        vehicle_pos = vehicle.current_pos
        dist = np.linalg.norm(np.asarray(vehicle_pos[0:2]) - np.asarray(target_pos))
        if vehicle.type == 'uav':
            return dist <= self.config['uav']['search_dist']
        elif vehicle.type == 'ugv':
            return dist <= self.config['ugv']['defense_radius']
        else:
            return None

    def outdoor_progress(self, vehicle, target):
        req_progress = target['n_floors'] * target['perimeter']
        progesse_rate = vehicle.search_speed / req_progress
        progress_goals = self.config['simulation']['time_step'] * progesse_rate
        return progress_goals

    def indoor_progress(self, vehicle, target):
        req_progress = target['n_floors'] * target['area']
        progesse_rate = vehicle.search_speed / req_progress
        progress_goals = self.config['simulation']['time_step'] * progesse_rate
        return progress_goals

    def outdoor_target_progress_update(self, vehicles):  # noqa
        for target in self.targets:
            progress_goals = 0
            for vehicle in vehicles:
                if self.check_vehicle(vehicle) == 'uav':
                    if self.check_closeness(vehicle, target):
                        progress_goals += self.outdoor_progress(vehicle, target)
            if progress_goals > 1:
                progress_goals = 1
            target['progress_goals'] = target['progress_goals'] + progress_goals
            if target['progress_goals'] > 1:
                target['progress_goals'] = 1

        sum_progress = 0
        found_goal = 0
        for j, target in enumerate(self.targets):
            if target['id'] == self.config['simulation']['goal_node']:  # found the goal
                if target['progress_goals'] >= np.random.rand():
                    # if it finds it in out of building indoor search
                    # target is guaranteed
                    for i in range(len(self.targets)):
                        if i == j:
                            self.targets[i]['probability_goals'] = 1
                            self.targets[i]['probability_goals_indoor'] = 1
                            self.targets[i]['probability_goals_outdoor'] = 1
                        else:
                            self.targets[i]['probability_goals'] = 0
                            self.targets[i]['probability_goals_indoor'] = 0
                            self.targets[i]['probability_goals_outdoor'] = 0
                    found_goal = 1
                    break
            sum_progress += self.targets[j]['progress_goals']

        if found_goal == 0:
            for target in self.targets:
                target['probability_goals_outdoor'] = (1 - target['progress_goals']) / (
                    len(self.targets) - sum_progress
                )  # 1- progress is probability of each of them
                target['probability_goals'] = (
                    target['probability_goals_outdoor']
                    + target['probability_goals_indoor']
                ) / 2
        return None

    def indoor_target_progress_update(self, vehicles):  # noqa
        for target in self.targets:
            progress_goals = 0
            vehicle_count = 0

            for vehicle in vehicles:
                if self.check_vehicle(vehicle) == 'ugv':
                    if self.check_closeness(vehicle, target):
                        vehicle_count += 1
                        if target['n_defence_perimeter'] < vehicle_count:
                            progress_goals += self.indoor_progress(vehicle, target)
            if progress_goals > 1:
                progress_goals = 1
            target['progress_goals'] = target['progress_goals'] + progress_goals
            if target['progress_goals'] > 1:
                target['progress_goals'] = 1

        sum_progress = 0
        found_goal = 0
        for j, target in enumerate(self.targets):
            if target['id'] == self.config['simulation']['goal_node']:  # found the goal
                if target['progress_goals'] >= np.random.rand():
                    # if it finds it in out of building indoor search
                    # target is guaranteed
                    for i in range(len(self.targets)):
                        if i == j:
                            self.targets[i]['probability_goals'] = 1
                            self.targets[i]['probability_goals_indoor'] = 1
                            self.targets[i]['probability_goals_outdoor'] = 1
                        else:
                            self.targets[i]['probability_goals'] = 0
                            self.targets[i]['probability_goals_indoor'] = 0
                            self.targets[i]['probability_goals_outdoor'] = 0
                    found_goal = 1
                    break
            sum_progress += self.targets[j]['progress_goals']

        if found_goal == 0:
            for target in self.targets:
                target['probability_goals_outdoor'] = (1 - target['progress_goals']) / (
                    len(self.targets) - sum_progress
                )  # 1- progress is probability of each of them

        return None

    def update_progress(self, uav, ugv):
        """Update all the probability"""
        self.indoor_target_progress_update(uav)
        self.outdoor_target_progress_update(ugv)
        done = False
        if self.targets[1]['probability_goals_outdoor'] == 0:
            done = True
        return done
