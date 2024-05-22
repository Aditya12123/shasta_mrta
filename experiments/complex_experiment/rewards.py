import numpy as np


class Reward:
    def __init__(self, config):
        super(Reward, self).__init__()
        self.config = config

    def get_time_dist(self, vehicle, target_pos):
        """Calculated the information of probable goal building

        Parameters
        ---------
        vehicle : vehicle class
            We use the speed and current position
        target_pos : array
            final desired position of robot

        Returns
        -------
        float
            time to reach the target [minimum time]
        """
        # Read from co-ordinate file
        diff = np.asarray(vehicle.current_pos[0:2]) - np.asarray(target_pos)
        distance = np.linalg.norm(diff)
        if vehicle.type == 'uav':
            time_to_reach = distance / vehicle.speed
        elif vehicle.type == 'ugv':
            time_to_reach = (
                self.config['ugv']['coef_slowness'] * distance / vehicle.speed
            )
        return time_to_reach

    def goal_information(self, goal_id, config):
        """Calculates the information of probable goal building

        Parameters
        ----------
        goal_id : int
            Goal ID
        config : yaml
            The configuration file

        Returns
        -------
        dict
            A dictionary containing goal position, perimeter, floors,
            progress, and probability
        """
        # Read from co-ordinate file
        node_info = self.target_info(goal_id)
        info = {}
        info['goal_position'] = node_info['position']
        info['perimeter'] = node_info['perimeter']
        info['floors'] = node_info['n_floors']
        info['goal_progress'] = 0
        info['goal_probability'] = 0
        return info

    def mission_reward(self, ugv, uav, config):
        """Caculated the total mission reward depending on the progress

        Parameters
        ----------
        ugv : list
            List of all UGVs
        uav : list
            List of all UAVs
        config : yaml
            The configuration file

        Returns
        -------
        float
            The reward for the mission at an instant of time.
        """

        # Simulation parameters
        total_time = config['simulation']['total_time']
        goals = config['simulation']['goal_node']

        # UAV reward weight parameters
        w_time_uav = config['weights']['w_time_uav']
        w_battery_uav = config['weights']['w_battery_uav']
        w_b_uav_0 = 1  # Need to implement

        # Reward for UAV
        r_uav_time = 0
        r_uav_battery = 0

        # Calculate the reward for UAV
        for vehicle in uav:
            r_uav_battery += w_battery_uav * vehicle.battery / w_b_uav_0
            for goal in goals:
                position, _ = vehicle.get_pos_and_orientation()
                info = self.goal_information(goal, self.config)
                time_to_goal = self.get_time_dist(vehicle, info['goal_position'])
                r_uav_time += (
                    w_time_uav
                    * (1 - info['goal_progress'])
                    * (total_time - time_to_goal)
                    / total_time
                )

        # Reward for UGV
        r_ugv_time = 0
        r_ugv_ammo = 0

        # UGV reward weight parameters
        w_time_ugv = config['weights']['w_time_ugv']
        w_battery_ugv = config['weights']['w_ammo_ugv']
        w_b_ugv_0 = 1  # Need to implement

        # Calculate the reward for UGV
        for vehicle in ugv:
            r_ugv_ammo += w_battery_ugv * vehicle.ammo / w_b_ugv_0
            for goal in goals:
                position, _ = vehicle.get_pos_and_orientation()
                info = self.goal_information(goal, self.config)
                time_to_goal = self.get_time_dist(vehicle, info['goal_position'])
                r_ugv_time += (
                    w_time_ugv
                    * (1 - info['goal_progress'])
                    * (total_time - time_to_goal)
                    / total_time
                )

        # Search reward parameters
        w_search = self.config['weights']['w_search']
        # mission_success = self.config['weights']['mission_success']
        r_search = 0
        for target in self.target:
            r_search += w_search * target['progress_goals']

        reward = r_ugv_time + r_ugv_ammo + r_uav_time + r_uav_battery + r_search

        return reward
