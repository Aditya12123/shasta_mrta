from shasta.base_experiment import BaseExperiment
from shasta.primitives import  Formation

from .agents.uav import UaV
from gymnasium import spaces

class distExperiment(BaseExperiment):
    def __init__(self, config, core, exp):
        super().__init__(config, core)
        self.env_map = core.get_map()
        self.observation_space = []
        self.no_of_buildings = len(self.env_map.get_all_buildings())
        for i in range(self.no_of_buildings):
            self.observation_space.append(0)
        self.formation = Formation()
        self.actor_groups = core.get_actor_groups()

    def reset(self):
        """Called at the beginning and each time the simulation is reset"""
        pass

    def get_action_space(self):
        """Returns the action space"""
        return spaces.multi_discrete.MultiDiscrete([self.no_of_buildings,self.no_of_buildings,self.no_of_buildings])

    def get_observation_space(self):
        """Returns the observation space"""
        return spaces.Box(low=0, high=1, shape=(self.no_of_buildings,))

    def get_actions(self):
        """Returns the actions"""
        pass

    def apply_actions(self, actions, core):
        """Given the action, returns a carla.VehicleControl() which will be applied to the hero

        :param action: value outputted by the policy
        """
        print(actions)
        while True:
            Status = []
            for i in range(3):
                current_pos = self.actor_groups[i][0].current_pos
                new_pos = self.env_map.get_cartesian_node_position(actions[i])
                print(new_pos)
                asdfdasf
                new_pos[2] = 10
                self.formation.execute(self.actor_groups[i], new_pos, current_pos, 'solid')
                if int(current_pos[0]) == int(new_pos[0]) and int(current_pos[1]) == int(new_pos[1]):
                    Status.append(1)
                    self.observation_space[actions[i]] = 1
            if sum(Status) == 3:
                break
            core.tick()

    def get_observation(self, observation, core):
        """Function to do all the post processing of observations (sensor data).

        :param sensor_data: dictionary {sensor_name: sensor_data}

        Should return a tuple or list with two items, the processed observations,
        as well as a variable with additional information about such observation.
        The information variable can be empty
        """
        return self.observation_space, {}

    def get_truncated_status(self, observation, core):
        """Computes the reward"""
        return False

    def get_done_status(self, observation, core):
        """Returns whether or not the experiment has to end"""
        if sum(self.observation_space) == self.no_of_buildings:
            return True
        return False

    def compute_reward(self, observation, core):
        """Computes the reward"""
        return sum(self.observation_space)
