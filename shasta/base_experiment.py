#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
from abc import ABCMeta, abstractmethod
from future.utils import with_metaclass


class BaseExperiment(with_metaclass(ABCMeta, object)):
    def __init__(self, config, core, experiment_config=None, *args, **kargs):
        """The base experiment which other experiments should inherit

        Parameters
        ----------
        config : yaml
            The experiment configuration file
        core : object
            The SHASTA core object

        Returns
        -------
        None
        """
        self.config = config
        self.core = core
        self.exp_config = experiment_config

        return None

    def reset(self):
        """Called at the beginning and each time the simulation is reset"""
        pass

    @abstractmethod
    def get_action_space(self):
        """Returns the action space"""
        raise NotImplementedError

    @abstractmethod
    def get_observation_space(self):
        """Returns the observation space"""
        raise NotImplementedError

    def get_actions(self):
        """Returns the actions"""
        raise NotImplementedError

    @abstractmethod
    def apply_actions(self, actions, core):
        """Given the action, applies the action to group or individual actor

        :param action: value outputted by the policy
        """
        raise NotImplementedError

    @abstractmethod
    def get_observation(self, observation, core):
        """Function to do all the post processing of observations (sensor data).

        :param sensor_data: dictionary {sensor_name: sensor_data}

        Should return a tuple or list with two items, the processed observations,
        as well as a variable with additional information about such observation.
        The information variable can be empty
        """
        return NotImplementedError

    def get_done_status(self, observation, core):
        """Returns whether or not the experiment has to end"""
        return NotImplementedError

    def compute_reward(self, observation, core):
        """Computes the reward"""
        return NotImplementedError
