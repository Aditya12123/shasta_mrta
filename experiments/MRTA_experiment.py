from shasta.base_experiment import BaseExperiment
from shasta.primitives import  Formation
import numpy as np
import torch
from .agents.uav import UaV
from gymnasium import spaces

"""
This file creates MRTA decentralized setup to train a policy. The RL environment is defined in shasta/env and the reset and step functions in that file calls 
the functions implemented here to get the observation and execute actions. 


"""

class mrta_experiment(BaseExperiment):
    def __init__(self, config, core, exp):
        super().__init__(config, core)
        self.core = core
        self.no_tasks = 30      #required number of delivery locations
        self.time_constraint = 45*60
        self.max_payload = 5
        self.env_map = core.get_map()
        self.observation_space = []
        self.no_of_buildings = len(self.env_map.get_all_buildings())
        self.calc_dist = {}
        self.range =  np.array([10000], dtype=np.float32)
        self.payload = np.array([5], dtype=np.float32)
        self.actor_groups = core.get_actor_groups()
        self.assigned_location = np.zeros(shape=len(self.actor_groups))

        self.exhausted_agent = [] # keeps track of which agents ran out of battery or payload capacity. They can no longer go to any locations
        self.target_positions = {}

        for k in range(len(self.actor_groups)):
            self.calc_dist[f'actor_{k}'] = 0

        #Initialize the 30 (no of tasks) random locations. Task 0 is the depot
        self.random_task_list = np.random.randint(1, self.no_of_buildings, self.no_tasks)  
        for k in range(self.no_tasks):
            cartesion_info = self.env_map.get_cartesian_node_position(self.random_task_list[k])
            if k == 0:
                self.random_task_location = np.array(cartesion_info)
            else:
                self.random_task_location = np.vstack((self.random_task_location, cartesion_info))

        self.task_available = np.ones(self.no_tasks) # keeps track of which locations are still not visited
        self.distance_tr = np.array([0], dtype=np.float32)     
        self.current_location = np.array([0, 0, 0])
        self.formation = Formation()
        self.check_reward = False

    def reset(self):
        """Called at the beginning and each time the simulation is reset
        
        At each reset:
         - randomly initialize the building locations to make the policy generalizable
         - Make the current location of all the robots as the depot
         - Reset the variables - distance travelled, time steps, remaining range, visited locations to their default values
        """

        self.random_task_list = np.random.randint(1, self.no_of_buildings, self.no_tasks)
        for k in range(self.no_tasks):
            cartesion_info = self.env_map.get_cartesian_node_position(self.random_task_list[k])
            if k == 0:
                self.random_task_location = np.array(cartesion_info)
            else:
                self.random_task_location = np.vstack((self.random_task_location, cartesion_info))


        for k in range(len(self.actor_groups)):
            self.calc_dist[f'actor_{k}'] = 0
            self.actor_groups[k][0].current_pos = self.random_task_location[0]
            self.target_positions[k] = self.actor_groups[k][0].current_pos

        self.assigned_location[0] = 1
        self.total_distance = 0
        self.time_step = 0
        self.distance_tr = np.array([0], dtype=np.float32)
        self.remaining_range = np.array([0], dtype=np.float32)
        self.task_available = np.ones(self.no_tasks)
        self.payload = np.array([5], dtype=np.float32)
        self.exhausted_agent = []
        

        pass

    def get_action_space(self):
        """Returns the action space"""
        return spaces.box.Box(0, 1, shape=(self.no_tasks, ))

    def get_observation_space(self):
        """Returns the observation space"""
        self.observation_space = spaces.Dict(
            {
                'remaining_range':spaces.box.Box(0, np.inf, shape=(1,), dtype=np.float32),
                # 'range':spaces.box.Box(0, np.inf, shape=(1,), dtype=np.float32),
                'current_location':spaces.box.Box(-np.inf, np.inf, shape=(3, )),
                'all_nodes': spaces.box.Box(-np.inf, np.inf, shape=(self.no_tasks, 3), dtype=np.float32), # the other way is to use MultiDiscrete
                'rem_locs':spaces.multi_binary.MultiBinary(n=self.no_tasks),
                'payload':spaces.box.Box(0, 5, shape=(1, ), dtype=np.float32)
            }
        )
        return self.observation_space

    def get_actions(self):
        """Returns the actions"""
        pass

    def depot_configurations(self, agent):
        """
        This function is called whenever the robot goes to the depot. That is if the policy outputs the depot location.
        It will recharge itself and also get the payload.
        """
        self.actor_groups[agent][0].payload = 5
        self.actor_groups[agent][0].remaining_range = self.range
        if agent in self.exhausted_agent:
            self.exhausted_agent.remove(agent)

    def execute_movement(self, core):
        """
        This function implements decentralized movement of the robots according to their planned locations.
        You get out of while loop when a robot has reached its planned location
        """
        done = False
        while not done:
            for i in range(len(self.actor_groups)):
                if i in self.exhausted_agent:
                    # do not execute movement of the robot which is exhausted. Hence, the target_position of that robot will still be available to the policy
                    continue
                if np.round(self.actor_groups[i][0].current_pos[0], 1) == np.round(self.target_positions[i][0], 1) \
                    and np.round(self.actor_groups[i][0].current_pos[1], 1) == np.round(self.target_positions[i][1], 1):

                    self.distance_tr[0] = self.calc_dist[f'actor_{i}']
                    self.total_distance += self.distance_tr[0]
                    self.assigned_location[i] = 1
                    self.actor_groups[i][0].payload -= 1 
                    self.actor_groups[i][0].remaining_range -= self.distance_tr
                    self.remaining_range = self.actor_groups[i][0].remaining_range
                    self.payload = self.actor_groups[i][0].payload 
                    self.current_location = self.actor_groups[i][0].current_pos
                    done = True                    
                else:
                    self.formation.execute(self.actor_groups[i], self.target_positions[i], self.actor_groups[i][0].current_pos, 'solid')
                    self.calc_dist[f'actor_{i}'] += self.formation.speed[0]*self.formation.dt
            core.tick()
            self.time_step += 1

    def apply_actions(self, actions, core):
        """Given the action, returns a carla.VehicleControl() which will be applied to the hero

        :param action: value outputted by the policy
        
        The action gives a number from 1-30 meaning the id of the location to be visited. Then, a decentralized execution of the robot movement is done
        through execute_movement(). That function stops executing once a robot reached its planned location. The remaining range and payload of the robots 
        get updated in execute_movement() which are passed as inputs to the policy (among other inputs) through get_observations() function. Thus, the action 
        received in this function will be for the same robot that reached its planned point. Again, after getting action decentralized execution of the
        movement continues and whichever robot reaches its planned location gives the policy its states and gets action(new location to visit)

        """

        # self.assigned_location is 0 for all the robots except for the robot which has reached its planned location
        current_agent = np.argmax(self.assigned_location) 
        if current_agent in self.exhausted_agent or sum(self.assigned_location) == 0:
            # if current_agent is exhausted, we keep executing movement of the rest of the drones
            if len(self.exhausted_agent) >= len(self.actor_groups):
                # terminate if all the agents are exhausted
                return None
            self.execute_movement(core=core)
            return None

        if self.actor_groups[current_agent][0].remaining_range <= 1 :
            # if the current robot does not have the battery capacity to visit the next location, terminate this operation
            self.time_step += 1
            self.exhausted_agent.append(current_agent)
            self.assigned_location[current_agent] = 0
            return None
        else:

            actions = actions.reshape(1, -1)*self.task_available # masking the actions which are not available
            soft = torch.nn.Softmax(dim=1) # taking softmax over available actions
            actions = soft(torch.Tensor(actions)).numpy()[0]    
            selected_action = np.argmax(actions)    # selection action with max prob
            self.target_positions[current_agent] = self.random_task_location[selected_action]

            if selected_action != 0:
                self.task_available[selected_action] = 0    # update locations available

            if self.actor_groups[current_agent][0].payload == 0 and selected_action != 0:
                # If robot does not have payload and the output of policy is not depot, add that robot to exhausted agent
                self.time_step += 1
                self.exhausted_agent.append(current_agent)
                self.assigned_location[current_agent] = 0
                
                return None
            
            self.assigned_location[current_agent] = 0   # this robot has been assigned location. So updating that to 0, this list gets updated in execute_movement()
                                                        # depending on which robot reaches its own assigned location
            self.execute_movement(core=core)
            if selected_action == 0:
                # If output of the policy is depo, update payload and range
                self.depot_configurations(current_agent)


    def get_observation(self, observation, core):
        """Function to do all the post processing of observations (sensor data).

        :param sensor_data: dictionary {sensor_name: sensor_data}

        Should return a tuple or list with two items, the processed observations,
        as well as a variable with additional information about such observation.
        The information variable can be empty
        """
        observations = {}
        # print(self.payload)
        observations['all_nodes'] = self.random_task_location
        observations['remaining_range'] = self.remaining_range
        observations['rem_locs'] = self.task_available
        observations['current_location'] = self.current_location
        observations['payload'] = self.payload


        # self.calc_dist[f'actor_{i}'] = 0
        return observations, {}

    def get_truncated_status(self, observation, core):
        """Computes the reward"""
        return False

    def get_done_status(self, observation, core):
        """Returns whether or not the experiment has to end
        Experiment ends:
         - if time steps cross the time constraints
         - if all locations are visited
         - all agents are exhausted
        """

        if self.time_step >= self.time_constraint or sum(self.task_available) == 1 or len(self.exhausted_agent) >= len(self.actor_groups):
            done = True
        else :
            done = False

        return done
    
    def compute_end_of_episode_reward(self):
        if sum(self.task_available) == 1:
            reward = 10
        else:
            reward = -(sum(self.task_available))

        return reward

    def compute_reward(self, observation, core):
        """Computes the reward"""
        if sum(self.task_available) == 1:
            reward = 10
        elif self.time_step >= self.time_constraint or len(self.exhausted_agent) >= len(self.actor_groups):
            reward = self.compute_end_of_episode_reward() 
        else:
            reward = -1
        
        self.reward = reward
        self.check_reward = True
        return reward