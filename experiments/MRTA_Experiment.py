from shasta.base_experiment import BaseExperiment
from shasta.primitives import  Formation
import numpy as np
import torch
from .agents.uav import UaV
from gymnasium import spaces

class mrta_experiment(BaseExperiment):
    def __init__(self, config, core, exp):
        super().__init__(config, core)
        self.core = core
        self.no_tasks = 30
        self.time_constraint = 45*60
        self.env_map = core.get_map()
        self.observation_space = []
        self.no_of_buildings = len(self.env_map.get_all_buildings())
        self.calc_dist = {}
        self.range =  np.array([10000], dtype=np.float32)
        self.actor_groups = core.get_actor_groups()
        self.assigned_location = np.zeros(shape=len(self.actor_groups))
        self.exhausted_agent = []
        self.target_positions = {}

        for k in range(len(self.actor_groups)):
            self.calc_dist[f'actor_{k}'] = 0
        # self.random_task_list = np.random.randint(1, self.no_of_buildings, self.no_tasks)
        self.random_task_list = np.arange(0, 30, 1)
        print(self.random_task_list, 'task_list')
        for k in range(self.no_tasks):
            cartesion_info = self.env_map.get_cartesian_node_position(self.random_task_list[k])
            if k == 0:
                self.random_task_location = np.array(cartesion_info)
            else:
                self.random_task_location = np.vstack((self.random_task_location, cartesion_info))

        print('depot_assigned: ', self.random_task_location[0])
        self.task_available = np.ones(self.no_tasks)
        self.distance_tr = np.array([0], dtype=np.float32)
        self.current_location = np.array([0, 0, 0])
        self.formation = Formation()
        self.check_reward = False

    def reset(self):
        """Called at the beginning and each time the simulation is reset"""
        # print('reset_called')
        # print('episode_ends: ', )
        for k in range(len(self.actor_groups)):
            self.calc_dist[f'actor_{k}'] = 0
            self.actor_groups[k][0].current_pos = self.random_task_location[0]
            self.target_positions[k] = self.actor_groups[k][0].current_pos
            # print("current_positions_reset_function: ", self.actor_groups[k][0].current_pos)
        # if self.check_reward:
        #     print('episode ends: ', sum(self.task_available), self.reward)
        self.assigned_location[0] = 1
        self.total_distance = 0
        self.time_step = 0
        self.distance_tr = np.array([0], dtype=np.float32)
        self.remaining_range = np.array([0], dtype=np.float32)
        self.task_available = np.ones(self.no_tasks)
        self.exhausted_agent = []
        
        # self.random_task_list = np.random.randint(1, self.no_of_buildings, self.no_tasks)
        # for k in range(self.no_tasks):
        #     cartesion_info = self.env_map.get_cartesian_node_position(self.random_task_list[k])
        #     if k == 0:
        #         self.random_task_location = np.array(cartesion_info)
        #     else:
        #         self.random_task_location = np.vstack((self.random_task_location, cartesion_info))

        pass

    def get_action_space(self):
        """Returns the action space"""
        return spaces.box.Box(0, 1, shape=(self.no_tasks, ))

    def get_observation_space(self):
        """Returns the observation space"""
        self.observation_space = spaces.Dict(
            {
                'remaining_range':spaces.box.Box(0, np.inf, shape=(1,), dtype=np.float32),
                'range':spaces.box.Box(0, np.inf, shape=(1,), dtype=np.float32),
                'current_location':spaces.box.Box(-np.inf, np.inf, shape=(3, )),
                'all_nodes': spaces.box.Box(-np.inf, np.inf, shape=(self.no_tasks, 3), dtype=np.float32), # the other way is to use MultiDiscrete
                'rem_locs':spaces.multi_binary.MultiBinary(n=self.no_tasks)
            }
        )
        return self.observation_space

    def get_actions(self):
        """Returns the actions"""
        pass

    def depot_configurations(self, agent):
        # print(self.random_task_location[0], self.random_task_location[1], 'depot_location in depot_configurations')
        self.actor_groups[agent][0].payload = 5
        self.actor_groups[agent][0].remaining_range = self.range
        if agent in self.exhausted_agent:
            self.exhausted_agent.remove(agent)

    def apply_actions(self, actions, core):
        """Given the action, returns a carla.VehicleControl() which will be applied to the hero

        :param action: value outputted by the policy
        """
        current_agent = np.argmax(self.assigned_location) 
        if current_agent in self.exhausted_agent or sum(self.assigned_location) == 0:
            done = False
            # print('just_action')
            if len(self.exhausted_agent) >= len(self.actor_groups):
                return None
            while not done:
                for i in range(len(self.actor_groups)):
                    if i in self.exhausted_agent:
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
            
            return None

        if self.actor_groups[current_agent][0].remaining_range <= 1 :
            self.time_step += 1
            self.exhausted_agent.append(current_agent)
            self.assigned_location[current_agent] = 0
            # print(self.exhausted_agent, " :exhausted agents")
            return None
        else:
            actions = actions.reshape(1, -1)*self.task_available
            soft = torch.nn.Softmax(dim=1)
            actions = soft(torch.Tensor(actions)).numpy()[0]
            selected_action = np.argmax(actions)
            # print('selected_action: ', selected_action, self.random_task_location[selected_action])
            # new_pos = self.random_task_location[selected_action]
            # new_pos[2] = 10
            self.target_positions[current_agent] = self.random_task_location[selected_action]
            # self.actor_groups[current_agent][0].target_pos = new_pos

            if selected_action != 0:
                self.task_available[selected_action] = 0

            if self.actor_groups[current_agent][0].payload == 0 and selected_action != 0:
                self.time_step += 1
                self.exhausted_agent.append(current_agent)
                self.assigned_location[current_agent] = 0
                # print('agent: ', current_agent, 'no payload')
                
                return None
            
            self.assigned_location[current_agent] = 0
            done = False

            while not done:
                for i in range(len(self.actor_groups)):
                    # print('current_actor_run: ', i, )
                    if i in self.exhausted_agent:
                        # print('agent', i, 'exhausted', self.exhausted_agent)
                        continue
                    # current_pos = self.actor_groups[i][0].current_pos
                    # new_pos = self.target_positions[i]
                    # print(current_pos, new_pos, i)
                    # print(self.random_task_location[0], self.random_task_location[1],
                    #     self.random_task_location[3],self.random_task_location[4],
                    #     self.random_task_location[5],  'depot_location_for_loop')
                    # print(self.target_positions[i], self.actor_groups[i][0].current_pos)
                    # input("current_new")
                    # print(self.random_task_location[0], self.random_task_location[1],
                    #     self.random_task_location[3],self.random_task_location[4],
                    #     self.random_task_location[5],  'depot_location_for_loop')
                    # print('--------after execute---------------')
                    # print(self.target_positions)
                    # print(self.target_positions[i], self.actor_groups[i][0].current_pos)
                    # input("after_execute")
                    # print('------------------')
                    # print(self.calc_dist, ": current_distance")
                    if np.round(self.actor_groups[i][0].current_pos[0], 1) == np.round(self.target_positions[i][0], 1) \
                        and np.round(self.actor_groups[i][0].current_pos[1], 1) == np.round(self.target_positions[i][1], 1):
                        # print('agent: ', i, ' reached')
                        # print(current_pos, new_pos, i)
                        # print(self.actor_groups[i][0].current_pos)
                        # print(self.target_positions)
                        # print(self.random_task_location[0], self.random_task_location[1],
                        # self.random_task_location[3],self.random_task_location[4],
                        # self.random_task_location[5],  'depot_location after_assign')
                        # input("current_new_reached")
                        self.distance_tr[0] = self.calc_dist[f'actor_{i}']
                        # print(self.calc_dist, ' :distance')
                        # print(self.distance_tr)
                        # print('=========distance===========')
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
                # print(self.random_task_location[0], self.random_task_location[1],
                #       self.random_task_location[3],self.random_task_location[4],
                #         self.random_task_location[5],  'depot_location after_assign')
                self.time_step += 1
            if selected_action == 0:
                # print(current_agent, ' goes to depo')
                self.depot_configurations(current_agent)


    def get_observation(self, observation, core):
        """Function to do all the post processing of observations (sensor data).

        :param sensor_data: dictionary {sensor_name: sensor_data}

        Should return a tuple or list with two items, the processed observations,
        as well as a variable with additional information about such observation.
        The information variable can be empty
        """
        observations = {}
        # print(observations)
        observations['all_nodes'] = self.random_task_location
        observations['remaining_range'] = self.remaining_range
        observations['range'] = self.range
        observations['rem_locs'] = self.task_available
        observations['current_location'] = self.current_location


        # self.calc_dist[f'actor_{i}'] = 0
        return observations, {}

    def get_truncated_status(self, observation, core):
        """Computes the reward"""
        return False

    def get_done_status(self, observation, core):
        """Returns whether or not the experiment has to end"""
        # if sum(self.observation_space) == self.no_of_buildings:
            # return True
        # print(self.time_step, self.time_constraint)
        # print(len(self.actor_groups))
        # print(self.time_step, self.time_constraint, len(self.exhausted_agent))
        # print('get done status: ', sum(self.task_available), self.exhausted_agent)
        # input('task available')
        if self.time_step >= self.time_constraint or sum(self.task_available) == 1 or len(self.exhausted_agent) >= len(self.actor_groups):
            done = True
        else :
            done = False

        return done
    
    def compute_end_of_episode_reward(self):
        # print(self.total_distance, 'distance_travelled so far')
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
            # print(self.time_constraint, self.time_constraint)
            reward = self.compute_end_of_episode_reward() 
        else:
            reward = -1
        
        self.reward = reward
        self.check_reward = True
        return reward