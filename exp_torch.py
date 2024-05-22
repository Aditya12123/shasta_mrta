from stable_baselines3 import PPO
from gym import spaces
import gym
import numpy as np

x = np.array([[2, 4], [1, 3], [5, 7], [7, 8]])
observation_space = spaces.Discrete(n=5)
# observation_space = gym.spaces.Box(0, 1, shape=(10, 2))
# observation_space = spaces.Dict(
#             {
#                 'distance_tr':spaces.Box(0, np.inf, shape=(1,), dtype=np.float32),
#                 'range':spaces.Box(0, np.inf, shape=(1,), dtype=np.float32),
#                 'current_location':spaces.Box(-np.inf, np.inf, shape=(2, )),
#                 'all_nodes': spaces.Box(-np.inf, np.inf, shape=(30, 2)), # the other way is to use MultiDiscrete
#                 # 'rem_locs':spaces.multi_binary.MultiBinary(n=30)
#             })
print(observation_space.sample())

# a = np.ones(shape=(1, 5))
# b = np.array([1 , 0, 1, 0, 1])

# print(a*b)