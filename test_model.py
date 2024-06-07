import yaml
import time

from utils import skip_run

from shasta.env import ShastaEnv
from shasta.preprocessing.utils import extract_building_info

from experiments.simple_experiment import SimpleExperiment
# from experiments.MRTA_Experiment import mrta_experiment
from experiments.MRTA_experiment import mrta_experiment
from experiments.exp_get_dist import dist_Experiment
from experiments.get_total_dist_point import distExperiment
from experiments.complex_experiment import SearchingExperiment
from experiments.actor_groups import create_actor_groups
from experiments.agents.uav import UaV
from experiments.agents.ugv import UgV
from stable_baselines3 import PPO

config_path = 'config/simulation_config.yml'
config = yaml.load(open(str(config_path)), Loader=yaml.SafeLoader)


#SimpleExperiment works with Gymnasium
# with skip_run('run', 'Test New Framework') as check, check():
#     n_actor_groups = 5
#     actor_groups = {}
#     for i in range(n_actor_groups): 
#         actor_groups[i] = UgV()
#     config['experiment']['type'] = mrta_experiment
#     env = ShastaEnv(config, actor_groups=actor_groups)
#     model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log='shasta_mrta/generalize')
#     # model = PPO("MlpPolicy", env, verbose=1)
#     model.learn(total_timesteps=1_00_000)
#     model.save("SearchExperiment")

n_actor_groups = 5
actor_groups = {}
for i in range(n_actor_groups): 
    actor_groups[i] = UgV()
config['experiment']['type'] = mrta_experiment
env = ShastaEnv(config, actor_groups=actor_groups)
req_model = PPO.load('SearchExperiment_new_reward', env=env)
observation, _ = env.reset()
while True:
    action, _ = req_model.predict(observation)
    observation, reward, done, truncated, info = env.step(action)
    if done == True:
        break


with skip_run('skip', 'Test Building') as check, check():
    osm_path = 'assets/buffalo-small/map.osm'
    extract_building_info(osm_path, save_fig=False)
