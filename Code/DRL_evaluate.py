#This fucntion defines the evaluation function for the trained DRL model. The results will be summarized in progress.csv and monitor.csv files.
#import packages
import os
import sys
import pprint
import sys
from datetime import datetime
import gymnasium as gym
import numpy as np
import stable_baselines3
from stable_baselines3 import *
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.logger import HumanOutputFormat
from stable_baselines3.common.logger import Logger as SB3Logger
from stable_baselines3.common.monitor import Monitor
import sinergym
from sinergym.utils.callbacks import *
from sinergym.utils.constants import *
from sinergym.utils.logger import CSVLogger, WandBOutputFormat
from sinergym.utils.rewards import *
from sinergym.utils.wrappers import *
from sinergym.utils.common import get_ids
from typing import Any, Dict, Optional, Sequence, Tuple, Union, List

class CustomCSVLogger(CSVLogger):

    def __init__(
            self,
            monitor_header: str,
            progress_header: str,
            log_progress_file: str,
            log_file: Optional[str] = None,
            flag: bool = True):
        super(CustomCSVLogger, self).__init__(monitor_header,progress_header,log_progress_file,log_file,flag)
        self.last_10_steps_reward = [0]*10

    def _create_row_content(
            self,
            obs: List[Any],
            action: Union[int, np.ndarray, List[Any]],
            terminated: bool,
            truncated: bool,
            info: Optional[Dict[str, Any]]) -> List:

        if info.get('reward') is not None:
            self.last_10_steps_reward.pop(0)
            self.last_10_steps_reward.append(info['reward'])


        return [
            info.get('timestep',0)] + list(obs) + list(action) + [
            info.get('time_elapsed(hours)',0),
            info.get('reward',None),
            info.get('energy_term'),
            info.get('ITE_term'),
            info.get('comfort_term'),
            terminated,
            truncated]


def evaluate_model(trace_path, environment, experiment_name, building_file, weather_file,
                   config_params, training_mean, training_var, model):
    # initialize trace
    util_rate=np.loadtxt(trace_path, dtype='float')
    # Initialize environment
    eval_env = gym.make(environment, 
                        env_name=experiment_name, 
                        building_file=building_file, 
                        weather_files=weather_file, 
                        config_params=config_params,
                        evaluation_flag=1)
    
    eval_env = NormalizeObservation(eval_env, mean=training_mean, var=training_var, automatic_update=False)
    eval_env = LoggerWrapper(eval_env, logger_class=CustomCSVLogger,
                             monitor_header=['timestep'] + 
                             eval_env.get_wrapper_attr('observation_variables') +
                             eval_env.get_wrapper_attr('action_variables') + 
                             ['time (hours)', 'reward', 'energy_term', 'comfort_term', 
                              'terminated', 'truncated'])
    
    print(eval_env.get_wrapper_attr('observation_variables'))

    obs, info = eval_env.reset()
    state = None
    truncated = terminated = False
    rewards = []
    energy_term = []
    comfort_term = []
    current_month = 0
    count = 0

    while not (terminated or truncated):
        action, state = model.predict(obs)
        action[5]=util_rate[count]
        count += 1
        obs, reward, terminated, truncated, info = eval_env.step(action)
        rewards.append(reward)
        energy_term.append(info['energy_term'])
        comfort_term.append(info['comfort_term'])

        if info['month'] != current_month:
            current_month = info['month']
            print('Reward: ', sum(rewards))
            print('Mean Reward: ', np.mean(rewards))
            print('Info: ', info)

    print('Mean reward:', np.mean(rewards))
    eval_env.close()
