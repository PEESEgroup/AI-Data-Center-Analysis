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


def evaluate_model(environment, experiment_name, building_file, weather_file,
                   config_params, training_mean, training_var, model):
    # Initialize environment
    eval_env = gym.make(environment, 
                        env_name=experiment_name, 
                        building_file=building_file, 
                        weather_files=weather_file, 
                        config_params=config_params)
    
    eval_env = NormalizeObservation(eval_env, mean=training_mean, var=training_var, automatic_update=False)
    eval_env = NormalizeAction(eval_env)
    eval_env = LoggerWrapper(eval_env, logger_class=CustomCSVLogger,
                             monitor_header=['timestep'] + 
                             eval_env.get_wrapper_attr('observation_variables') +
                             eval_env.get_wrapper_attr('action_variables') + 
                             ['time (hours)', 'reward', 'energy_term', 'ITE_term', 'comfort_term', 
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

    print('Episode', locations, 'Mean reward:', np.mean(rewards))
    eval_env.close()
