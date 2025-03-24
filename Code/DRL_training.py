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

# Get the list of available environments
sinergym_environment_ids = get_ids()
print(sinergym_environment_ids)

# Environment Parameter definition
environment = "Eplus-DC-Cooling"
building_file_1=['DRL_DC.epJSON']
whole_weather_file=['AUS_NSW_Sydney.Intl.AP.947670_TMYx.2009-2023.epw', 'DEU_HE_Frankfurt.AP.106370_TMYx.2009-2023.epw',
                    'SGP_SG_Singapore-Changi.Intl.AP.486980_TMYx.2009-2023.epw', 'SWE_NB_Lulea.AP.021860_TMYx.2009-2023.epw',
                    'USA_CA_San.Francisco.Intl.AP.724940_TMYx.2009-2023.epw', 'USA_IA_Des.Moines.Intl.AP.725460_TMYx.2009-2023.epw',
                    'USA_NE_Omaha-Eppley.AF.Intl.AP.725500_TMYx.2009-2023.epw', 
                    'USA_NY_New.York-Kennedy.Intl.AP.744860_TMYx.2009-2023.epw',
                    'USA_TX_Dallas-Fort.Worth.Intl.AP.722590_TMYx.2009-2023.epw',
                    'USA_VA_Dulles-Washington.Dulles.Intl.AP.724030_TMYx.2009-2023.epw']
config_params={
    'runperiod':(1,1,2025,31,12,2025),
    'timesteps_per_hour':1
}
#Name of the experiment
experiment_date = datetime.today().strftime('%Y-%m-%d_%H:%M')
experiment_name = 'SAC_DC_Cooling-' + environment
experiment_name = experiment_date + '_' + experiment_name

# Define the monitor file to output simulation results
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
            info.get('comfort_term'),
            terminated,
            truncated]

# Define the training process
env = gym.make(environment, env_name=experiment_name, building_file=building_file_1,
              weather_files=whole_weather_file, config_params=config_params)
env = NormalizeObservation(env)
env = (env)
env = LoggerWrapper(env, logger_class=CustomCSVLogger,monitor_header = ['timestep'] + env.get_wrapper_attr('observation_variables') +
                env.get_wrapper_attr('action_variables') + ['time (hours)', 'reward', 'energy_term', 'ITE_term', 'comfort_term', 
                'terminated', 'truncated'])
policy_kwargs= dict(net_arch=[512])
model = SAC("MlpPolicy", env, batch_size=512, learning_rate=5e-5, learning_starts=8760, gamma=0.99, policy_kwargs=policy_kwargs)
episodes = 300
timesteps = episodes * (env.get_wrapper_attr('timestep_per_episode')-1)

model.learn(
    total_timesteps=timesteps,
    log_interval=500)
model.save(env.get_wrapper_attr('workspace_path') + '/model')
