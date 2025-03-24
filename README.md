# AI-Data-Center-Analysis
This study presents the first framework to integrate deep reinforcement learning (DRL) and cost-effectiveness optimizations with sustainability metrics, including energy, water, and carbon efficiencies, alongside renewable energy adoption strategies for AI data centers. The following contents are included in this repository to support our key findings:
1. Codes: include major codes to conduct the estimation process.
2. Data: include data used during the analysis.
3. sinergym.tar.gz: include the revised sinergym package used in this work

## Requirements
To run the codes in this repository, the following software must be installed (version is given for refenrence):
- gyamnasium 0.29.1
- numpy 2.1.1
- opyplus 2.0.7
- pandas 2.2.3
- pyomo 6.9.1
- stable-baselines3 2.3.2
- EnergyPlus 23.1.0
- Ubuntu 22.04
- sinergym 2.5.0 (please use the docker file provided in https://github.com/ugr-sail/sinergym to install dependencies)

The tools mentioned above require relatively complex environment configurations. A detailed tutorial is available on the introduction page of the Sinergym package: https://ugr-sail.github.io/sinergym/compilation/main/pages/installation.html. Please note that we have made several modifications to the original Sinergym package files for a better performance. To apply these changes, simply download the sinergym.tar.gz file, extract its contents, and overwrite the original Sinergym package.

## Codes
- **DRL_training.py**: the file defines the DRL model training process.
- **DRL_evaluate.py**: the file defines the evaluation process for trained DRL model.
- **Demand Response Potential.py**: the file defines the optimization problem for evaluating the maximum BESS-based demand response potential.
- **Cost_Effective_Renewable_Adoption.py**: the file defines the optimization problem for generating the cost-effective renewable energy adoption strategies.

## Data
- **Grid Data folder**: contains the recent two year grid carbon and price cost data of considered locations.
- **buildings folder**: contains the EnergyPlus model used for the DRL and baseline controller
- **weather folder**: contains the weather data of considered locations.

## Running the code


## Citation
Please use the following citation when using the data, methods or results of this work:

Xiao, T., You, F., Leveraging AI to Explore Cost-Effective Decarbonization Solutions for AI Data Centers. Submitted to Nature Climate Change.



