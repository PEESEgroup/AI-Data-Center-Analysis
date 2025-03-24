from pyomo.environ import *
from pyomo.environ import value
import pandas as pd
import numpy as np
import os
import time

# --- Import data based on installation path ---
DRL_data_dir = ['PATH_TO_DRL_DATA']          # data path list of DRL-based DC model outputs (monitor.csv)
BAS_data_dir = ['PATH_TO_BAS_DATA']          # data path list of baseline DC model outputs (monitor.csv)
grid_data_dir = 'PATH_TO_GRID_DATA'            # data path of grid factors
optimization_output_dir = 'PATH_TO_OPTIMIZATION_OUTPUTS'     # data path of optimization results

os.makedirs(results_output_dir, exist_ok=True)

# --- Static Parameters ---
locations = ['Sydney', 'Frankfurt', 'Singapore', 'Lulea', 'SiliconValley', 'DesMoines', 'Omaha', 'NewYork', 'Dallas', 'NorthVirginia']
trace_names = ['Earth', 'Kalos', 'PAI2020', 'Saturn', 'Seren', 'Uranus', 'Venus']

wind_price = [43, 53.3, 200, 43.6, 47, 32, 30, 37, 31, 40]
solar_price = [37.9, 62.6, 63, 39.5, 41, 45, 43, 53, 39, 46]
storage_price = [0.280e6, 0.862e6, 0.470e6, 0.375e6, 0.477e6, 0.477e6, 0.477e6, 0.477e6, 0.477e6, 0.477e6]

# --- Initialize Arrays ---
T = 8760
Electricity = np.zeros((T, 1))
Electricity_0 = np.zeros((T, 1))
CF = np.zeros((T, 1))
Price = np.zeros((T, 1))
Produce_wind = np.zeros((T, 1))
Produce_solar = np.zeros((T, 1))

flag = 0

# --- Begin Optimization Loop ---
for loc_idx, location in enumerate(locations):
    # Load hourly price data
    price_file = os.path.join(grid_data_dir, f"{location}.csv")
    price_df = pd.read_csv(price_file, header=None, names=["Price_1", "Price_2"])
    Price[:, 0] = price_df["Price_2"].values[:8760]

    # Load capacity factor (CF) data
    cf_file = os.path.join(grid_data_dir, f"{location}_CF.csv")
    cf_df = pd.read_csv(cf_file, header=None, names=["CF_1", "CF_2"])
    CF[:, 0] = cf_df["CF_2"].values[:8760]

    for trace_name in trace_names:
        # Load DRL (optimized) electricity usage
        trace_file = os.path.join(DRL_data_dir[flag], "monitor.csv")
        df_trace = pd.read_csv(trace_file)
        Electricity[:, 0] = df_trace['Electricity:Facility'].values[:T] / 3600 / 1e6  # Convert to MW

        # Load baseline (non-optimized) electricity
        base_file = os.path.join(BAS_data_dir[flag],"monitor.csv")
        df_base = pd.read_csv(base_file)
        Electricity_0[:, 0] = df_base['Electricity:Facility'].values[:T] / 3600 / 1e6

        # Load baseline renewable generation data from EnergyPlus model outputs()
        renewable_file = os.path.join(BAS_data_dir[flag],"monitor.csv")
        df_renew = pd.read_csv(renewable_file)

        Produce_wind[:, 0] = df_renew['WindTurbine:ElectricityProduced'].values[:T] / 3600 / 1e6
        Produce_solar[:, 0] = df_renew['Photovoltaic:ElectricityProduced'].values[:T] / 3600 / 1e6

        # --- Optimization Model ---
        start_time = time.time()

        model = ConcreteModel()
        model.T = RangeSet(1, T)

        # Decision Variables
        model.charge = Var(model.T, within=NonNegativeReals)
        model.discharge = Var(model.T, within=NonNegativeReals)
        model.soc = Var(model.T, within=NonNegativeReals)
        model.grid_power = Var(model.T, within=NonNegativeReals)
        model.wind = Var(within=NonNegativeReals)
        model.solar = Var(within=NonNegativeReals)
        model.battery = Var(within=NonNegativeReals)

        # Objective Function
        model.cost = Objective(
            expr=model.battery * storage_price[loc_idx] / 15 +
                 sum(model.grid_power[t] * Price[t - 1, 0] +
                     wind_price[loc_idx] * model.wind * Produce_wind[t - 1, 0] +
                     solar_price[loc_idx] * model.solar * Produce_solar[t - 1, 0]
                     for t in model.T),
            sense=minimize
        )

        # Constraints
        model.balance = ConstraintList()
        model.soc_update = ConstraintList()

        model.balance.add(
            sum(model.grid_power[t] * CF[t - 1, 0] - 0.5 * Electricity_0[t - 1, 0] * CF[t - 1, 0] for t in model.T) <= 0
        )

        for t in model.T:
            demand = Electricity[t - 1, 0]
            wind = Produce_wind[t - 1, 0]
            solar = Produce_solar[t - 1, 0]

            model.balance.add(demand <= model.wind * wind + model.solar * solar + model.grid_power[t] +
                              model.discharge[t] - model.charge[t])
            model.balance.add(model.grid_power[t] - demand + model.discharge[t] - model.charge[t] <= 0)

            model.soc_update.add(model.soc[t] >= model.battery * 0.2)
            model.soc_update.add(model.soc[t] <= model.battery * 0.95)
            model.soc_update.add(model.discharge[t] <= model.battery * 0.25)
            model.soc_update.add(model.charge[t] <= model.battery * 0.25)

            if t == 1:
                model.soc_update.add(model.soc[t] == model.battery * 0.2 + model.charge[t] - model.discharge[t])
            else:
                model.soc_update.add(model.soc[t] == model.soc[t - 1] + model.charge[t] - model.discharge[t])

        # Solve
        solver = SolverFactory('glpk')
        solver.solve(model)

        optimal_value = value(model.cost)
        print(f"[{flag+1}] {location}-{trace_name} Optimal Cost: {optimal_value:.2f}")

        # Save Results
        results = pd.DataFrame({
            'charge': [model.charge[t]() for t in model.T],
            'discharge': [model.discharge[t]() for t in model.T],
            'grid_power': [model.grid_power[t]() for t in model.T],
            'soc': [model.soc[t]() for t in model.T],
            'produce_wind': [Produce_wind[t - 1, 0] * model.wind() for t in model.T],
            'produce_solar': [Produce_solar[t - 1, 0] * model.solar() for t in model.T],
            'electricity': [Electricity_0[t - 1, 0] for t in model.T],
            'CF': [CF[t - 1, 0] for t in model.T],
            'Price': [Price[t - 1, 0] for t in model.T],
            'wind_factor': [model.wind() * 3.2 for _ in model.T],
            'solar_factor': [model.solar() * 3.2 for _ in model.T],
            'battery_capacity': [model.battery() for _ in model.T],
            'optimal_cost': [optimal_value for _ in model.T]
        })

        output_filename = f"Optimal_{loc_idx}_{trace_name}.csv"
        results.to_csv(os.path.join(optimization_output_dir, output_filename), index=False)

        flag += 1
        elapsed_time = time.time() - start_time
        print(f"Finished run {flag} in {elapsed_time:.2f} seconds.\n")
