from pyomo.environ import *
import pandas as pd
import numpy as np
import os
import time

# --- Import data based on installation path ---
data_trace_dir = 'PATH_TO_TRACE_DATA'          # data path of DRL-based DC model outputs (monitor.csv)
grid_data_dir = 'PATH_TO_GRID_DATA'            # data path of grid factors
optimization_output_dir = 'PATH_TO_OPTIMIZATION_OUTPUTS'     # data path of optimization results

# --- Define Parameters ---
locations = ['Sydney', 'Frankfurt', 'Singapore', 'Lulea', 'SiliconValley', 'DesMoines', 'Omaha', 'NewYork', 'Dallas', 'NorthVirginia']
trace_names = ['Earth', 'Kalos', 'PAI2020', 'Saturn', 'Seren', 'Uranus', 'Venus']
loc_num = len(locations)
trace_num = len(trace_names)

# --- Preallocate Shared Arrays ---
Electricity = np.zeros((8760, 1))
CF = np.zeros((8760, 1))
Price = np.zeros((8760, 1))
Produce = np.zeros((8760, 1))
flag = 0

# --- Main Loop ---
for loc_idx, location in enumerate(locations):
    # Load hourly price data
    price_file = os.path.join(grid_data_dir, f"{location}.csv")
    price_df = pd.read_csv(price_file, header=None, names=["Price_1", "Price_2"])
    Price[:, 0] = price_df["Price_2"].values[:8760]

    # Load capacity factor (CF) data
    cf_file = os.path.join(grid_data_dir, f"{location}_CF.csv")
    cf_df = pd.read_csv(cf_file, header=None, names=["CF_1", "CF_2"])
    CF[:, 0] = cf_df["CF_2"].values[:8760]

    # Loop through all traces
    for trace_idx, trace in enumerate(trace_names):
        # Load trace electricity usage data
        trace_path = os.path.join(data_trace_dir, "monitor.csv")
        trace_df = pd.read_csv(trace_path)
        Electricity[:min(len(trace_df), 8760), 0] = trace_df['Electricity:Facility'].values[:8760]

        # Load renewable energy production data
        adopt_path = os.path.join(data_trace_dir, "monitor.csv")
        adopt_df = pd.read_csv(adopt_path)
        Produce[:min(len(adopt_df), 8760), 0] = (
            adopt_df['WindTurbine:ElectricityProduced'].values[:8760] +
            adopt_df['Photovoltaic:ElectricityProduced'].values[:8760]
        )

        start_time = time.time()
        T = 8760
        battery_capacity = 15*0.5
        max_charge = battery_capacity / 4
        max_discharge = battery_capacity / 4
        soc_min = 0.2 * battery_capacity
        soc_max = 0.95 * battery_capacity

        # Define Pyomo model
        model = ConcreteModel()
        model.T = RangeSet(1, T)
        model.charge = Var(model.T, bounds=(0, max_charge))
        model.discharge = Var(model.T, bounds=(0, max_discharge))
        model.soc = Var(model.T, bounds=(soc_min, soc_max))
        model.grid_power = Var(model.T, within=NonNegativeReals)

        # Objective: Minimize grid electricity cost
        model.cost = Objective(
            expr=sum(model.grid_power[t] * Price[t - 1, 0] for t in model.T),
            sense=minimize
        )

        # Constraints
        model.balance = ConstraintList()
        model.soc_update = ConstraintList()

        for t in model.T:
            demand = Electricity[t - 1, 0] / 3600 / 1e6
            renewable = Produce[t - 1, 0] / 3600 / 1e6

            # Power balance
            model.balance.add(demand <= renewable + model.grid_power[t] + model.discharge[t] - model.charge[t])
            model.balance.add(model.grid_power[t] - demand + model.discharge[t] - model.charge[t] <= 0)

            # Battery state-of-charge
            if t == 1:
                model.soc_update.add(model.soc[t] == soc_min + model.charge[t] - model.discharge[t])
            else:
                model.soc_update.add(model.soc[t] == model.soc[t - 1] + model.charge[t] - model.discharge[t])

        # Solve the model
        solver = SolverFactory('glpk')
        solver.solve(model)

        # Save results
        results = pd.DataFrame({
            'charge': [model.charge[t]() for t in model.T],
            'discharge': [model.discharge[t]() for t in model.T],
            'grid_power': [model.grid_power[t]() for t in model.T],
            'produce': [Produce[t - 1, 0] / 3600 / 1e6 for t in model.T],
            'electricity': [Electricity[t - 1, 0] / 3600 / 1e6 for t in model.T],
            'price': [Price[t - 1, 0] for t in model.T]
        })

        output_file = os.path.join(optimization_output_dir, 'Grid Results', f"Save_{loc_idx}_{trace_idx}.csv")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        results.to_csv(output_file, index=False)

        flag += 1
        print(f"Completed optimization run {flag}")
        print(f"Optimization completed in {time.time() - start_time:.2f} seconds.
