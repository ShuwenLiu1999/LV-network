import sys
from pathlib import Path
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pandapower as pp, pandapower.networks as pn
from Network_Plotting import plot_pf_with_transformer_capacity
import numba
# This is for the RC optimization model
import RC_Optimization as rc

from Residential_CIGRE_LV_network import R_LV_CIGRE

from Load_aggregation import mc_assign_households, load_aggregation_by_nodes

def run_and_save_monte_carlo_simulation(n_samples: int = 100, hhp_percentage: float = 0.2, df_HHP_dir=None, df_HP_dir=None,out_dir: str = r"E:\GitHubProjects\LV network\Codes\Output", percentile: float = 0.95,numba=False):
    if not 0.0 <= hhp_percentage <= 1.0:
        raise ValueError("HHP percentage must be between 0 and 1 inclusive.")
    if hhp_percentage ==1.0 or hhp_percentage == 0.0:
        n_samples = 1  # If HHP percentage is 0 or 100, only one sample is needed
    net = R_LV_CIGRE()  # Reinitialize the network for each sample
    # Prepare storage arrays
    n_lines = len(net.line)
    n_trafos = len(net.trafo)
    n_buses = len(net.bus)

    monte_max_line_loading = np.zeros((n_samples, n_lines))
    monte_max_line_power = np.zeros((n_samples, n_lines))
    monte_max_trafo_loading = np.zeros((n_samples, n_trafos))
    monte_min_bus_voltage = np.ones((n_samples, n_buses))  # Init to 1.0 pu

    # NEW: time‐flags (datetime64) for each extreme
    monte_time_line_loading  = np.empty((n_samples, n_lines), dtype="datetime64[ns]")
    monte_time_line_power    = np.empty((n_samples, n_lines), dtype="datetime64[ns]")
    monte_time_trafo_loading = np.empty((n_samples, n_trafos), dtype="datetime64[ns]")
    monte_time_bus_voltage   = np.empty((n_samples, n_buses), dtype="datetime64[ns]")

    for s in range(n_samples):
        print(f"\n--- MC Sample {s + 1}/{n_samples} ---")
        net = R_LV_CIGRE()  # Reinitialize the network for each sample
        # Step 1: Generate household config & aggregated load
        df_load_info = mc_assign_households(net, hhp_percentage=hhp_percentage)
        # This function assigns households to each node in the pandapower network, and randomly assigns HHPs and HPs to the households based on the given percentage.

        df_load_by_nodes = load_aggregation_by_nodes(df_load_info, df_HHP_dir, df_HP_dir)

        timestamps = df_load_by_nodes.index
        n_steps = len(timestamps)

        # Temporary per-sample arrays
        line_power = np.zeros((n_steps, n_lines))
        line_loading = np.zeros((n_steps, n_lines))
        trafo_loading = np.zeros((n_steps, n_trafos))
        bus_voltage = np.zeros((n_steps, n_buses))

        for i, t in enumerate(timestamps):
            # Update loads in the network
            for name in df_load_by_nodes.columns:
                if name in net.load.name.values:
                    load_idx = net.load.index[net.load["name"] == name][0]
                    p_kw = df_load_by_nodes.loc[t, name] / 1000
                    net.load.at[load_idx, "p_mw"] = p_kw / 1000  # kW → MW
                    net.load.at[load_idx, "q_mvar"] = (p_kw/1000)*0.329  # assuming 0.95 power factor
            # Run PF
            try:
                pp.runpp(net, numba=numba)
            except:
                print(f"Power flow failed at {t} in sample {s}")
                continue

            # Store results
            line_power[i, :] = net.res_line.p_from_mw.values
            line_loading[i, :] = net.res_line.loading_percent.values
            trafo_loading[i, :] = net.res_trafo.loading_percent.values
            bus_voltage[i, :] = net.res_bus.vm_pu.values

        # 1) find per‐component argmax/argmin
        for j in range(n_lines):
            idx1 = np.argmax(line_loading[:, j])
            idx2 = np.argmax(np.abs(line_power[:, j]))
            monte_time_line_loading[s, j] = timestamps[idx1]
            monte_time_line_power[s, j]   = timestamps[idx2]
        for k in range(n_trafos):
            idxT = np.argmax(trafo_loading[:, k])
            monte_time_trafo_loading[s, k] = timestamps[idxT]
        for b in range(n_buses):
            idxV = np.argmin(bus_voltage[:, b])
            monte_time_bus_voltage[s, b]   = timestamps[idxV]

        # store the extreme values
        monte_max_line_loading[s, :]  = line_loading.max(axis=0)
        monte_max_line_power[s, :]    = np.abs(line_power).max(axis=0)
        monte_max_trafo_loading[s, :] = trafo_loading.max(axis=0)
        monte_min_bus_voltage[s, :]   = bus_voltage.min(axis=0)
#-------------------------------After all samples, save the results----------------------------------------------------------------------------
    # Build column names with categories
    line_cols_loading = [f"line_loading_{i}" for i in range(monte_max_line_loading.shape[1])]
    line_cols_power = [f"line_power_{i}" for i in range(monte_max_line_power.shape[1])]
    trafo_cols = [f"trafo_loading_{i}" for i in range(monte_max_trafo_loading.shape[1])]
    bus_cols = [f"bus_voltage_{i}" for i in range(monte_min_bus_voltage.shape[1])]
    # Convert arrays to DataFrames
    df_loading = pd.DataFrame(monte_max_line_loading, columns=line_cols_loading)
    df_power = pd.DataFrame(monte_max_line_power, columns=line_cols_power)
    df_trafo = pd.DataFrame(monte_max_trafo_loading, columns=trafo_cols)
    df_voltage = pd.DataFrame(monte_min_bus_voltage, columns=bus_cols)
    # Concatenate horizontally
    df_all = pd.concat([df_loading, df_power, df_trafo, df_voltage], axis=1)
    # NEW: build DataFrames for timestamps
    df_t_line_loading = pd.DataFrame(
        monte_time_line_loading,
        columns=[f"line_loading_time_{i}" for i in range(n_lines)]
    )
    df_t_line_power = pd.DataFrame(
        monte_time_line_power,
        columns=[f"line_power_time_{i}" for i in range(n_lines)]
    )
    df_t_trafo_loading = pd.DataFrame(
        monte_time_trafo_loading,
        columns=[f"trafo_loading_time_{i}" for i in range(n_trafos)]
    )
    df_t_bus_voltage = pd.DataFrame(
        monte_time_bus_voltage,
        columns=[f"bus_voltage_time_{i}" for i in range(n_buses)]
    )

    # concat the time‐flags too
    df_all = pd.concat([
        df_all,
        df_t_line_loading,
        df_t_line_power,
        df_t_trafo_loading,
        df_t_bus_voltage
    ], axis=1)
    # Save to CSV with formatted HHP percentage in filename
    df_all.index.name = "monte_carlo_iter"
    perc_str = f"{int(hhp_percentage * 100):02d}p"
    filename = f"{out_dir}/montecarlo_results_HHP_{perc_str}_{n_samples}samples.csv"
    df_all.to_csv(filename, index=False)
    print(f"Saved Monte Carlo results to: {filename}")
    # --------------------return percentiles of the results--------------------------
    line_perc = np.percentile(monte_max_line_loading, percentile, axis=0)
    line_p_perc = np.percentile(monte_max_line_power, percentile, axis=0)
    trafo_perc = np.percentile(monte_max_trafo_loading, percentile, axis=0)
    bus_vmin_perc = np.percentile(monte_min_bus_voltage, 100 - percentile, axis=0)
    return line_perc, line_p_perc, trafo_perc, bus_vmin_perc