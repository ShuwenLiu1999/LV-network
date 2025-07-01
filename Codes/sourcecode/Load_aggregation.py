import sys
from pathlib import Path
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pandapower as pp, pandapower.networks as pn

def mc_assign_households(net, hhp_percentage=0.2, seed: int = None):
    # Now we have the pandapower network, let's add the number of households per load
    net.load["n_household"] = (net.load["p_mw"] * 1000 / 3.25).round()  # assume 1 household per load
    N_hh = net.load["n_household"].sum()
    # Prepare household DataFrame
    load_df = net.load.copy()
    load_df = load_df[load_df["n_household"].notna()].copy()
    load_df["n_household"] = load_df["n_household"].astype(int)

    # Expand households
    households = load_df.loc[load_df.index.repeat(load_df["n_household"])].copy()
    households.reset_index(drop=True, inplace=True)
    households["household_id"] = [f"HH_{i}" for i in range(len(households))]

    # Total households and target number of HHPs (e.g. 20% penetration)
    N_total = len(households)
    N_HHP = int(hhp_percentage * N_total)

    # Randomly select indices to assign HHP
    if seed is not None:
        np.random.seed(seed)  # for reproducibility

    # Randomly assign HHPs
    households["tech"] = "HP"  # default
    if N_HHP == 0:
        pass  # all HPs already
    elif N_HHP == N_total:
        households["tech"] = "HHP"  # all HHPs
    else:
        hhp_indices = np.random.choice(households.index, size=N_HHP, replace=False)
        households.loc[hhp_indices, "tech"] = "HHP"

    # Final tidy view
    households = households[["household_id", "name", "bus", "tech"]]
    hhp_counts_by_bus = (
        households[households["tech"] == "HHP"]
        .groupby("name")
        .size()
        .rename("n_HHP")
        .reset_index()
    )
    if hhp_counts_by_bus.empty:
        # no HHP at all → zero for every load name
        hhp_counts_by_bus = pd.DataFrame({
            "name": net.load["name"].unique(),
            "n_HHP": 0
        })
    df_load_info = net.load.copy()
    print(hhp_counts_by_bus)
    df_load_info = pd.merge(df_load_info, hhp_counts_by_bus, on="name", how="left")
    df_load_info["n_HHP"] = df_load_info["n_HHP"].fillna(0).astype(int)
    return df_load_info


def load_aggregation_by_nodes(df_load_info, df_HHP_dir, df_HP_dir,
                              baseload_dir="E:\GitHubProjects\LV network\Data_for_CIGRE_Network\Baseload_from_SERL.csv",
                              out_dir: str = None):
    import pandas as pd
    import os

    # Load HHP and HP profiles
    df_HHP = pd.read_csv(df_HHP_dir, index_col=0, parse_dates=True)
    df_HP = pd.read_csv(df_HP_dir, index_col=0, parse_dates=True)
    sim_start = df_HHP.index[0].normalize()
    sim_end = df_HHP.index[-1].normalize()

    # -------------------------------------------------------------------------
    # 1) Load and prepare the baseload profile
    # -------------------------------------------------------------------------
    df_base = pd.read_csv(
        baseload_dir,
        names=["x", "Pe_W"],  # assumes headerless file
        header=0,
    )
    df_base["td"] = pd.to_timedelta(df_base["x"], unit="h")
    df_base = df_base.set_index("td")[["Pe_W"]]

    # Replicate for each simulation day
    all_days = pd.date_range(sim_start, sim_end, freq="D")
    profiles = []
    for day in all_days:
        prof = df_base.copy()
        prof.index = day + prof.index
        profiles.append(prof)

    df_base = pd.concat(profiles)
    df_base.index = pd.to_datetime(df_base.index)
    df_HHP.index = pd.to_datetime(df_HHP.index)
    df_HP.index = pd.to_datetime(df_HP.index)

    # -------------------------------------------------------------------------
    # 2) Strict check for index match
    # -------------------------------------------------------------------------
    # Sanity check: no node should have more HHPs than total households
    if any(df_load_info["n_HHP"] > df_load_info["n_household"]):
        raise ValueError("HHP count exceeds total household count at one or more nodes.")

    if not (df_base.index.equals(df_HHP.index) and df_base.index.equals(df_HP.index)):
        raise ValueError("Timestamp indices of baseload, HHP, and HP do not match exactly.")

    # -------------------------------------------------------------------------
    # 3) Aggregate
    # -------------------------------------------------------------------------
    df_load_by_nodes = df_base.rename(columns={"Pe_W": "Baseload(W)"})
    df_load_by_nodes["HHP(W)"] = df_HHP["Pe_hp"].values
    df_load_by_nodes["HP(W)"] = df_HP["Pe_hp"].values

    for load, n_hh, n_HHP in df_load_info[["name", "n_household", "n_HHP"]].values:
        df_load_by_nodes[load] = (
            df_load_by_nodes["Baseload(W)"] * n_hh +
            df_load_by_nodes["HHP(W)"] * n_HHP +
            df_load_by_nodes["HP(W)"] * (n_hh - n_HHP)
        )

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        df_load_by_nodes.to_csv(os.path.join(out_dir, "df_load_by_nodes.csv"))

    return df_load_by_nodes
