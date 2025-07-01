import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# This function takes in the parameters for a hybrid heat pump (HHP) operation
# optimization problem and solve for the solution
def optimize_hhp_operation(
    R1: float,
    C1: float,
    g: float,
    tariff: pd.DataFrame,
    Tout: np.ndarray,
    S: np.ndarray,
    dt: float,
    T0: float,
    T_setpoint: np.ndarray,
    tol: float | np.ndarray,
    COP: float,
    etaB: float,
    Qhp_max: float,
    Qbo_max: float,
) -> pd.DataFrame:
    """
    Hybrid‐HP dispatch with comfort constraints.
    Returns: DataFrame with ['Q_hp','Q_bo','Tin','T_set','T_low','T_high']
    """

    T = len(tariff)
    beta = dt * R1 / (C1 * R1 + dt)

    m = gp.Model("hhp_dispatch")
    m.Params.OutputFlag = 0

    # decision vars
    Q_hp = m.addVars(T, lb=0, ub=Qhp_max, name="Q_hp")
    Q_bo = m.addVars(T, lb=0, ub=Qbo_max, name="Q_bo")
    Tin  = m.addVars(T, lb=-GRB.INFINITY, name="Tin")

    # objective
    m.setObjective(
        gp.quicksum(
            (Q_hp[t]/COP)*tariff['elec_price'].iat[t] +
            (Q_bo[t]/etaB)*tariff['gas_price'].iat[t]
            for t in range(T)
        ),
        GRB.MINIMIZE
    )

    # initial condition
    m.addConstr(Tin[0] == T0, name="init_temp")

    # dynamics
    for t in range(1, T):
        m.addConstr(
            Tin[t] == Tin[t-1]
                      + beta * (
                          (Tout[t-1] - Tin[t-1]) / R1
                        + Q_hp[t-1]
                        + Q_bo[t-1]
                        + g * S[t-1]
                      ),
            name=f"dyn_{t}"
        )

    # build comfort bands
    tol_arr = tol if np.ndim(tol) > 0 else np.full(T, tol)
    T_low = np.where(T_setpoint >= 19.0, T_setpoint - tol_arr, 15.0)
    T_high = np.where(T_setpoint >= 19.0, T_setpoint + tol_arr, np.nan)  # NaN if no upper bound

    for t in range(T):
        m.addConstr(Tin[t] >= T_low[t], name=f"tmin_{t}")
        if not np.isnan(T_high[t]):
            m.addConstr(Tin[t] <= T_high[t], name=f"tmax_{t}")

    m.optimize()

    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Gurobi did not find an optimal solution. Status: {m.Status}")

    Qhp_res = np.array([Q_hp[t].X for t in range(T)])
    Qbo_res = np.array([Q_bo[t].X for t in range(T)])
    Tin_res = np.array([Tin[t].X for t in range(T)])

    return pd.DataFrame({
        'Q_hp': Qhp_res,
        'Q_bo': Qbo_res,
        'Tin': Tin_res,
        'T_set': T_setpoint,
        'T_low': T_low,
        'T_high': T_high
    }, index=tariff.index)

# Function to generate a daily tariff DataFrame with electricity and gas prices
def build_tariff(start_date,n_days,step="30min",type:str="cosy"):
    if type == "flat":
        return flat_tariff(start_date,n_days=n_days,step=step)
    else:
        tariff_list = []
        for i in range(n_days):
            day = start_date + pd.Timedelta(days=i)
            tariff_list.append(daily_tariff(day, step))
        return pd.concat(tariff_list)

def daily_tariff(day, step="30min"):
    times = pd.date_range(day, day + pd.Timedelta("1D") - pd.Timedelta(step), freq=step)

    elec = pd.Series(15.0, index=times) # default electricity price 15.0
    gas  = pd.Series(5.0, index=times) # default gas price 5.0

    cosy_periods = [
        (day + pd.Timedelta("02:00:00"), day + pd.Timedelta("05:00:00")),
        (day + pd.Timedelta("13:00:00"), day + pd.Timedelta("17:00:00")),
        (day + pd.Timedelta("21:00:00"), day + pd.Timedelta("24:00:00")),
    ]
    high_periods = [(day + pd.Timedelta("17:00:00"), day + pd.Timedelta("20:00:00"))]

    for start, end in cosy_periods:
        elec[start:end] = 5.0
    for start, end in high_periods:
        elec[start:end] = 30.0

    return pd.DataFrame({"elec_price": elec, "gas_price": gas})

def flat_tariff(start_date,n_days,step="30min"):
    """
    Generate a flat tariff DataFrame with constant electricity and gas prices.
    """
    times = pd.date_range(start_date, start_date + pd.Timedelta(days=n_days) - pd.Timedelta(step), freq=step)
    elec = pd.Series(15.0, index=times)  # flat electricity price
    gas = pd.Series(5.0, index=times)    # flat gas price

    return pd.DataFrame({"elec_price": elec, "gas_price": gas})
# Function to plot the results of the hybrid heat pump operation
def plot_hhp_results_components(
        Tin: np.ndarray,
        Q_hp: np.ndarray,
        Q_bo: np.ndarray,
        T_set: np.ndarray,
        T_low: np.ndarray,
        T_high: np.ndarray,
        index: pd.DatetimeIndex,
        tariff: pd.DataFrame,
        df: pd.DataFrame,
        include_measured: bool = True,
        title_prefix: str = "Hybrid Heat Pump Dispatch",
        save_dir: str = "figures",
        filename: str = "hhp_dispatch.png"
):
    """
    Modular plotting function using raw result arrays instead of DataFrame.

    New params:
      save_dir : str, directory to save the figure
      filename : str, file name (e.g. 'hhp_dispatch.png')

    Other parameters unchanged...
    """
    fig, axs = plt.subplots(6, 1, figsize=(12, 14), sharex=True)

    # 1. Tariff prices
    axs[0].plot(tariff.index, tariff["elec_price"], label="Electricity (p/kWh)", color='blue')
    axs[0].plot(tariff.index, tariff["gas_price"], label="Gas (p/kWh)", color='orange')
    axs[0].set_ylabel("Tariff (p/kWh)")
    axs[0].legend()
    axs[0].grid(True)

    # 2. Thermal energy delivered
    axs[1].plot(index, Q_hp / 1000, label="HP Heat (kW)", color='green')
    axs[1].plot(index, Q_bo / 1000, label="Boiler Heat (kW)", color='red')
    axs[1].set_ylabel("Heat Output (kW)")
    axs[1].legend()
    axs[1].grid(True)

    # 3. Indoor air temperature + comfort band
    axs[2].plot(index, T_set, label="Setpoint", color="blue")
    axs[2].fill_between(index, T_low, T_high, color="lightblue", alpha=0.3, label="Comfort Band")
    axs[2].plot(index, Tin, label="Modelled Indoor T", color='purple')
    axs[2].set_ylabel("Indoor Temp (°C)")
    axs[2].legend()
    axs[2].grid(True)

    # 4. Outdoor temperature and GHI
    axs[3].plot(df.index, df["External_Air_Temperature"], label="Outdoor Temp (°C)", color='black')
    axs[3].plot(df.index, df["GHI"] / 100, label="GHI (x0.01 W/m²)", color='gold', linestyle=':')
    axs[3].set_ylabel("Outdoor / GHI")
    axs[3].legend()
    axs[3].grid(True)

    # 5. Total thermal power
    axs[4].plot(index, (Q_hp + Q_bo) / 1000, label="Total Heat (kW)", color='teal')
    axs[4].set_ylabel("Total Heat (kW)")
    axs[4].legend()
    axs[4].grid(True)

    # 6. Measured indoor temperature
    if include_measured and "Internal_Air_Temperature" in df.columns:
        axs[5].plot(df.index, df["Internal_Air_Temperature"], label="Measured Indoor T", color='purple')
        axs[5].set_ylabel("Measured T (°C)")
        axs[5].legend()
        axs[5].grid(True)

    plt.xlabel("Time")
    plt.suptitle(title_prefix, y=1.02)
    plt.tight_layout()

    # --- SAVE FIGURE ---
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, filename)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved to {out_path}")

if __name__ == "__main__":
    start_date = pd.to_datetime("2022-01-03")
    n_days = 25  # change to however many days you want
    step = "30min"
    # --- Build multi-day tariff ---
    tariff_list = [daily_tariff(start_date + pd.Timedelta(days=i)) for i in range(n_days)]
    tariff = pd.concat(tariff_list)

    # load data

    df = pd.read_csv(r"E:\Python projects\RC model\Test data\Property_ID=EOH0031_2022_01.csv",
                     parse_dates=["Timestamp"], index_col="Timestamp")
    df = df.loc[tariff.index[0]:tariff.index[-1]]  # align with tariff index
    dt = df.index.to_series().diff().median().total_seconds()

    Tout = df["External_Air_Temperature"].values
    S = df["GHI"].values

    # define a daily setpoint schedule plus ±0.5°C tolerance
    # e.g. 20°C from midnight–06:00, 22°C from 06:00–22:00, back to 20°C
    hours = tariff.index.hour
    T_set = np.where((hours >= 6) & (hours < 10) | (hours >= 17) & (hours < 21), 25.0, 15.0)
    print(len(T_set))
    tol = 1

    # other parameters
    R1, C1, g = 1 / 200, 3e7, 10.0
    T0 = 20
    COP = 3.5
    etaB = 0.9
    Qhp_max = 4e3
    Qbo_max = 24e3

    results = optimize_hhp_operation(
        R1, C1, g, tariff, Tout, S, dt, T0,
        T_setpoint=T_set, tol=tol,
        COP=COP, etaB=etaB,
        Qhp_max=Qhp_max, Qbo_max=Qbo_max
    )

    print(results)

    plot_hhp_results_components(
        Tin=results["Tin"].values,
        Q_hp=results["Q_hp"].values,
        Q_bo=results["Q_bo"].values,
        T_set=results["T_set"].values,
        T_low=results["T_low"].values,
        T_high=results["T_high"].values,
        index=results.index,
        tariff=tariff,
        df=df,
        title_prefix="Cooling Scenario"
    )
