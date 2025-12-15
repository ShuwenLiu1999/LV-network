import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# This function takes in the parameters for a hybrid heat pump (HHP) operation
# optimization problem and solve for the solution
# def optimize_hhp_operation(
#     R1: float,
#     C1: float,
#     g: float,
#     tariff: pd.DataFrame,
#     Tout: np.ndarray,
#     S: np.ndarray,
#     dt: float,
#     T0: float,
#     T_setpoint: np.ndarray,
#     tol: float | np.ndarray,
#     COP: float,
#     etaB: float,
#     Qhp_max: float,
#     Qbo_max: float,
# ) -> pd.DataFrame:
#     """
#     Hybrid‐HP dispatch with comfort constraints.
#     Returns: DataFrame with ['Q_hp','Q_bo','Tin','T_set','T_low','T_high']
#     """
#
#     T = len(tariff)
#     beta = dt * R1 / (C1 * R1 + dt)
#
#     m = gp.Model("hhp_dispatch")
#     m.Params.OutputFlag = 0
#
#     # decision vars
#     Q_hp = m.addVars(T, lb=0, ub=Qhp_max, name="Q_hp")
#     Q_bo = m.addVars(T, lb=0, ub=Qbo_max, name="Q_bo")
#     Tin  = m.addVars(T, lb=-GRB.INFINITY, name="Tin")
#
#     # objective
#     m.setObjective(
#         gp.quicksum(
#             (Q_hp[t]/COP)*tariff['elec_price'].iat[t] +
#             (Q_bo[t]/etaB)*tariff['gas_price'].iat[t]
#             for t in range(T)
#         ),
#         GRB.MINIMIZE
#     )
#
#     # initial condition
#     m.addConstr(Tin[0] == T0, name="init_temp")
#
#     # dynamics
#     for t in range(1, T):
#         m.addConstr(
#             Tin[t] == Tin[t-1]
#                       + beta * (
#                           (Tout[t-1] - Tin[t-1]) / R1
#                         + Q_hp[t-1]
#                         + Q_bo[t-1]
#                         + g * S[t-1]
#                       ),
#             name=f"dyn_{t}"
#         )
#
#     # build comfort bands
#     tol_arr = tol if np.ndim(tol) > 0 else np.full(T, tol)
#     T_low = np.where(T_setpoint >= 19.0, T_setpoint - tol_arr, 15.0)
#     T_high = np.where(T_setpoint >= 19.0, T_setpoint + tol_arr, np.nan)  # NaN if no upper bound
#
#     for t in range(T):
#         m.addConstr(Tin[t] >= T_low[t], name=f"tmin_{t}")
#         if not np.isnan(T_high[t]):
#             m.addConstr(Tin[t] <= T_high[t], name=f"tmax_{t}")
#
#     m.optimize()
#
#     if m.Status != GRB.OPTIMAL:
#         raise RuntimeError(f"Gurobi did not find an optimal solution. Status: {m.Status}")
#
#     Qhp_res = np.array([Q_hp[t].X for t in range(T)])
#     Qbo_res = np.array([Q_bo[t].X for t in range(T)])
#     Tin_res = np.array([Tin[t].X for t in range(T)])
#
#     return pd.DataFrame({
#         'Q_hp': Qhp_res,
#         'Q_bo': Qbo_res,
#         'Tin': Tin_res,
#         'T_set': T_setpoint,
#         'T_low': T_low,
#         'T_high': T_high
#     }, index=tariff.index)

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
    day_ahead: bool = False
) -> pd.DataFrame:
    """
    Hybrid‐HP dispatch with comfort constraints.
    If day_ahead=True, solves one optimization per calendar day using only that day's tariff & weather,
    carrying the final indoor temperature forward as the next day's initial condition.

    Returns: DataFrame with ['Q_hp','Q_bo','Tin','T_set','T_low','T_high'] at the same time index as tariff.
    """

    def _single_dispatch(R1, C1, g,
                         tariff_sub, Tout_sub, S_sub, dt,
                         T0_sub, T_set_sub, tol_sub,
                         COP, etaB, Qhp_max, Qbo_max):
        T = len(tariff_sub)
        # comfort bounds
        tol_arr = tol_sub if np.ndim(tol_sub) > 0 else np.full(T, tol_sub)
        T_low = np.where(T_set_sub >= 19.0, T_set_sub - tol_arr, 15.0)
        T_high = np.where(T_set_sub >= 19.0, T_set_sub + tol_arr, np.nan)

        beta = dt * R1 / (C1 * R1 + dt)
        m = gp.Model()
        m.Params.OutputFlag = 0

        Q_hp = m.addVars(T, lb=0, ub=Qhp_max, name="Q_hp")
        Q_bo = m.addVars(T, lb=0, ub=Qbo_max, name="Q_bo")
        Tin  = m.addVars(T, lb=-GRB.INFINITY, name="Tin")

        m.setObjective(
            gp.quicksum(
                (Q_hp[t]/COP)*tariff_sub['elec_price'].iat[t] +
                (Q_bo[t]/etaB)*tariff_sub['gas_price'].iat[t]
                for t in range(T)
            ),
            GRB.MINIMIZE
        )
        m.addConstr(Tin[0] == T0_sub, name="init_temp")
        for t in range(1, T):
            m.addConstr(
                Tin[t] == Tin[t-1]
                          + beta * (
                              (Tout_sub[t-1] - Tin[t-1]) / R1
                            + Q_hp[t-1]
                            + Q_bo[t-1]
                            + g * S_sub[t-1]
                          ),
                name=f"dyn_{t}"
            )
        for t in range(T):
            m.addConstr(Tin[t] >= T_low[t], name=f"tmin_{t}")
            if not np.isnan(T_high[t]):
                m.addConstr(Tin[t] <= T_high[t], name=f"tmax_{t}")
        m.optimize()
        if m.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Gurobi failed on subproblem. Status {m.Status}")

        Qhp_res = np.array([Q_hp[t].X for t in range(T)])
        Qbo_res = np.array([Q_bo[t].X for t in range(T)])
        Tin_res = np.array([Tin[t].X for t in range(T)])
        return pd.DataFrame({
            'Q_hp': Qhp_res,
            'Q_bo': Qbo_res,
            'Tin' : Tin_res,
            'T_set': T_set_sub,
            'T_low': T_low,
            'T_high': T_high
        }, index=tariff_sub.index)

    # If day-ahead mode, split calendar days
    if day_ahead:
        all_days = tariff.index.normalize().unique()
        results = []
        T0_curr = T0
        # build tolerance array once
        tol_arr_full = tol if np.ndim(tol) > 0 else np.full(len(tariff), tol)
        # pre-calc boolean mask array per day
        norm_idx = tariff.index.normalize()
        for day in all_days:
            day_mask = (norm_idx == day)
            sub_tariff = tariff.loc[day_mask]
            sub_Tout    = Tout[day_mask]
            sub_S       = S[day_mask]
            sub_Tset    = T_setpoint[day_mask]
            sub_tol     = tol_arr_full[day_mask]
            if len(sub_tariff) == 0:
                continue
            df_day = _single_dispatch(
                R1, C1, g,
                sub_tariff, sub_Tout, sub_S, dt,
                T0_curr, sub_Tset, sub_tol,
                COP, etaB, Qhp_max, Qbo_max
            )
            # carry forward
            T0_curr = df_day['Tin'].iat[-1]
            results.append(df_day)
        # concatenate daily results
        return pd.concat(results).loc[tariff.index]  # ensure original order

    # otherwise, solve full horizon at once
    return _single_dispatch(R1, C1, g, tariff, Tout, S, dt, T0, T_setpoint, tol, COP, etaB, Qhp_max, Qbo_max)


def optimize_full_energy_system(
    R1: float,
    C1: float,
    g: float,
    tariff: pd.DataFrame,
    Tout: np.ndarray,
    S: np.ndarray,
    dt: float,
    T0: float,
    setpoint_sequences: list[np.ndarray] | np.ndarray,
    tol: float | np.ndarray,
    COP: float,
    etaB: float,
    Qhp_max: float,
    Qbo_max: float,
    hw_demand: np.ndarray | None = None,
    base_electric: np.ndarray | None = None,
    ev_capacity: float | None = None,
    ev_target: float = 0.0,
    ev_charge_max: float | None = None,
    ev_availability: np.ndarray | None = None,
    eta_ev_charge: float = 0.95,
    day_ahead: bool = False,
) -> dict:
    """
    Solve the full optimization problem including hot water demand, EV charging,
    and different daily temperature set‑point sequences.

    Parameters
    ----------
    R1, C1, g : float
        RC thermal parameters and solar gain factor.
    tariff : pd.DataFrame
        Must contain ``elec_price`` and ``gas_price`` indexed by time.
    Tout, S : np.ndarray
        Outdoor temperature and solar irradiance aligned with ``tariff``.
    dt : float
        Time step in seconds.
    T0 : float
        Initial indoor temperature.
    setpoint_sequences : list[np.ndarray] | np.ndarray
        Either a single array (used directly) or a list of candidate set-point
        sequences. Each entry must match ``len(tariff)``.
    tol : float | np.ndarray
        Comfort tolerance (scalar or array matching the horizon).
    COP : float
        Heat pump coefficient of performance.
    etaB : float
        Boiler efficiency (thermal output / fuel input).
    Qhp_max, Qbo_max : float
        Maximum thermal output (W) for heat pump and boiler respectively.
    hw_demand : np.ndarray, optional
        Thermal hot water demand (W). Defaults to zeros if omitted.
    base_electric : np.ndarray, optional
        Other electric demand (W). Defaults to zeros if omitted.
    ev_capacity : float, optional
        EV battery capacity (kWh). If ``None`` EV charging is disabled.
    ev_target : float, default 0.0
        Minimum EV state of charge (kWh) required at the end of the horizon.
    ev_charge_max : float, optional
        Maximum EV charging power (kW). Defaults to ``ev_capacity`` if not
        provided.
    ev_availability : np.ndarray, optional
        Binary mask (1/0) indicating when the EV is at home and can charge.
        Defaults to always available.
    eta_ev_charge : float, default 0.95
        Charging efficiency.
    day_ahead : bool, default False
        If True, solve one optimization per day carrying indoor temperature and
        EV state of charge to the next day.

    Returns
    -------
    dict
        Dictionary keyed by set-point label (``schedule_0``, ``schedule_1``, …)
        containing the resulting ``DataFrame`` and cost. The entry ``"best"``
        contains the minimum‑cost schedule.
    """

    def _solve_single_schedule(tariff_sub, Tout_sub, S_sub, T_set_sub, tol_sub, T0_sub, soc0):
        T = len(tariff_sub)
        dt_hours = dt / 3600.0
        tol_arr = tol_sub if np.ndim(tol_sub) > 0 else np.full(T, tol_sub)
        T_low = np.where(T_set_sub >= 19.0, T_set_sub - tol_arr, 15.0)
        T_high = np.where(T_set_sub >= 19.0, T_set_sub + tol_arr, np.nan)

        hw_profile = hw_demand_sub if hw_demand_sub is not None else np.zeros(T)
        base_elec_profile = base_electric_sub if base_electric_sub is not None else np.zeros(T)
        ev_mask = ev_availability_sub if ev_availability_sub is not None else np.ones(T)

        m = gp.Model("full_energy_system")
        m.Params.OutputFlag = 0

        # Thermal variables
        Q_hp_space = m.addVars(T, lb=0, ub=Qhp_max, name="Q_hp_space")
        Q_bo_space = m.addVars(T, lb=0, ub=Qbo_max, name="Q_bo_space")
        Q_hp_hw = m.addVars(T, lb=0, ub=Qhp_max, name="Q_hp_hw")
        Q_bo_hw = m.addVars(T, lb=0, ub=Qbo_max, name="Q_bo_hw")
        Tin = m.addVars(T, lb=-GRB.INFINITY, name="Tin")

        # EV variables (kW for power, kWh for energy)
        if ev_capacity is not None and ev_capacity > 0:
            ev_power_cap = ev_charge_max if ev_charge_max is not None else ev_capacity
            P_ev_charge = m.addVars(T, lb=0, ub=ev_power_cap, name="P_ev_charge")
            ev_soc = m.addVars(T, lb=0, ub=ev_capacity, name="ev_soc")
        else:
            P_ev_charge = None
            ev_soc = None

        # Objective: electricity + gas cost
        elec_cost = []
        gas_cost = []
        for t in range(T):
            heat_pump_elec = (Q_hp_space[t] + Q_hp_hw[t]) / COP
            other_elec = base_elec_profile[t]
            ev_elec = P_ev_charge[t] if P_ev_charge is not None else 0.0
            elec_cost.append((heat_pump_elec + other_elec + ev_elec) * tariff_sub["elec_price"].iat[t] * dt_hours)

            gas_cost.append(((Q_bo_space[t] + Q_bo_hw[t]) / etaB) * tariff_sub["gas_price"].iat[t] * dt_hours)

        m.setObjective(gp.quicksum(elec_cost) + gp.quicksum(gas_cost), GRB.MINIMIZE)

        # Indoor temperature dynamics
        m.addConstr(Tin[0] == T0_sub, name="init_temp")
        for t in range(1, T):
            m.addConstr(
                Tin[t] == Tin[t - 1]
                + (dt / C1)
                * (
                    Q_hp_space[t - 1]
                    + Q_bo_space[t - 1]
                    + g * S_sub[t - 1]
                    - (Tin[t - 1] - Tout_sub[t - 1]) / R1
                ),
                name=f"temp_dyn_{t}",
            )

        # Comfort constraints
        for t in range(T):
            m.addConstr(Tin[t] >= T_low[t], name=f"Tin_min_{t}")
            if not np.isnan(T_high[t]):
                m.addConstr(Tin[t] <= T_high[t], name=f"Tin_max_{t}")

            # Total HP/boiler output cannot exceed capacity when combining space + HW
            m.addConstr(Q_hp_space[t] + Q_hp_hw[t] <= Qhp_max, name=f"Qhp_cap_{t}")
            m.addConstr(Q_bo_space[t] + Q_bo_hw[t] <= Qbo_max, name=f"Qbo_cap_{t}")

            # Hot water demand must be met
            m.addConstr(Q_hp_hw[t] + Q_bo_hw[t] >= hw_profile[t], name=f"HW_demand_{t}")

        # EV charging dynamics
        if P_ev_charge is not None:
            m.addConstr(ev_soc[0] == soc0, name="ev_soc0")
            for t in range(T):
                m.addConstr(P_ev_charge[t] <= ev_power_cap * ev_mask[t], name=f"ev_avail_{t}")
                if t > 0:
                    m.addConstr(
                        ev_soc[t]
                        == ev_soc[t - 1]
                        + P_ev_charge[t] * dt_hours * eta_ev_charge,
                        name=f"ev_soc_dyn_{t}",
                    )
            m.addConstr(ev_soc[T - 1] >= ev_target, name="ev_target_end")

        m.optimize()
        if m.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Gurobi failed on full problem. Status {m.Status}")

        # Extract results
        res = pd.DataFrame(index=tariff_sub.index)
        res["Tin"] = [Tin[t].X for t in range(T)]
        res["T_set"] = T_set_sub
        res["T_low"] = T_low
        res["T_high"] = T_high
        res["Q_hp_space"] = [Q_hp_space[t].X for t in range(T)]
        res["Q_bo_space"] = [Q_bo_space[t].X for t in range(T)]
        res["Q_hp_hw"] = [Q_hp_hw[t].X for t in range(T)]
        res["Q_bo_hw"] = [Q_bo_hw[t].X for t in range(T)]
        res["elec_cost"] = [float(elec_cost[t]) for t in range(T)]
        res["gas_cost"] = [float(gas_cost[t]) for t in range(T)]

        if P_ev_charge is not None:
            res["P_ev_charge"] = [P_ev_charge[t].X for t in range(T)]
            res["ev_soc"] = [ev_soc[t].X for t in range(T)]
        else:
            res["P_ev_charge"] = 0.0
            res["ev_soc"] = 0.0

        total_cost = res["elec_cost"].sum() + res["gas_cost"].sum()
        return res, total_cost, res["Tin"].iat[-1], res["ev_soc"].iat[-1] if P_ev_charge is not None else 0.0

    # --- Input validation and defaults ---
    n_steps = len(tariff)
    Tout = np.asarray(Tout)
    S = np.asarray(S)
    if len(Tout) != n_steps or len(S) != n_steps:
        raise ValueError("Tout, S, and tariff must have matching lengths")

    hw_demand_sub = None if hw_demand is None else np.asarray(hw_demand)
    base_electric_sub = None if base_electric is None else np.asarray(base_electric)
    ev_availability_sub = None if ev_availability is None else np.asarray(ev_availability)

    if isinstance(setpoint_sequences, np.ndarray) and setpoint_sequences.ndim == 1:
        schedules = [setpoint_sequences]
    else:
        schedules = list(setpoint_sequences)

    results = {}
    best_cost = np.inf
    best_key = None

    if day_ahead:
        # solve one day at a time for each schedule
        days = tariff.index.normalize().unique()
        tol_full = tol if np.ndim(tol) > 0 else np.full(n_steps, tol)
        norm_idx = tariff.index.normalize()
        for idx, sched in enumerate(schedules):
            T_curr = T0
            ev_soc_curr = 0.0
            day_frames = []
            day_costs = []
            for day in days:
                mask = norm_idx == day
                res, cost, T_curr, ev_soc_curr = _solve_single_schedule(
                    tariff.loc[mask],
                    Tout[mask],
                    S[mask],
                    np.asarray(sched)[mask],
                    tol_full[mask],
                    T_curr,
                    ev_soc_curr,
                )
                day_frames.append(res)
                day_costs.append(cost)
            df_all = pd.concat(day_frames).loc[tariff.index]
            total_cost = float(np.sum(day_costs))
            key = f"schedule_{idx}"
            results[key] = {"results": df_all, "cost": total_cost}
            if total_cost < best_cost:
                best_cost = total_cost
                best_key = key
    else:
        for idx, sched in enumerate(schedules):
            res, total_cost, _, _ = _solve_single_schedule(
                tariff,
                Tout,
                S,
                np.asarray(sched),
                tol,
                T0,
                0.0,
            )
            key = f"schedule_{idx}"
            results[key] = {"results": res, "cost": float(total_cost)}
            if total_cost < best_cost:
                best_cost = float(total_cost)
                best_key = key

    if best_key is not None:
        results["best"] = results[best_key]
    return results


# Function to generate a daily tariff DataFrame with electricity and gas prices
def build_tariff(start_date,n_days,step="30min",type:str="cosy"):
    if type == "flat":
        return flat_tariff(start_date,n_days=n_days,step=step)
    elif type == "cosy":
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
        (day + pd.Timedelta("04:00:00"), day + pd.Timedelta("07:00:00")),
        (day + pd.Timedelta("13:00:00"), day + pd.Timedelta("16:00:00")),
        (day + pd.Timedelta("22:00:00"), day + pd.Timedelta("24:00:00")),
    ]
    high_periods = [(day + pd.Timedelta("16:00:00"), day + pd.Timedelta("19:00:00"))]

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
    fig, axs = plt.subplots(5, 1, figsize=(12, 14), sharex=True)

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

    # 4. Outdoor temperature and GHI (dual axis)
    weather_ax = axs[3]
    ghi_ax = weather_ax.twinx()

    temp_line, = weather_ax.plot(
        df.index,
        df["External_Air_Temperature"],
        label="Outdoor Temp (°C)",
        color="black",
    )
    ghi_line, = ghi_ax.plot(
        df.index,
        df["GHI"],
        label="GHI (W/m²)",
        color="gold",
        linestyle=":",
    )

    weather_ax.set_ylabel("Outdoor Temp (°C)")
    ghi_ax.set_ylabel("GHI (W/m²)")

    lines = [temp_line, ghi_line]
    labels = [line.get_label() for line in lines]
    weather_ax.legend(lines, labels)
    weather_ax.grid(True)

    # 5. Total thermal power
    axs[4].plot(index, (Q_hp + Q_bo) / 1000, label="Total Heat (kW)", color='teal')
    axs[4].set_ylabel("Total Heat (kW)")
    axs[4].legend()
    axs[4].grid(True)

    plt.xlabel("Time")
    plt.suptitle(title_prefix, y=1.02)
    plt.tight_layout()

    # --- SAVE FIGURE ---
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, filename)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved to {out_path}")

if __name__ == "__main__":
    start_date = pd.to_datetime("2022-01-01")
    n_days = 31  # change to however many days you want
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
    T_set = np.where((hours >= 6) & (hours < 10) | (hours >= 17) & (hours < 21), 21.0, 15.0)
    print(len(T_set))
    tol = 1

    # other parameters
    R1, C1, g = 1 / 200, 3e7, 10.0
    T0 = 21
    COP = 3.5
    etaB = 0.9
    Qhp_max = 4e3
    Qbo_max = 24e3

    results = optimize_hhp_operation(
        R1, C1, g, tariff, Tout, S, dt, T0,
        T_setpoint=T_set, tol=tol,
        COP=COP, etaB=etaB,
        Qhp_max=Qhp_max, Qbo_max=Qbo_max,day_ahead= False
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
        title_prefix="Cooling Scenario",
        filename="HHP_global_4kW.png",
    )
