# This is the function for plotting the results from power flow, according to 3 metrics:
# a. Bus voltage (pu)
# b. Line real power (MW) from the "from" bus
# c. Transformer sizing, which is 4/3 of the maximum transformer load (MVA)

import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandapower.plotting as plot

def plot_pf_with_transformer_capacity(net,
                                      bus_cmap: str = "viridis",
                                      power_cmap: str = "coolwarm",
                                      max_line_width: float = 5.0,
                                      save_dir: str = "figures",
                                      filename: str = "pf_plot.png"):
    """
    Plot PF results on 'net', showing:
     - buses colored by vm_pu
     - lines colored & width-scaled by real power (p_from_mw)
     - annotation for min transformer size = 4/3 * max transformer load (MVA)

    Assumes you've already called pp.runpp(net).

    Parameters
    ----------
    save_dir : str
        Directory where the plot will be saved (created if it doesn't exist).
    filename : str
        Name of the image file (e.g. 'pf_plot.png').
    """
    # --- BUSES ---
    vm = net.res_bus.vm_pu
    norm_bus = mpl.colors.Normalize(vmin=vm.min(), vmax=vm.max())
    cmap_bus = mpl.colormaps[bus_cmap]
    bus_colors = [cmap_bus(norm_bus(v)) for v in vm]

    # --- LINES ---
    p = net.res_line.p_from_mw
    norm_line = mpl.colors.Normalize(vmin=p.min(), vmax=p.max())
    cmap_line = mpl.colormaps[power_cmap]
    line_colors = [cmap_line(norm_line(val)) for val in p]
    line_widths = (abs(p) / abs(p).max()) * max_line_width + 0.1

    # --- TRANSFORMER SIZING ---
    loads_mva = []
    for idx in net.trafo.index:
        sn = net.trafo.at[idx, "sn_mva"]
        loading = net.res_trafo.at[idx, "loading_percent"]
        loads_mva.append(sn * loading / 100.0)
    max_load = max(loads_mva)
    min_size = (4 / 3) * max_load

    # --- DRAW NETWORK ---
    ax = plot.simple_plot(
        net,
        show_plot=False,
        bus_size=1,
        bus_color=bus_colors,
        line_color=line_colors,
        line_width=line_widths
    )

    # --- COLORBARS ---
    sm_bus = mpl.cm.ScalarMappable(norm=norm_bus, cmap=cmap_bus)
    sm_bus.set_array([])
    cbar_bus = plt.colorbar(sm_bus, ax=ax, fraction=0.046, pad=0.06)
    cbar_bus.set_label("Bus Voltage (pu)")

    sm_line = mpl.cm.ScalarMappable(norm=norm_line, cmap=cmap_line)
    sm_line.set_array([])
    cbar_line = plt.colorbar(sm_line, ax=ax, fraction=0.046, pad=0.14)
    cbar_line.set_label("Real Power (MW) from ‘from_bus’")

    # --- ANNOTATION ---
    ax.text(0.5, 1.05,
            f"Min\nTransformer Size ≈ {min_size:.2f} MVA\n(25% headroom)",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.title("CIGRE LV — Voltages, Flows & Trafo-Sizing")
    plt.axis("off")

    # --- SAVE PLOT ---
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    out_file = save_path / filename
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {out_file}")

# Example usage:
if __name__ == "__main__":
    import pandapower as pp
    import pandapower.networks as pn

    net = pn.create_cigre_network_lv()
    pp.runpp(net, numba=False)
    plot_pf_with_transformer_capacity(net,
                                      save_dir="figures",
                                      filename="cigre_lv_pf1.png")
