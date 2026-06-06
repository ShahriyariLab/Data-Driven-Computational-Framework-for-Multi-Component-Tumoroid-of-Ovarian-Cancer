#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares


plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "cm",
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.linestyle": "-",
    "grid.linewidth": 0.35,
    "grid.alpha": 0.35,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "lines.linewidth": 1.6,
    "legend.frameon": False,
    "legend.fontsize": 11,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


SUBSTEPS_FIT = 100
SUBSTEPS_PLOT = 100

t_data = np.array([0.0, 1.0, 2.0, 5.0], dtype=float)

condition = "Tumor+MSC+M2"

data = {
    "OVCAR3": np.array([0.0, 1.37, 3.32, 9.36], dtype=float),
    "OVCAR4": np.array([0.0, 0.30, 0.45, 1.38], dtype=float),
}

# Initial condition for Tumor+MSC+M2
# T(0)=1, S(0)=1, M(0)=1, V(0)=first observed void value
T0, S0, M0 = 1.0, 1.0, 1.0

# Endpoint weight, same idea as your original code
weights = np.array([1.0, 1.0, 1.0, 1.5], dtype=float)



#    etaSM and etaMS are estimated separately for each cell line.

fixed_params = {
    "OVCAR3": dict(
        KT=50.0,
        KS=4.85965,
        KM=14.1112,
        Vmax=50.0,
        rho=1.22477,

        rS=0.879043,
        rM=0.99387,
        dS=0.502941,
        dM=0.0554736,

        etaTS=1.6506,
        etaST=-2.70125,
        etaMT=-4.98143,

        rT=0.753007,
        dT=5.03247e-34,
        beta=8.26125,
        aTS=34.8529,
        aTM=3.80526,
        etaTM=-1.16454,
    ),

    "OVCAR4": dict(
        KT=50.0,
        KS=49.9971,
        KM=4.95469,
        Vmax=50.0,
        rho=5.0,

        rS=1.55336e-39,
        rM=0.5052,
        dS=0.208744,
        dM=0.243386,

        etaTS=-3.27475,
        etaST=5.0,
        etaMT=2.48226,

        rT=0.596518,
        dT=2.45687e-46,
        beta=53.3633,
        aTS=18.7049,
        aTM=-1.17822,
        etaTM=-0.0911392,
    ),
}


def frac(x, K):
    return x / (K + x)


def Phi(T, S, M, KT, KS, KM, aTS, aTM):
    return frac(T, KT) * (1.0 + aTS * frac(S, KS)) * (1.0 + aTM * frac(M, KM))


def rhs_tsmv(T, S, M, V, p):
    KT, KS, KM = p["KT"], p["KS"], p["KM"]
    rT, rS, rM = p["rT"], p["rS"], p["rM"]
    dT, dS, dM = p["dT"], p["dS"], p["dM"]

    etaTS, etaTM = p["etaTS"], p["etaTM"]
    etaST, etaMT = p["etaST"], p["etaMT"]

    # Newly estimated reciprocal MSC <-> M2 coupling parameters
    etaSM, etaMS = p["etaSM"], p["etaMS"]

    Vmax, rho = p["Vmax"], p["rho"]
    beta, aTS, aTM = p["beta"], p["aTS"], p["aTM"]

    dTdt = (
        rT * T * (1.0 - T / KT)
        + etaTS * T * frac(S, KS)
        + etaTM * T * frac(M, KM)
        - dT * T
    )

    dSdt = (
        rS * S * (1.0 - S / KS)
        + etaST * S * frac(T, KT)
        + etaSM * S * frac(M, KM)
        - dS * S
    )

    dMdt = (
        rM * M * (1.0 - M / KM)
        + etaMT * M * frac(T, KT)
        + etaMS * M * frac(S, KS)
        - dM * M
    )

    dVdt = (
        beta * Phi(T, S, M, KT, KS, KM, aTS, aTM) * (1.0 - V / Vmax)
        - rho * V
    )

    return dTdt, dSdt, dMdt, dVdt


def step_TSMV_euler(T, S, M, V, dt, p):
    dTdt, dSdt, dMdt, dVdt = rhs_tsmv(T, S, M, V, p)

    Tn = max(T + dt * dTdt, 0.0)
    Sn = max(S + dt * dSdt, 0.0)
    Mn = max(M + dt * dMdt, 0.0)
    Vn = max(V + dt * dVdt, 0.0)

    return Tn, Sn, Mn, Vn


def simulate_points(t_points, y0_TSMV, p, substeps=100):
    T, S, M, V = map(float, y0_TSMV)
    V_out = [V]

    for i in range(len(t_points) - 1):
        dt_big = float(t_points[i + 1] - t_points[i])
        dt = dt_big / substeps

        for _ in range(substeps):
            T, S, M, V = step_TSMV_euler(T, S, M, V, dt, p)

        V_out.append(V)

    return np.array(V_out, dtype=float)



def make_parameter_dict(cell_line, x):
    p = fixed_params[cell_line].copy()
    p["etaSM"] = float(x[0])
    p["etaMS"] = float(x[1])
    return p


def fit_etaSM_etaMS_for_cell_line(cell_line):
    V_obs = data[cell_line]
    y0 = [T0, S0, M0, float(V_obs[0])]

    def residuals(x):
        p = make_parameter_dict(cell_line, x)
        V_hat = simulate_points(t_data, y0, p, substeps=SUBSTEPS_FIT)
        return weights * (V_hat - V_obs)

    # Initial guess for etaSM and etaMS
    x0 = np.array([0.0, 0.0], dtype=float)

    # Bounds for etaSM and etaMS
    lb = np.array([-5.0, -5.0], dtype=float)
    ub = np.array([5.0, 5.0], dtype=float)

    fit = least_squares(
        residuals,
        x0,
        bounds=(lb, ub),
        method="trf",
        max_nfev=5000,
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
        x_scale="jac",
    )

    p_best = make_parameter_dict(cell_line, fit.x)
    V_fit = simulate_points(t_data, y0, p_best, substeps=SUBSTEPS_FIT)

    rmse = float(np.sqrt(np.mean((V_fit - V_obs) ** 2)))
    weighted_rmse = float(np.sqrt(np.mean((weights * (V_fit - V_obs)) ** 2)))

    result = {
        "cell_line": cell_line,
        "fit": fit,
        "p_best": p_best,
        "etaSM": float(fit.x[0]),
        "etaMS": float(fit.x[1]),
        "V_obs": V_obs,
        "V_fit": V_fit,
        "rmse": rmse,
        "weighted_rmse": weighted_rmse,
    }

    return result



results = {}

for cl in ["OVCAR3", "OVCAR4"]:
    results[cl] = fit_etaSM_etaMS_for_cell_line(cl)


print("\n============================================================")
print("Two-parameter fits using fixed 18-parameter sets")
print("Condition: Tumor+MSC+M2")
print("Estimated parameters: etaSM and etaMS")
print("============================================================")

summary_lines = []
summary_lines.append("=== Estimated etaSM and etaMS for Tumor+MSC+M2 ===\n")

for cl, res in results.items():
    fit = res["fit"]

    print(f"\n\n==================== {cl} ====================")
    print("Success:", fit.success)
    print("Message:", fit.message)
    print("Cost:", fit.cost)

    print("\nEstimated new parameters:")
    print(f"etaSM = {res['etaSM']:.10g}")
    print(f"etaMS = {res['etaMS']:.10g}")

    print("\nModel fit at data times:")
    for t, obs, pred in zip(t_data, res["V_obs"], res["V_fit"]):
        print(
            f"t = {t:4.1f} days | "
            f"data = {obs: .6f} | "
            f"model = {pred: .6f} | "
            f"residual = {pred - obs: .6f}"
        )

    print("\nError metrics:")
    print(f"RMSE          = {res['rmse']:.10g}")
    print(f"Weighted RMSE = {res['weighted_rmse']:.10g}")

    summary_lines.append(f"[{cl}]")
    summary_lines.append(f"etaSM = {res['etaSM']:.10g}")
    summary_lines.append(f"etaMS = {res['etaMS']:.10g}")
    summary_lines.append(f"Success = {fit.success}")
    summary_lines.append(f"Message = {fit.message}")
    summary_lines.append(f"Cost = {fit.cost:.10g}")
    summary_lines.append(f"RMSE = {res['rmse']:.10g}")
    summary_lines.append(f"Weighted RMSE = {res['weighted_rmse']:.10g}")
    summary_lines.append("Data vs model:")
    for t, obs, pred in zip(t_data, res["V_obs"], res["V_fit"]):
        summary_lines.append(
            f"t={t:.1f}, data={obs:.10g}, model={pred:.10g}, residual={pred - obs:.10g}"
        )
    summary_lines.append("")

outfile = "estimated_etaSM_etaMS_both_cell_lines.txt"
with open(outfile, "w") as f:
    f.write("\n".join(summary_lines))

print(f"\nSaved parameter estimates to: {outfile}")


# PLOTS: BASELINE VS EXTENDED MODEL

def plot_cell_line_fit(cell_line, res):
    V_obs = res["V_obs"]

    # Extended model: uses estimated etaSM and etaMS
    p_extended = res["p_best"].copy()

    # Baseline model: same fixed 18-parameter model, but etaSM = etaMS = 0
    p_baseline = p_extended.copy()
    p_baseline["etaSM"] = 0.0
    p_baseline["etaMS"] = 0.0

    y0 = [T0, S0, M0, float(V_obs[0])]

    t_fine = np.linspace(0.0, 5.0, 1001)

    V_baseline = simulate_points(
        t_fine, y0, p_baseline, substeps=SUBSTEPS_PLOT
    )

    V_extended = simulate_points(
        t_fine, y0, p_extended, substeps=SUBSTEPS_PLOT
    )

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    ax.plot(
        t_fine,
        V_baseline,
        color="tab:green",
        linewidth=2.2,
        linestyle="--",
        label="Baseline model"
    )

    ax.plot(
        t_fine,
        V_extended,
        color="tab:red",
        linewidth=2.2,
        linestyle="-",
        label="Extended model"
    )

    ax.scatter(
        t_data,
        V_obs,
        s=55,
        marker="o",
        color="black",
        zorder=5,
        label="Data"
    )

    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Normalized LP-9 Void Area")

    ax.set_title(
        f"{cell_line} Tumor+MSC+M2: baseline vs. extended model"
    )

    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)

    ax.legend(frameon=False)

    plt.tight_layout()

    pdf_name = f"{cell_line}_Tumor_MSC_M2_baseline_vs_extended.pdf"
    fig.savefig(pdf_name, bbox_inches="tight")

    return fig, pdf_name


saved_figs = []

for cl, res in results.items():
    fig, pdf_name = plot_cell_line_fit(cl, res)
    saved_figs.append(pdf_name)

plt.show()

print("\nSaved figures:")
for pdf_name in saved_figs:
    print(pdf_name)
