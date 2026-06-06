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
    "lines.linewidth": 1.5,
    "legend.frameon": False,
    "legend.fontsize": 11,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# DATA
t_data = np.array([0.0, 1.0, 2.0, 5.0], dtype=float)

conditions_all = {
    "Tumor":        (1.0, 0.0, 0.0),
    "Tumor+MSC":    (1.0, 1.0, 0.0),
    "Tumor+M2":     (1.0, 0.0, 1.0),
    "Tumor+MSC+M2": (1.0, 1.0, 1.0),
}

# Fitting only on these three exp configurations
fit_conditions = {
    "Tumor":     conditions_all["Tumor"],
    "Tumor+MSC": conditions_all["Tumor+MSC"],
    "Tumor+M2":  conditions_all["Tumor+M2"],
}

# Held-out validation condition
validation_condition = "Tumor+MSC+M2"

data = {
    "OVCAR3": {
        "Tumor":        np.array([0.0, 0.19, 0.29, 1.62]),
        "Tumor+MSC":    np.array([0.0, 1.26, 3.47, 7.21]),
        "Tumor+M2":     np.array([0.0, 0.19, 0.44, 1.69]),
        "Tumor+MSC+M2": np.array([0.0, 1.37, 3.32, 9.36]),
    },
    "OVCAR4": {
        "Tumor":        np.array([0.0, 0.35, 0.55, 2.13]),
        "Tumor+MSC":    np.array([0.0, 0.45, 0.65, 2.93]),
        "Tumor+M2":     np.array([0.0, 0.25, 0.40, 1.00]),
        "Tumor+MSC+M2": np.array([0.0, 0.30, 0.45, 1.38]),
    },
}

# Slightly focus on endpoint during fitting
weights = {
    cl: {
        c: np.array([1.0, 1.0, 1.0, 1.5], dtype=float)
        for c in fit_conditions
    }
    for cl in data
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
    Vmax, rho = p["Vmax"], p["rho"]
    beta, aTS, aTM = p["beta"], p["aTS"], p["aTM"]

    dTdt = rT * T * (1.0 - T / KT) + etaTS * T * frac(S, KS) + etaTM * T * frac(M, KM) - dT * T
    dSdt = rS * S * (1.0 - S / KS) + etaST * S * frac(T, KT) - dS * S
    dMdt = rM * M * (1.0 - M / KM) + etaMT * M * frac(T, KT) - dM * M
    dVdt = beta * Phi(T, S, M, KT, KS, KM, aTS, aTM) * (1.0 - V / Vmax) - rho * V

    return dTdt, dSdt, dMdt, dVdt

def step_TSMV_euler(T, S, M, V, dt, p):
    dTdt, dSdt, dMdt, dVdt = rhs_tsmv(T, S, M, V, p)

    Tn = max(T + dt * dTdt, 0.0)
    Sn = max(S + dt * dSdt, 0.0)
    Mn = max(M + dt * dMdt, 0.0)
    Vn = max(V + dt * dVdt, 0.0)

    return Tn, Sn, Mn, Vn

def simulate_points(t_points, y0_TSMV, p, substeps=200):
    """
    Simulate and return V evaluated at the given t_points.
    Uses forward Euler for all four variables.
    """
    T, S, M, V = map(float, y0_TSMV)
    V_out = [V]

    for i in range(len(t_points) - 1):
        t0 = float(t_points[i])
        t1 = float(t_points[i + 1])
        dt_big = t1 - t0
        dt = dt_big / substeps

        for _ in range(substeps):
            T, S, M, V = step_TSMV_euler(T, S, M, V, dt, p)

        V_out.append(V)

    return np.array(V_out, dtype=float)


param_names = [
    "KT", "KS", "KM", "Vmax", "rho",
    "rS", "rM", "dS", "dM",
    "etaTS", "etaST", "etaMT",
    "rT", "dT", "beta", "aTS", "aTM", "etaTM"
]

def unpack_params(x):
    return dict(
        KT=x[0], KS=x[1], KM=x[2], Vmax=x[3], rho=x[4],
        rS=x[5], rM=x[6], dS=x[7], dM=x[8],
        etaTS=x[9], etaST=x[10], etaMT=x[11],
        rT=x[12], dT=x[13], beta=x[14], aTS=x[15], aTM=x[16], etaTM=x[17]
    )

x0 = np.array([
    2.0, 2.0, 2.0, 12.0, 0.05,
    0.05, 0.05, 0.02, 0.02,
    0.30, 0.10, 0.05,
    0.10, 0.02, 6.0, 6.0, 0.05, 0.00
], dtype=float)

lb = np.array([
    1e-3, 1e-3, 1e-3,  1.0, 0.0,
    0.0,  0.0,  0.0,   0.0,
   -5.0, -5.0, -5.0,
    0.0,  0.0,  0.0,   0.0, -5.0, -5.0,
], dtype=float)

ub = np.array([
    50.0, 50.0, 50.0, 50.0, 5.0,
    2.0,  2.0,  2.0,  2.0,
    5.0,  5.0,  5.0,
    2.0,  2.0, 200.0, 50.0, 5.0, 5.0,
], dtype=float)


def residuals_one_cell(x, cell_line):
    p = unpack_params(x)
    res = []

    for cond, (T0, S0, M0) in fit_conditions.items():
        y_obs = data[cell_line][cond]
        w = weights[cell_line][cond]

        # initialize V from first observed value in that condition
        y0 = [T0, S0, M0, float(y_obs[0])]
        V_hat = simulate_points(t_data, y0, p, substeps=200)

        res.append(w * (V_hat - y_obs))

    return np.concatenate(res)

def fit_cell_line(cell_line):
    fit = least_squares(
        residuals_one_cell,
        x0,
        bounds=(lb, ub),
        args=(cell_line,),
        method="trf",
        max_nfev=4000,
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
        x_scale="jac",
    )
    return fit, unpack_params(fit.x)


# FIT OVCAR3 AND OVCAR4 SEPARATELY
fit3, p3 = fit_cell_line("OVCAR3")
fit4, p4 = fit_cell_line("OVCAR4")

print("\n=== FIT STATUS ===")
print("\nOVCAR3")
print("Success:", fit3.success)
print("Message:", fit3.message)
print("Cost:", fit3.cost)

print("\nOVCAR4")
print("Success:", fit4.success)
print("Message:", fit4.message)
print("Cost:", fit4.cost)


# RMSE

def rmse_on_fit_set(cell_line, p):
    err2 = []
    for cond, (T0, S0, M0) in fit_conditions.items():
        y_obs = data[cell_line][cond]
        y0 = [T0, S0, M0, float(y_obs[0])]
        V_hat = simulate_points(t_data, y0, p, substeps=200)
        err2.append((V_hat - y_obs) ** 2)
    err2 = np.concatenate(err2)
    return float(np.sqrt(np.mean(err2)))

def rmse_on_validation(cell_line, p):
    T0, S0, M0 = conditions_all[validation_condition]
    y_obs = data[cell_line][validation_condition]
    y0 = [T0, S0, M0, float(y_obs[0])]
    V_hat = simulate_points(t_data, y0, p, substeps=200)
    return float(np.sqrt(np.mean((V_hat - y_obs) ** 2))), V_hat

rmse3_fit = rmse_on_fit_set("OVCAR3", p3)
rmse4_fit = rmse_on_fit_set("OVCAR4", p4)

rmse3_val, Vhat3_val = rmse_on_validation("OVCAR3", p3)
rmse4_val, Vhat4_val = rmse_on_validation("OVCAR4", p4)

print("\n=== RMSE SUMMARY ===")
print(f"OVCAR3 fit-set RMSE: {rmse3_fit:.6f}")
print(f"OVCAR3 tri-component validation RMSE: {rmse3_val:.6f}")
print(f"OVCAR4 fit-set RMSE: {rmse4_fit:.6f}")
print(f"OVCAR4 tri-component validation RMSE: {rmse4_val:.6f}")


def write_best_fit(cell_line, fit_obj, p, outfile):
    lines = []
    lines.append(f"=== Best-fit parameters for {cell_line} ===")
    for k in param_names:
        lines.append(f"{k:>8s} = {p[k]: .6g}")

    lines.append("\n=== Fit summary ===")
    lines.append(f"Success: {fit_obj.success}")
    lines.append(f"Message: {fit_obj.message}")
    lines.append(f"Cost:    {fit_obj.cost:.6g}")
    lines.append(f"Fit-set RMSE: {rmse_on_fit_set(cell_line, p):.6g}")

    val_rmse, _ = rmse_on_validation(cell_line, p)
    lines.append(f"Held-out Tumor+MSC+M2 RMSE: {val_rmse:.6g}")

    txt = "\n".join(lines)
    print("\n" + txt)

    with open(outfile, "w") as f:
        f.write(txt + "\n")

write_best_fit("OVCAR3", fit3, p3, "best_fit_parameters_OVCAR3_validation.txt")
write_best_fit("OVCAR4", fit4, p4, "best_fit_parameters_OVCAR4_validation.txt")


def _style_axes(ax):
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)

def predicted_values_from_curve(t_dense, V_dense, t_obs):
    return np.interp(t_obs, t_dense, V_dense)

def plot_fit_and_validation(cell_line, p, save_name):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=False)
    t_fine = np.linspace(0, 5, 1501)

    
    # Left: calibration
    
    ax = axes[0]
    for cond, (T0, S0, M0) in fit_conditions.items():
        y_obs = data[cell_line][cond]
        y0 = [T0, S0, M0, float(y_obs[0])]

        V_fine = simulate_points(t_fine, y0, p, substeps=200)

        ax.plot(t_fine, V_fine, linewidth=2, label=f"{cond} (Fit)")
        ax.scatter(t_data, y_obs, s=42, marker="o", zorder=3, label=f"{cond} (Data)")

    ax.set_title(f"{cell_line}: Calibration")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Normalized LP-9 Void Area")
    _style_axes(ax)
    ax.legend(ncol=2, fontsize=9)

    
    # Right: held-out validation
    
    ax = axes[1]
    T0, S0, M0 = conditions_all[validation_condition]
    y_obs = data[cell_line][validation_condition]
    y0 = [T0, S0, M0, float(y_obs[0])]

    V_fine = simulate_points(t_fine, y0, p, substeps=200)
    V_hat = predicted_values_from_curve(t_fine, V_fine, t_data)

    ax.plot(t_fine, V_fine, color="black", linewidth=2.2, label="Model Prediction")
    ax.scatter(t_data, y_obs, s=50, color="crimson", marker="o", zorder=3, label="Held-out Data")
    # ax.scatter(t_data, V_hat, s=40, color="black", marker="s", zorder=3, label="Predicted Values")

    ax.set_title(f"{cell_line}: Held-out Tri-component Model Validation")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Normalized LP-9 Void Area")
    _style_axes(ax)
    ax.legend(fontsize=10)

    plt.tight_layout()
    fig.savefig(save_name, bbox_inches="tight")
    plt.close(fig)

def plot_heldout_only(cell_line, p, save_name):
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    t_fine = np.linspace(0, 5, 1501)

    T0, S0, M0 = conditions_all[validation_condition]
    y_obs = data[cell_line][validation_condition]
    y0 = [T0, S0, M0, float(y_obs[0])]

    V_fine = simulate_points(t_fine, y0, p, substeps=200)
    V_hat = predicted_values_from_curve(t_fine, V_fine, t_data)

    ax.plot(t_fine, V_fine, color="black", linewidth=2.2, label="Model Prediction")
    ax.scatter(t_data, y_obs, s=55, color="crimson", marker="o", zorder=3, label="Experimental Data")
    ax.scatter(t_data, V_hat, s=42, color="black", marker="s", zorder=3, label="Predicted Values")

    ax.set_title(f"{cell_line}: Predicted Tumor+MSC+M2 trajectory")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Normalized LP-9 Void Area")
    _style_axes(ax)
    ax.legend()
    plt.tight_layout()
    fig.savefig(save_name, bbox_inches="tight")
    plt.close(fig)

plot_fit_and_validation("OVCAR3", p3, "OVCAR3_fit_and_tri_component_validation.pdf")
plot_fit_and_validation("OVCAR4", p4, "OVCAR4_fit_and_tri_component_validation.pdf")

plot_heldout_only("OVCAR3", p3, "OVCAR3_tri_component_prediction_only.pdf")
plot_heldout_only("OVCAR4", p4, "OVCAR4_tri_component_prediction_only.pdf")


def save_validation_table(cell_line, p, outfile):
    T0, S0, M0 = conditions_all[validation_condition]
    y_obs = data[cell_line][validation_condition]
    y0 = [T0, S0, M0, float(y_obs[0])]

    t_fine = np.linspace(0, 5, 1501)
    V_fine = simulate_points(t_fine, y0, p, substeps=200)
    V_hat = predicted_values_from_curve(t_fine, V_fine, t_data)

    with open(outfile, "w") as f:
        f.write("time_days,observed_heldout,predicted_heldout\n")
        for t, yo, yp in zip(t_data, y_obs, V_hat):
            f.write(f"{t:.6g},{yo:.6g},{yp:.6g}\n")

save_validation_table("OVCAR3", p3, "OVCAR3_tri_component_validation_table.csv")
save_validation_table("OVCAR4", p4, "OVCAR4_tri_component_validation_table.csv")


def save_calibration_tables(cell_line, p, prefix):
    t_fine = np.linspace(0, 5, 1501)
    for cond, (T0, S0, M0) in fit_conditions.items():
        y_obs = data[cell_line][cond]
        y0 = [T0, S0, M0, float(y_obs[0])]
        V_fine = simulate_points(t_fine, y0, p, substeps=200)
        V_hat = predicted_values_from_curve(t_fine, V_fine, t_data)

        outfile = f"{prefix}_{cond.replace('+', '_')}.csv"
        with open(outfile, "w") as f:
            f.write("time_days,observed,predicted\n")
            for t, yo, yp in zip(t_data, y_obs, V_hat):
                f.write(f"{t:.6g},{yo:.6g},{yp:.6g}\n")

save_calibration_tables("OVCAR3", p3, "OVCAR3_calibration_table")
save_calibration_tables("OVCAR4", p4, "OVCAR4_calibration_table")


print("\nSaved files:")
print("  best_fit_parameters_OVCAR3_validation.txt")
print("  best_fit_parameters_OVCAR4_validation.txt")
print("  OVCAR3_fit_and_tri_component_validation.pdf")
print("  OVCAR4_fit_and_tri_component_validation.pdf")
print("  OVCAR3_tri_component_prediction_only.pdf")
print("  OVCAR4_tri_component_prediction_only.pdf")
print("  OVCAR3_tri_component_validation_table.csv")
print("  OVCAR4_tri_component_validation_table.csv")
print("  OVCAR3_calibration_table_Tumor.csv")
print("  OVCAR3_calibration_table_Tumor_MSC.csv")
print("  OVCAR3_calibration_table_Tumor_M2.csv")
print("  OVCAR4_calibration_table_Tumor.csv")
print("  OVCAR4_calibration_table_Tumor_MSC.csv")
print("  OVCAR4_calibration_table_Tumor_M2.csv")
