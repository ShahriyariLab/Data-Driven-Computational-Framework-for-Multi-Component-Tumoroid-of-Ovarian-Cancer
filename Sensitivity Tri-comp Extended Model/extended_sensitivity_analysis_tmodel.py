#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PRCC sensitivity analysis for Tumor+MSC+M2 extended model
----------------------------------------------------
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from numpy.linalg import lstsq


SEED = 123

# Simulation settings
SUBSTEPS = 50
T_END = 5.0
N_TIME = 301
t_fine = np.linspace(0.0, T_END, N_TIME)

# PRCC settings
N_SAMPLES = 2000
RANGE_FRAC = 0.30
ZERO_THR = 1e-8


plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "axes.linewidth": 0.8,
    "axes.grid": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

COLORS = {
    "blue":   "#4C78A8",
    "red":    "#E45756",
    "green":  "#54A24B",
    "orange": "#F58518",
    "purple": "#B279A2",
    "gray":   "#6C757D",
    "dark":   "#222222",
}

def full_box(ax):
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.8)

def subtle_zero(ax):
    ax.axhline(0, color="#333333", linewidth=0.9, zorder=0)


# TUMOR+MSC+M2 INITIAL CONDITION ONLY

CONDITION_NAME = "Tumor+MSC+M2"

# Initial condition: T(0)=1, S(0)=1, M(0)=1, V(0)=0
Y0_TSMV = [1.0, 1.0, 1.0, 0.0]


# BEST-FIT PARAMETERS

BEST_FITS = {
    "OVCAR3": {
        "KT": 50,
        "KS": 4.85965,
        "KM": 14.1112,
        "Vmax": 50,
        "rho": 1.22477,
        "rS": 0.879043,
        "rM": 0.99387,
        "dS": 0.502941,
        "dM": 0.0554736,
        "etaTS": 1.6506,
        "etaST": -2.70125,
        "etaMT": -4.98143,
        "rT": 0.753007,
        "dT": 5.03247e-34,
        "beta": 8.26125,
        "aTS": 34.8529,
        "aTM": 3.80526,
        "etaTM": -1.16454,
        "etaSM": -0.0696322592,
        "etaMS": -0.3006346204,
    },
    "OVCAR4": {
        "KT": 50,
        "KS": 49.9971,
        "KM": 4.95469,
        "Vmax": 50,
        "rho": 5,
        "rS": 1.55336e-39,
        "rM": 0.5052,
        "dS": 0.208744,
        "dM": 0.243386,
        "etaTS": -3.27475,
        "etaST": 5,
        "etaMT": 2.48226,
        "rT": 0.596518,
        "dT": 2.45687e-46,
        "beta": 53.3633,
        "aTS": 18.7049,
        "aTM": -1.17822,
        "etaTM": -0.0911392,
        "etaSM": 0.08106281916,
        "etaMS": 1.361609009,
    },
}

param_names_20 = [
    "KT", "KS", "KM", "Vmax", "rho",
    "rS", "rM", "dS", "dM",
    "etaTS", "etaST", "etaMT",
    "rT", "dT", "beta", "aTS", "aTM", "etaTM",
    "etaSM", "etaMS"
]

LATEX = {
    "KT":    r"$K_T$",
    "KS":    r"$K_S$",
    "KM":    r"$K_M$",
    "Vmax":  r"$V_{\max}$",
    "rho":   r"$\rho$",
    "rS":    r"$r_S$",
    "rM":    r"$r_M$",
    "dS":    r"$d_S$",
    "dM":    r"$d_M$",
    "etaTS": r"$\eta_{TS}$",
    "etaST": r"$\eta_{ST}$",
    "etaMT": r"$\eta_{MT}$",
    "rT":    r"$r_T$",
    "dT":    r"$d_T$",
    "beta":  r"$\beta$",
    "aTS":   r"$\alpha_{TS}$",
    "aTM":   r"$\alpha_{TM}$",
    "etaTM": r"$\eta_{TM}$",
    "etaSM": r"$\eta_{SM}$",
    "etaMS": r"$\eta_{MS}$",
}

NONNEG = {
    "KT", "KS", "KM", "Vmax", "rho",
    "rS", "rM", "dS", "dM", "rT", "dT", "beta", "aTS"
}


ACTIVE_PARAMS = [
    "KT", "KS", "KM", "Vmax", "rho",
    "rS", "rM", "dS", "dM",
    "etaTS", "etaST", "etaMT",
    "rT", "dT", "beta", "aTS", "aTM", "etaTM",
    "etaSM", "etaMS"
]


PARAM_ORDER = [
    "rT", "beta", "rho", "KT",
    "rS", "dS", "KS", "etaTS", "etaST", "aTS",
    "rM", "dM", "KM", "etaTM", "etaMT", "aTM",
    "etaSM", "etaMS",
    "Vmax", "dT"
]

GROUP_BREAKS = [4, 10, 16, 18]

def get_active_params(p_best):
    return [p for p in ACTIVE_PARAMS if abs(float(p_best[p])) >= ZERO_THR]

def get_plot_parameter_order(df):
    return [p for p in PARAM_ORDER if p in df.index]


# MODEL

def frac(x, K):
    den = K + x
    if den <= 1e-14:
        return 0.0
    return x / den

def Phi(T, S, M, KT, KS, KM, aTS, aTM):
    return frac(T, KT) * (1.0 + aTS * frac(S, KS)) * (1.0 + aTM * frac(M, KM))

def rhs_all(T, S, M, V, p):
    KT, KS, KM = p["KT"], p["KS"], p["KM"]
    Vmax, rho = p["Vmax"], p["rho"]
    rT, rS, rM = p["rT"], p["rS"], p["rM"]
    dT, dS, dM = p["dT"], p["dS"], p["dM"]

    etaTS = p["etaTS"]
    etaST = p["etaST"]
    etaMT = p["etaMT"]
    etaTM = p["etaTM"]

    etaSM = p["etaSM"]
    etaMS = p["etaMS"]

    beta = p["beta"]
    aTS = p["aTS"]
    aTM = p["aTM"]

    fracT = frac(T, KT)
    fracS = frac(S, KS)
    fracM = frac(M, KM)

    dTdt = (
        rT * T * (1.0 - T / KT)
        + etaTS * T * fracS
        + etaTM * T * fracM
        - dT * T
    )

    dSdt = (
        rS * S * (1.0 - S / KS)
        + etaST * S * fracT
        + etaSM * S * fracM
        - dS * S
    )

    dMdt = (
        rM * M * (1.0 - M / KM)
        + etaMT * M * fracT
        + etaMS * M * fracS
        - dM * M
    )

    ph = Phi(T, S, M, KT, KS, KM, aTS, aTM)

    dVdt = beta * ph * (1.0 - V / Vmax) - rho * V

    return dTdt, dSdt, dMdt, dVdt

def step_TSMV_euler(T, S, M, V, dt, p):
    dTdt, dSdt, dMdt, dVdt = rhs_all(T, S, M, V, p)

    Tn = max(T + dt * dTdt, 0.0)
    Sn = max(S + dt * dSdt, 0.0)
    Mn = max(M + dt * dMdt, 0.0)
    Vn = max(V + dt * dVdt, 0.0)

    return Tn, Sn, Mn, Vn

def simulate_points(t_points, y0_TSMV, p, substeps=SUBSTEPS):
    T, S, M, V = map(float, y0_TSMV)

    Tout = [T]
    Sout = [S]
    Mout = [M]
    Vout = [V]

    for i in range(len(t_points) - 1):
        dt_big = float(t_points[i + 1] - t_points[i])
        dt = dt_big / substeps

        for _ in range(substeps):
            T, S, M, V = step_TSMV_euler(T, S, M, V, dt, p)

        Tout.append(T)
        Sout.append(S)
        Mout.append(M)
        Vout.append(V)

    Tout = np.array(Tout, dtype=float)
    Sout = np.array(Sout, dtype=float)
    Mout = np.array(Mout, dtype=float)
    Vout = np.array(Vout, dtype=float)

    return {
        "T": Tout,
        "S": Sout,
        "M": Mout,
        "V": Vout,
        "Ctot": Tout + Sout + Mout,
    }

def simulate_metric(p, metric_key):
    sol = simulate_points(t_fine, Y0_TSMV, p, substeps=SUBSTEPS)
    return sol[metric_key]

def auc_trapz(t, y):
    return float(np.trapz(y, t))


# METRICS

metric_info = {
    "V": {
        "title": r"LP-9 void area $V(t)$",
        "ylabel": r"$V(t)$",
        "short": "V",
    },
    "T": {
        "title": r"Tumor cells $T(t)$",
        "ylabel": r"$T(t)$",
        "short": "T",
    },
    "Ctot": {
        "title": r"Total cancer-associated burden $T(t)+S(t)+M(t)$",
        "ylabel": r"$T(t)+S(t)+M(t)$",
        "short": "Ctot",
    },
}


# PRCC

def lhs(n, d, rng):
    X = np.zeros((n, d), dtype=float)

    for j in range(d):
        cut = np.linspace(0.0, 1.0, n + 1)
        u = rng.uniform(cut[:-1], cut[1:], size=n)
        rng.shuffle(u)
        X[:, j] = u

    return X

def make_bounds(best, keys, frac=0.30):
    lo = []
    hi = []

    for k in keys:
        v = float(best[k])

        if abs(v) < 1e-14:
            lo_k, hi_k = -frac, frac
        else:
            a = v * (1.0 - frac)
            b = v * (1.0 + frac)
            lo_k, hi_k = min(a, b), max(a, b)

        if k in NONNEG:
            lo_k = max(lo_k, 0.0)

        lo.append(lo_k)
        hi.append(hi_k)

    return np.array(lo, dtype=float), np.array(hi, dtype=float)

def residualize(y, Xcov):
    if Xcov.size == 0:
        return y.copy()

    A = np.column_stack([np.ones(Xcov.shape[0]), Xcov])
    beta, *_ = lstsq(A, y, rcond=None)

    return y - A @ beta

def prcc(X_in, Y_in, x_names, y_name):
    Xr = np.column_stack([
        rankdata(X_in[:, j], method="average")
        for j in range(X_in.shape[1])
    ]).astype(float)

    Yr = rankdata(Y_in, method="average").astype(float)

    P = Xr.shape[1]
    out = np.zeros(P, dtype=float)

    for j in range(P):
        others = [k for k in range(P) if k != j]
        Xo = Xr[:, others] if others else np.empty((Xr.shape[0], 0))

        xj = residualize(Xr[:, j], Xo)
        yq = residualize(Yr, Xo)

        xj = (xj - xj.mean()) / (xj.std(ddof=1) + 1e-14)
        yq = (yq - yq.mean()) / (yq.std(ddof=1) + 1e-14)

        out[j] = np.corrcoef(xj, yq)[0, 1]

    return pd.Series(out, index=x_names, name=y_name)


# PLOTTING HELPERS

def add_group_breaks(ax, nbars):
    for gb in GROUP_BREAKS:
        if 0 < gb < nbars:
            ax.axvline(
                gb - 0.5,
                color=COLORS["gray"],
                linewidth=0.8,
                linestyle="--",
                alpha=0.7
            )

def prcc_bar_plot(df, title, ylabel, outfile, outdir):
    keep = get_plot_parameter_order(df)
    dfp = df.loc[keep, :]

    xtick_labels = [LATEX.get(k, k) for k in keep]

    fig, ax = plt.subplots(figsize=(9.5, 4.8))

    vals = dfp.iloc[:, 0].values
    x = np.arange(len(dfp.index))

    ax.bar(
        x,
        vals,
        color=COLORS["blue"],
        edgecolor=COLORS["dark"],
        linewidth=0.5,
        width=0.72
    )

    subtle_zero(ax)
    add_group_breaks(ax, len(dfp.index))

    ax.set_title(title, pad=8)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(
        xtick_labels,
        rotation=65,
        ha="right",
        fontsize=10
    )

    ax.set_ylim(-1.05, 1.05)
    ax.margins(x=0.01)
    full_box(ax)

    fig.subplots_adjust(bottom=0.30)
    fig.savefig(os.path.join(outdir, outfile), bbox_inches="tight")
    plt.close(fig)


# PRCC ANALYSIS

def run_prcc_analysis(p_best, outdir, cell_line):
    print(f"\nRunning Tumor+MSC+M2 PRCC analysis for {cell_line}...")

    rng = np.random.default_rng(SEED)

    active_params = get_active_params(p_best)
    lo, hi = make_bounds(p_best, active_params, frac=RANGE_FRAC)

    print(f"Active parameters for {cell_line}:")
    print(active_params)

    U01 = lhs(N_SAMPLES, len(active_params), rng)
    X = lo + U01 * (hi - lo)

    # Store sampled parameter values
    sample_df = pd.DataFrame(X, columns=active_params)
    sample_df.to_csv(
        os.path.join(outdir, f"{cell_line}_Tumor_MSC_M2_parameter_samples.csv"),
        index=False
    )

    for mk, info in metric_info.items():
        y_auc = np.zeros(N_SAMPLES, dtype=float)
        y_fin = np.zeros(N_SAMPLES, dtype=float)

        for i in range(N_SAMPLES):
            p = dict(p_best)

            for j, k in enumerate(active_params):
                p[k] = float(X[i, j])

            y = simulate_metric(p, mk)

            y_auc[i] = auc_trapz(t_fine, y)
            y_fin[i] = y[-1]

        prcc_auc = prcc(
            X,
            y_auc,
            active_params,
            f"AUC_{info['short']}|{CONDITION_NAME}"
        )

        prcc_fin = prcc(
            X,
            y_fin,
            active_params,
            f"{info['short']}_final|{CONDITION_NAME}"
        )

        df_auc = prcc_auc.to_frame()
        df_fin = prcc_fin.to_frame()

        df_auc.to_csv(
            os.path.join(outdir, f"prcc_Tumor_MSC_M2_{mk}_AUC.csv")
        )
        df_fin.to_csv(
            os.path.join(outdir, f"prcc_Tumor_MSC_M2_{mk}_final.csv")
        )

        prcc_bar_plot(
            df_auc,
            title=f"{cell_line}: PRCC on AUC of {info['title']} | Tumor+MSC+M2",
            ylabel="PRCC",
            outfile=f"prcc_Tumor_MSC_M2_{mk}_AUC.pdf",
            outdir=outdir,
        )

        prcc_bar_plot(
            df_fin,
            title=f"{cell_line}: PRCC on final value of {info['title']} | Tumor+MSC+M2",
            ylabel="PRCC",
            outfile=f"prcc_Tumor_MSC_M2_{mk}_final.pdf",
            outdir=outdir,
        )

        # Save raw metric outputs as well
        metric_output_df = pd.DataFrame({
            f"AUC_{mk}": y_auc,
            f"Final_{mk}": y_fin,
        })

        metric_output_df.to_csv(
            os.path.join(outdir, f"{cell_line}_Tumor_MSC_M2_outputs_{mk}.csv"),
            index=False
        )


# MAIN

if __name__ == "__main__":

    for cell_line, p_best in BEST_FITS.items():

        outdir = f"sensitivity_Tumor_MSC_M2_{cell_line}"
        os.makedirs(outdir, exist_ok=True)

        print("\n" + "=" * 70)
        print(f"Running PRCC analysis for {cell_line}")
        print(f"Configuration: Tumor+MSC+M2 only")
        print(f"Output folder: {outdir}")
        print("=" * 70)

        run_prcc_analysis(p_best, outdir, cell_line)

    print("\nDone. Outputs saved in:")
    print("  sensitivity_Tumor_MSC_M2_OVCAR3/")
    print("  sensitivity_Tumor_MSC_M2_OVCAR4/")
