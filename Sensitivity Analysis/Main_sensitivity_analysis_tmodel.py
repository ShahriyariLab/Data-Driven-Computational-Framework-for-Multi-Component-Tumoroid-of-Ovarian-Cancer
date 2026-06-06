#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sensitivity analysis for OVCAR3 and OVCAR4
-------------------------------------------------------------

Outputs:
1. PRCC analysis
2. SVD / singular spectrum analysis
3. LP-9 void-area principal response heatmap (for V only)

Saved in:
    sensitivity_OVCAR3/
    sensitivity_OVCAR4/
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from numpy.linalg import lstsq, svd


SEED = 123

# simulation
SUBSTEPS = 50
T_END = 5.0
N_TIME = 301
t_fine = np.linspace(0.0, T_END, N_TIME)

# PRCC settings
N_SAMPLES = 2000
RANGE_FRAC = 0.30
ZERO_THR = 1e-8

# PC/SVD settings
FD_REL_STEP = 0.02
FD_ABS_STEP = 1e-4
PC_KEEP_THRESHOLD = 0.20
MAX_PCS_TO_PLOT = 3


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


# INITIAL CONDITIONS

conditions = {
    "Tumor":        (1.0, 0.0, 0.0),
    "Tumor+MSC":    (1.0, 1.0, 0.0),
    "Tumor+M2":     (1.0, 0.0, 1.0),
    "Tumor+MSC+M2": (1.0, 1.0, 1.0),
}
cond_list = list(conditions.keys())


# BEST-FIT PARAMETERS

# BEST_FITS = {
#     "OVCAR3": {
#         "KT": 48.9142,
#         "KS": 5.11847,
#         "KM": 13.8181,
#         "Vmax": 49.6599,
#         "rho": 4.35393,
#         "rS": 1.61991,
#         "rM": 0.273379,
#         "dS": 5.36761e-08,
#         "dM": 0.974517,
#         "etaTS": 0.125684,
#         "etaST": -5.0,
#         "etaMT": 5.0,
#         "rT": 0.8862,
#         "dT": 2.29489e-42,
#         "beta": 19.7832,
#         "aTS": 20.7617,
#         "aTM": 4.99942,
#         "etaTM": -2.82493,
#     },
#     "OVCAR4": {
#         "KT": 50.0,
#         "KS": 49.8735,
#         "KM": 12.5725,
#         "Vmax": 50.0,
#         "rho": 5.0,
#         "rS": 7.30201e-12,
#         "rM": 0.778821,
#         "dS": 0.161294,
#         "dM": 0.541177,
#         "etaTS": -2.82038,
#         "etaST": 5.0,
#         "etaMT": 0.695097,
#         "rT": 0.589087,
#         "dT": 3.35294e-38,
#         "beta": 54.7095,
#         "aTS": 15.2512,
#         "aTM": -3.02377,
#         "etaTM": -0.192029,
#     }
# }

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
    },
}

param_names_18 = [
    "KT", "KS", "KM", "Vmax", "rho",
    "rS", "rM", "dS", "dM",
    "etaTS", "etaST", "etaMT",
    "rT", "dT", "beta", "aTS", "aTM", "etaTM"
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
}

NONNEG = {
    "KT", "KS", "KM", "Vmax", "rho",
    "rS", "rM", "dS", "dM", "rT", "dT", "beta", "aTS"
}


# CONDITION-SPECIFIC ACTIVE PARAMETERS

TUMOR_PARAMS = ["KT", "Vmax", "rho", "rT", "dT", "beta"]
STROMAL_PARAMS = ["KS", "rS", "dS", "etaTS", "etaST", "aTS"]
MACROPHAGE_PARAMS = ["KM", "rM", "dM", "etaTM", "etaMT", "aTM"]

ACTIVE_PARAMS_BY_CONDITION = {
    "Tumor": TUMOR_PARAMS,
    "Tumor+MSC": TUMOR_PARAMS + STROMAL_PARAMS,
    "Tumor+M2": TUMOR_PARAMS + MACROPHAGE_PARAMS,
    "Tumor+MSC+M2": TUMOR_PARAMS + STROMAL_PARAMS + MACROPHAGE_PARAMS,
}


PARAM_ORDER = [
    "rT", "beta", "rho", "KT",
    "rS", "dS", "KS", "etaTS", "etaST", "aTS",
    "rM", "dM", "KM", "etaTM", "etaMT", "aTM",
    "Vmax", "dT"
]
GROUP_BREAKS = [4, 10, 16]

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
    etaTS, etaST, etaMT, etaTM = p["etaTS"], p["etaST"], p["etaMT"], p["etaTM"]
    beta, aTS, aTM = p["beta"], p["aTS"], p["aTM"]

    fracT = frac(T, KT)
    fracS = frac(S, KS)
    fracM = frac(M, KM)

    dTdt = rT * T * (1.0 - T / KT) + etaTS * T * fracS + etaTM * T * fracM - dT * T
    dSdt = rS * S * (1.0 - S / KS) + etaST * S * fracT - dS * S
    dMdt = rM * M * (1.0 - M / KM) + etaMT * M * fracT - dM * M

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

    Tout, Sout, Mout, Vout = [T], [S], [M], [V]

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


def get_active_params_for_condition(cond, p_best):
    all_param_names = [k for k in param_names_18 if abs(float(p_best[k])) >= ZERO_THR]
    return [p for p in ACTIVE_PARAMS_BY_CONDITION[cond] if p in all_param_names]

def simulate_single_condition(p, cond, metric_key):
    T0, S0, M0 = conditions[cond]
    y0 = [T0, S0, M0, 0.0]
    sol = simulate_points(t_fine, y0, p, substeps=SUBSTEPS)
    return sol[metric_key]


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
    lo, hi = [], []
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
    Xr = np.column_stack([rankdata(X_in[:, j], method="average") for j in range(X_in.shape[1])]).astype(float)
    Yr = rankdata(Y_in, method="average").astype(float)

    P = Xr.shape[1]
    out = np.zeros(P, dtype=float)

    for j in range(P):
        others = [k for k in range(P) if k != j]
        Xo = Xr[:, others] if others else np.empty((Xr.shape[0], 0))

        xj = residualize(Xr[:, j], Xo)
        xj = (xj - xj.mean()) / (xj.std(ddof=1) + 1e-14)

        yq = residualize(Yr, Xo)
        yq = (yq - yq.mean()) / (yq.std(ddof=1) + 1e-14)

        out[j] = np.corrcoef(xj, yq)[0, 1]

    return pd.Series(out, index=x_names, name=y_name)


# SVD

def get_fd_step(k, v):
    base = max(abs(v), 1.0)
    h = max(FD_REL_STEP * base, FD_ABS_STEP)
    if k in NONNEG:
        h = min(h, max(0.49 * max(v, 0.0), FD_ABS_STEP))
    return h

def perturb_param(p, k, delta):
    q = dict(p)
    q[k] = float(q[k] + delta)
    if k in NONNEG:
        q[k] = max(q[k], 0.0)
    return q


def savefig(fig, outdir, fname):
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, fname), bbox_inches="tight")
    plt.close(fig)

def add_group_breaks(ax, nbars):
    for gb in GROUP_BREAKS:
        if 0 < gb < nbars:
            ax.axvline(gb - 0.5, color=COLORS["gray"], linewidth=0.8, linestyle="--", alpha=0.7)

# def prcc_bar_panels(df, title, ylabel, outfile, outdir):
#     keep = get_plot_parameter_order(df)
#     dfp = df.loc[keep, :]
#     xtick_labels = [LATEX.get(k, k) for k in keep]

#     ncond = dfp.shape[1]
#     fig, axes = plt.subplots(1, ncond, figsize=(3.2 * ncond, 3.9), sharey=True)
#     if ncond == 1:
#         axes = [axes]

#     for ax, col in zip(axes, dfp.columns):
#         vals = dfp[col].values
#         x = np.arange(len(dfp.index))
#         ax.bar(x, vals, color=COLORS["blue"], edgecolor=COLORS["dark"],
#                linewidth=0.5, width=0.8)
#         subtle_zero(ax)
#         add_group_breaks(ax, len(dfp.index))
#         ax.set_title(col, pad=6)
#         ax.set_xticks(x)
#         ax.set_xticklabels(xtick_labels, rotation=70, ha="right")
#         full_box(ax)

#     axes[0].set_ylabel(ylabel)
#     fig.suptitle(title, y=1.02, fontsize=12)
#     savefig(fig, outdir, outfile)

def prcc_bar_panels(df, title, ylabel, outfile, outdir):
    keep = get_plot_parameter_order(df)
    dfp = df.loc[keep, :]
    xtick_labels = [LATEX.get(k, k) for k in keep]

    ncond = dfp.shape[1]

    # Keep the panels wide enough for readable labels
    fig, axes = plt.subplots(
        1, ncond,
        figsize=(5.8 * ncond, 4.8),
        sharey=True
    )

    if ncond == 1:
        axes = [axes]

    for ax, col in zip(axes, dfp.columns):
        vals = dfp[col].values
        x = np.arange(len(dfp.index))

        # Larger width reduces gaps between vertical bars
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

        ax.set_title(col, pad=6)
        ax.set_xticks(x)

        # Larger parameter labels
        ax.set_xticklabels(
            xtick_labels,
            rotation=65,
            ha="right",
            fontsize=10
        )

        # Reduce extra empty space at left/right edges
        ax.margins(x=0.01)

        full_box(ax)

    axes[0].set_ylabel(ylabel)

    fig.suptitle(title, y=1.02, fontsize=12)

    # More bottom space for larger rotated labels
    fig.subplots_adjust(bottom=0.28, wspace=0.18)

    fig.savefig(os.path.join(outdir, outfile), bbox_inches="tight")
    plt.close(fig)

def singular_spectrum_plot_by_condition(svals_dict, threshold, title, outfile, outdir):
    fig, axes = plt.subplots(1, len(cond_list), figsize=(3.0 * len(cond_list), 3.1), sharey=True)
    if len(cond_list) == 1:
        axes = [axes]

    for ax, cond in zip(axes, cond_list):
        svals = svals_dict[cond]
        s_norm = svals / max(svals[0], 1e-16)
        x = np.arange(1, min(len(svals), 12) + 1)

        ax.plot(
            x,
            np.log10(np.maximum(s_norm[:len(x)], 1e-16)),
            color=COLORS["dark"],
            linewidth=1.2,
            marker="o",
            markersize=4.0,
            markerfacecolor="white",
            markeredgewidth=1.0
        )
        ax.axhline(np.log10(threshold), color=COLORS["red"], linestyle="--", linewidth=1.0)
        ax.set_title(cond, pad=6)
        ax.set_xlabel("Index")
        full_box(ax)

    axes[0].set_ylabel(r"$\log_{10}(\sigma_i/\sigma_1)$")
    fig.suptitle(title, y=1.03, fontsize=12)
    savefig(fig, outdir, outfile)

def response_heatmap_V_only(pc_maps_dict, cell_line, outdir, max_pcs=3):
    """
    New invasion-focused heatmap:
    rows = configurations
    columns = time
    panels = retained PCs
    metric = V(t) only
    """
    n_pcs = min(max_pcs, max(len(pc_maps_dict[cond]) for cond in cond_list))

    fig, axes = plt.subplots(1, n_pcs, figsize=(4.2 * n_pcs, 3.8), sharey=True)
    if n_pcs == 1:
        axes = [axes]

    stacked_maps = []
    global_absmax = 1e-12

    for pc_idx in range(n_pcs):
        rows = []
        for cond in cond_list:
            if pc_idx < len(pc_maps_dict[cond]):
                rows.append(pc_maps_dict[cond][pc_idx])
            else:
                rows.append(np.zeros_like(t_fine))
        H = np.vstack(rows)
        stacked_maps.append(H)
        global_absmax = max(global_absmax, np.max(np.abs(H)))

    im = None
    for pc_idx, ax in enumerate(axes):
        H = stacked_maps[pc_idx]
        im = ax.imshow(
            H,
            aspect="auto",
            cmap="coolwarm",
            extent=[t_fine[0], t_fine[-1], len(cond_list)-0.5, -0.5],
            interpolation="nearest",
            vmin=-global_absmax,
            vmax=global_absmax
        )
        ax.set_title(f"PC{pc_idx+1}", pad=6)
        ax.set_xlabel("Time (days)")
        ax.set_yticks(np.arange(len(cond_list)))
        ax.set_yticklabels(cond_list)
        full_box(ax)

    axes[0].set_ylabel("Configuration")
    fig.suptitle(f"{cell_line}: principal response modes (LP-9 void area $V(t)$)", y=0.98, fontsize=12)

    fig.subplots_adjust(left=0.14, right=0.88, top=0.84, bottom=0.16, wspace=0.28)
    cax = fig.add_axes([0.90, 0.20, 0.02, 0.58])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Scaled trajectory sensitivity", rotation=90, labelpad=16)

    fig.savefig(os.path.join(outdir, "pc_response_heatmaps_V.pdf"), bbox_inches="tight")
    plt.close(fig)


# PRCC ANALYSIS

def run_prcc_analysis(p_best, outdir, cell_line):
    print(f"\nRunning PRCC analysis for {cell_line}...")
    rng = np.random.default_rng(SEED)

    for mk, info in metric_info.items():
        auc_series = {}
        fin_series = {}

        for cond in cond_list:
            active_params = get_active_params_for_condition(cond, p_best)
            lo, hi = make_bounds(p_best, active_params, frac=RANGE_FRAC)
            U01 = lhs(N_SAMPLES, len(active_params), rng)
            X = lo + U01 * (hi - lo)

            y_auc = np.zeros(N_SAMPLES, dtype=float)
            y_fin = np.zeros(N_SAMPLES, dtype=float)

            for i in range(N_SAMPLES):
                p = dict(p_best)
                for j, k in enumerate(active_params):
                    p[k] = float(X[i, j])

                y = simulate_single_condition(p, cond, mk)
                y_auc[i] = auc_trapz(t_fine, y)
                y_fin[i] = y[-1]

            auc_series[cond] = prcc(X, y_auc, active_params, f"AUC_{info['short']}|{cond}")
            fin_series[cond] = prcc(X, y_fin, active_params, f"{info['short']}_final|{cond}")

        df_auc = pd.concat(auc_series.values(), axis=1).fillna(0.0)
        df_fin = pd.concat(fin_series.values(), axis=1).fillna(0.0)

        df_auc.to_csv(os.path.join(outdir, f"prcc_{mk}_AUC.csv"))
        df_fin.to_csv(os.path.join(outdir, f"prcc_{mk}_final.csv"))

        prcc_bar_panels(
            df_auc,
            title=f"{cell_line}: PRCC on AUC of {info['title']}",
            ylabel="PRCC",
            outfile=f"prcc_{mk}_AUC.pdf",
            outdir=outdir,
        )
        prcc_bar_panels(
            df_fin,
            title=f"{cell_line}: PRCC on final value of {info['title']}",
            ylabel="PRCC",
            outfile=f"prcc_{mk}_final.pdf",
            outdir=outdir,
        )


# SVD ANALYSIS

def run_svd_analysis(p_best, outdir, cell_line):
    print(f"\nRunning SVD analysis for {cell_line}...")

    v_pc_maps_dict = None

    for mk, info in metric_info.items():
        svals_dict = {}
        fd_rows_all = []
        pc_maps_dict = {}

        for cond in cond_list:
            active_params = get_active_params_for_condition(cond, p_best)
            y_base = simulate_single_condition(p_best, cond, mk)

            M = np.zeros((len(t_fine), len(active_params)), dtype=float)

            for j, k in enumerate(active_params):
                v = float(p_best[k])
                h = get_fd_step(k, v)

                p_plus = perturb_param(p_best, k, +h)
                p_minus = perturb_param(p_best, k, -h)

                denom = p_plus[k] - p_minus[k]
                if abs(denom) < 1e-14:
                    yp = simulate_single_condition(p_plus, cond, mk)
                    dYdp = (yp - y_base) / max(p_plus[k] - p_best[k], 1e-14)
                else:
                    yp = simulate_single_condition(p_plus, cond, mk)
                    ym = simulate_single_condition(p_minus, cond, mk)
                    dYdp = (yp - ym) / denom

                pscale = max(abs(v), 1.0)
                M[:, j] = dYdp * pscale

                fd_rows_all.append({
                    "cell_line": cell_line,
                    "metric": mk,
                    "condition": cond,
                    "parameter": k,
                    "best_value": v,
                    "step_used": h,
                    "plus_value": p_plus[k],
                    "minus_value": p_minus[k]
                })

            U, svals, Vt = svd(M, full_matrices=False)
            svals_dict[cond] = svals

            s_norm = svals / max(svals[0], 1e-16)
            n_keep = int(np.sum(s_norm >= PC_KEEP_THRESHOLD))
            n_keep = max(1, min(n_keep, MAX_PCS_TO_PLOT))

            pc_maps_dict[cond] = [svals[i] * U[:, i] for i in range(n_keep)]

        pd.DataFrame(fd_rows_all).to_csv(os.path.join(outdir, f"fd_steps_{mk}.csv"), index=False)

        singular_spectrum_plot_by_condition(
            svals_dict=svals_dict,
            threshold=PC_KEEP_THRESHOLD,
            title=f"{cell_line}: normalized singular spectrum ({info['title']})",
            outfile=f"singular_spectrum_{mk}.pdf",
            outdir=outdir,
        )

        if mk == "V":
            v_pc_maps_dict = pc_maps_dict

    # only the new invasion-focused heatmap for V(t)
    if v_pc_maps_dict is not None:
        response_heatmap_V_only(v_pc_maps_dict, cell_line, outdir, max_pcs=MAX_PCS_TO_PLOT)


# MAIN

if __name__ == "__main__":
    for cell_line, p_best in BEST_FITS.items():
        outdir = f"sensitivity_{cell_line}"
        os.makedirs(outdir, exist_ok=True)

        print("\n" + "=" * 70)
        print(f"Running analyses for {cell_line}")
        print(f"Output folder: {outdir}")
        print("=" * 70)

        run_prcc_analysis(p_best, outdir, cell_line)
        run_svd_analysis(p_best, outdir, cell_line)

    print("\nDone. Outputs saved in:")
    print("  sensitivity_OVCAR3/")
    print("  sensitivity_OVCAR4/")
