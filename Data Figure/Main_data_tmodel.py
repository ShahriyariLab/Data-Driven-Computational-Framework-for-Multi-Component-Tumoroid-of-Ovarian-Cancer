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
    "lines.linewidth": 1.3,
    "legend.frameon": False,
    "legend.fontsize": 12,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})



SUBSTEPS_FIT = 100
SUBSTEPS_PLOT = 100



# DATA

t_data = np.array([0.0, 1.0, 2.0, 5.0], dtype=float)

conditions = {
    "Tumor":        (1.0, 0.0, 0.0),
    "Tumor+MSC":    (1.0, 1.0, 0.0),
    "Tumor+M2":     (1.0, 0.0, 1.0),
    "Tumor+MSC+M2": (1.0, 1.0, 1.0),
}

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
        "Tumor+M2":     np.array([0.0, 0.25, 0.40, 1.0]),
        "Tumor+MSC+M2": np.array([0.0, 0.30, 0.45, 1.38]),
    },
}


param_names_18 = [
    "KT", "KS", "KM", "Vmax", "rho",
    "rS", "rM", "dS", "dM",
    "etaTS", "etaST", "etaMT",
    "rT", "dT", "beta", "aTS", "aTM", "etaTM"
]


# PLOT DATA POINTS

def plot_data_points(cell_line):
    fig, ax = plt.subplots(figsize=(7.4, 4.7))

    for condition in conditions:
        ax.scatter(
            t_data,
            data[cell_line][condition],
            s=50,
            marker="o",
            label=condition
        )

    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Normalized LP-9 Void Area")
    ax.set_title(f"LP-9 Void Normalized to {cell_line} Spheroid Area")
    ax.legend(frameon=False)

    plt.tight_layout()
    return fig


# FIGURES

fig = plot_data_points("OVCAR3")
fig.savefig("OVCAR3_data_points_only.pdf", bbox_inches="tight")
plt.close(fig)

fig = plot_data_points("OVCAR4")
fig.savefig("OVCAR4_data_points_only.pdf", bbox_inches="tight")
plt.close(fig)

print("Saved:")
print("OVCAR3_data_points_only.pdf")
print("OVCAR4_data_points_only.pdf")
