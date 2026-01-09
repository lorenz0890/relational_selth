import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from utils.functions import gamma_rgnn_lower_bound, rgnn_madds_estimate

# ---- Global style tweaks: bigger fonts/ticks everywhere ----
plt.rcParams.update({
    "font.size": 14,          # base font
    "axes.titlesize": 18,     # subplot titles
    "axes.labelsize": 16,     # axis labels
    "xtick.labelsize": 13,    # tick labels
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
})
s_mid = 2

# ----------------- Fallbacks for required constants (keeps your session if already defined) -----------------

L_fixed = 5
B_default_r = 4
M_r = 2
s_min_r = 2
s_min_comb_r = 2
N_max_r = 1000
N_max_comb_r = N_max_r

# ----------------- Shared settings / grids -----------------
rho_mid = 0.7
widths_shared = np.linspace(1, 64, 320)                 # <-- shared y-axis across all panels

# ----- Panel A data: x = N_max (log), y = width -----
N_vals_A = np.logspace(2.0001, 5.999, 320)                       # 1e2 ... 1e6
gamma_grid_A = np.zeros((len(N_vals_A), len(widths_shared)), dtype=float)
cost_grid_A  = np.zeros_like(gamma_grid_A, dtype=float)

for i, N_curr in enumerate(N_vals_A):
    for j, m_min in enumerate(widths_shared):
        gamma_grid_A[i, j] = gamma_rgnn_lower_bound(
            N_max=N_curr, s_min=s_min_r, m_min=m_min,
            N_max_comb=N_curr, s_min_comb=s_min_comb_r, m_min_comb=m_min,
            rho=rho_mid, L=L_fixed, M=M_r, B=B_default_r, use_simplified=False
        )
        cost_grid_A[i, j] = rgnn_madds_estimate(
            N_max=N_curr, N_max_comb=N_curr,
            m_min=m_min, m_min_comb=m_min,
            B=B_default_r, M=M_r, L=L_fixed, sparsity=rho_mid
        )

eps = 1e-12
eff_per_M_A = 1e6 * gamma_grid_A / (cost_grid_A + eps)
eff_pos_A = eff_per_M_A[eff_per_M_A > 0]
if eff_pos_A.size >= 8 and np.isfinite(eff_pos_A).all():
    levels_eff_A = np.geomspace(np.percentile(eff_pos_A, 20),
                                np.percentile(eff_pos_A, 95), 7)
else:
    levels_eff_A = np.geomspace(1e-6, 1e-2, 7)

# ----- Panel D data: x = s_min, y = width -----
s_vals_D = np.linspace(1.001, 3.999, 320)
gamma_grid_D = np.zeros((len(s_vals_D), len(widths_shared)), dtype=float)
for i, s_val in enumerate(s_vals_D):
    for j, m_min in enumerate(widths_shared):
        gamma_grid_D[i, j] = gamma_rgnn_lower_bound(
            N_max=N_max_r, s_min=s_val, m_min=m_min,
            N_max_comb=N_max_r, s_min_comb=s_val, m_min_comb=m_min,
            rho=rho_mid, L=L_fixed, M=M_r, B=B_default_r, use_simplified=False
        )

# ----- Panel C data: x = rho, y = width -----
rhos_C = np.linspace(0.01, 0.99, 600)
madds_grid_C = np.zeros((len(widths_shared), len(rhos_C)), dtype=float)
gamma_grid_C = np.zeros_like(madds_grid_C, dtype=float)
for j, rho in enumerate(rhos_C):
    for i, m_min in enumerate(widths_shared):
        gamma_grid_C[i, j] = gamma_rgnn_lower_bound(
            N_max=N_max_r, s_min=s_min_r, m_min=m_min,
            N_max_comb=N_max_comb_r, s_min_comb=s_min_comb_r, m_min_comb=m_min,
            rho=rho, L=L_fixed, M=M_r, B=B_default_r, use_simplified=False
        )
        madds_grid_C[i, j] = rgnn_madds_estimate(
            N_max=N_max_r, N_max_comb=N_max_comb_r,
            m_min=m_min, m_min_comb=m_min,
            B=B_default_r, M=M_r, L=L_fixed, sparsity=rho
        )

# ----- Panel B data: γ=0.99 frontiers only, x = rho, y = width -----
rhos_B   = np.linspace(0.01, 0.99, 160)
N_values_B = [100, 1000, 10000, 100000, 1000000, 10000000]
B_values_B = [2, 8, 32]
colors_N_B = {
    100: "tab:blue", 1000: "tab:orange", 10000: "tab:purple",
    100000: "tab:green", 1000000: "tab:red", 10000000: "tab:olive",
}
styles_B_B = {2: "-", 8: "--", 32: ":"}
gamma_target_B = 0.99

# ----------------- Plot (A, D, C, B) with shared y -----------------
fig, axes = plt.subplots(1, 4, figsize=(17, 4.5*0.6), sharey=True)

# Panel A
axA = axes[0]
cfA = axA.contourf(
    N_vals_A, widths_shared, gamma_grid_A.T,
    levels=np.linspace(0, 1, 51), cmap=cm.Oranges, vmin=0.0, vmax=1.0
)
csA = axA.contour(
    N_vals_A, widths_shared, eff_per_M_A.T,
    levels=levels_eff_A, colors="black", linewidths=1.2
)
# horizontal labels for efficiency contours
textsA = axA.clabel(csA, fmt=lambda v: f"{v:.1e}", inline=False, fontsize=11)
for t in textsA:
    t.set_rotation(0)
    t.set_rotation_mode('anchor')

#axA.set_title(rf"γ-efficiency (ρ={rho_mid:.2f})")
axA.set_title("(a)")
axA.set_xlabel(r"$N_{\max}$", fontsize=17)
axA.set_ylabel(r"Width $m_{\min}$", fontsize=17)
axA.set_xscale("log")
axA.grid(True, ls=":", lw=0.8, alpha=0.7)

# Panel D
axD = axes[1]
cfD = axD.contourf(
    s_vals_D, widths_shared, gamma_grid_D.T,
    levels=np.linspace(0, 1, 51), cmap=cm.Oranges, vmin=0.0, vmax=1.0
)
csD = axD.contour(
    s_vals_D, widths_shared, gamma_grid_D.T,
    levels=[0.99], colors="black", linewidths=1.4
)
axD.clabel(csD, fmt={0.99: r"$\gamma=0.99$"}, inline=True, fontsize=11, manual=[[1.1, 2.3]])
'''
csD = axD.contour(
    s_vals_D, widths_shared, gamma_grid_D.T,
    levels=[0.5], colors="black", linewidths=1.4
)
axD.clabel(csD, fmt={0.5: r"$\gamma=0.5$"}, inline=True, fontsize=11, manual=[[1.1, 2.3]])
'''
csD = axD.contour(
    s_vals_D, widths_shared, gamma_grid_D.T,
    levels=[1e-9], colors="black", linewidths=1.4
)
axD.clabel(csD, fmt={1e-9: r"$\gamma=10^{-9}$"}, inline=True, fontsize=11, manual=[[1.1, 2.3]])

#axD.set_title(rf"γ landscape vs $s_{{\min}}$ (ρ={rho_mid:.2f})")
axD.set_title("(b)")
axD.set_xlabel(r"$s_{\min}$", fontsize=17)
axD.grid(True, ls=":", lw=0.8, alpha=0.7)

# Panel C
axC = axes[2]
cmapC = cm.RdYlGn_r
cfC = axC.contourf(
    rhos_C, widths_shared, madds_grid_C,
    levels=64, cmap=cmapC
)
levels_C = [0.99,
            #0.5,
            1e-9]
labels_C = {0.99: r"$\gamma=0.99$",
            #0.50: r"$\gamma=0.50$",
            1e-9: r"$\gamma=10^{-9}$"}
for lev in levels_C:
    cs = axC.contour(
        rhos_C, widths_shared, gamma_grid_C,
        levels=[lev], colors="black",
        linewidths=1.4 if lev == 0.99 else 1.2
    )
    axC.clabel(cs, fmt={lev: labels_C[lev]}, inline=True, fontsize=11, manual=[[1.1, 2.3]])

#axC.set_title(r"Cost landscape + γ frontiers")
axC.set_title("(c)")
axC.set_xlabel(r"Sparsity $\rho$", fontsize=17)
axC.grid(True, ls=":", lw=0.8, alpha=0.7)

# Panel B
axB = axes[3]
for N_max in N_values_B:
    for B_val in B_values_B:
        gamma_grid = np.zeros((len(widths_shared), len(rhos_B)), dtype=float)
        for i, m_min in enumerate(widths_shared):
            for j, rho in enumerate(rhos_B):
                gamma_grid[i, j] = gamma_rgnn_lower_bound(
                    N_max=N_max, s_min=s_mid, m_min=m_min,
                    N_max_comb=N_max, s_min_comb=s_mid, m_min_comb=m_min,
                    rho=rho, L=L_fixed, M=M_r, B=B_val, use_simplified=False
                )
        axB.contour(
            rhos_B, widths_shared, gamma_grid,
            levels=[gamma_target_B],
            colors=[colors_N_B[N_max]],
            linestyles=styles_B_B[B_val],
            linewidths=1.4,
        )
#axB.set_title(rf"γ = {gamma_target_B} frontiers (s$_{{\min}}$={s_min_r})")
axB.set_title("(d)")
axB.set_xlabel(r"Sparsity $\rho$", fontsize=17)
axB.grid(True, ls=":", lw=0.8, alpha=0.7)

# --- Legends inside Panel B (rightmost) ---
from matplotlib.lines import Line2D  # ensure this import is present at top
'''
legend_colors = [
    Line2D([0], [0], color=colors_N_B[N], lw=2, label=rf"$N_{{\max}}={N}$")
    for N in N_values_B
]
'''
legend_colors = [
    Line2D(
        [0], [0],
        color=colors_N_B[N], lw=2,
        label=rf"$N_{{\max}}=10^{{{int(np.round(np.log10(N)))}}}$"
    )
    for N in N_values_B
]

legend_styles = [
    Line2D([0], [0], color="black", lw=2, linestyle=styles_B_B[b], label=rf"$B={b}$")
    for b in B_values_B
]

leg1 = axB.legend(
    handles=legend_colors,
    title=r"Color: $N_{\max}$",
    loc="upper left",
    fontsize=9,
    frameon=True
)
axB.add_artist(leg1)

axB.legend(
    handles=legend_styles,
    title=r"Line: $B$",
    loc="lower right",
    fontsize=9,
    frameon=True
)

# Colorbars (left: γ for A/D; right: MADDS for C)
caxA = fig.add_axes([0.935, 0.57, 0.012, 0.33])
smA = cm.ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap=cm.Oranges)
smA.set_array([])
cbarA = fig.colorbar(smA, cax=caxA)
cbarA.set_label(r"$\gamma_{\mathrm{RGNN}}$", fontsize=12)

caxC = fig.add_axes([0.935, 0.13, 0.012, 0.33])
smC = cm.ScalarMappable(
    norm=Normalize(vmin=madds_grid_C.min(), vmax=madds_grid_C.max()),
    cmap=cmapC
)
smC.set_array([])
cbarC = fig.colorbar(smC, cax=caxC)
cbarC.set_label("MADDS (est.)", fontsize=12)

plt.subplots_adjust(left=0.055, right=0.92, wspace=0.07, bottom=0.12, top=0.92)
plt.savefig("composite_row_selected_panels_smin_shared_width_fixedshapes_swapped.pdf", bbox_inches="tight")
plt.show()
