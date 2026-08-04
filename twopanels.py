"""
RCT Regression — Two-Panel Teaching Visualisation
==================================================

PANEL 1 — Regression geometry (vector / orb space)
  Five variables shown as arrows + orbs in a shared PCA space.
  Angle between arrows ≈ correlation between variables.
  Length ≈ variance explained.
  No students here — this is pure variable geometry.

PANEL 2 — Randomization (actual covariate space)
  10 students plotted by their real GPA and parental income.
  Matched pairs connected by a bracket line.
  Teal dot = got internship (Treatment), Coral = Control.
  The dashed boundary shows the matched-pair split along PC1.

RANDOMIZATION METHOD: Matched-pair (gold standard for n=10)
  Sort students by PC1 of standardised covariates.
  Pair consecutive students (most similar profiles).
  Coin-flip within each pair → exactly 5T / 5C.
  Guarantees r(internship, GPA) ≈ 0  and  r(internship, income) ≈ 0.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(7)

# ── 1. DATA ───────────────────────────────────────────────────────────────────
GPA             = np.array([3.8, 3.5, 3.9, 3.2, 2.8, 2.9, 3.5, 3.7, 2.5, 3.1])
parental_income = np.array([95,  60, 110,  45,  40,  55,  80,  75,  35,  50], dtype=float)
n = len(GPA)

# ── 2. MATCHED-PAIR RANDOMIZATION ────────────────────────────────────────────
cov_std = np.column_stack([
    (GPA             - GPA.mean())             / GPA.std(),
    (parental_income - parental_income.mean()) / parental_income.std()
])
_, _, Vt_cov = np.linalg.svd(cov_std, full_matrices=False)
pc1_score  = cov_std @ Vt_cov[0]
sorted_idx = np.argsort(pc1_score)
pairs      = [(sorted_idx[i], sorted_idx[i+1]) for i in range(0, n, 2)]

treatment = np.zeros(n, dtype=float)
for a, b in pairs:
    if np.random.random() < 0.5:
        treatment[a] = 1
    else:
        treatment[b] = 1

def r(a, b): return np.corrcoef(a, b)[0, 1]

# ── 3. OUTCOME & OLS ─────────────────────────────────────────────────────────
job_score = (-52.4 + 24.8*treatment + 29.1*GPA
             + 0.08*parental_income + np.random.normal(0, 15, n))
X_mat = np.column_stack([np.ones(n), treatment, GPA, parental_income])
b_hat = np.linalg.lstsq(X_mat, job_score, rcond=None)[0]
Y_hat = X_mat @ b_hat
R2    = 1 - np.sum((job_score-Y_hat)**2) / np.sum((job_score-job_score.mean())**2)

# ── 4. BIPLOT SVD for PANEL 1 (variable geometry only) ────────────────────────
X_data = np.column_stack([treatment, GPA, parental_income, job_score, Y_hat])
X_std  = (X_data - X_data.mean(axis=0)) / X_data.std(axis=0)
U_b, S_b, Vt_b = np.linalg.svd(X_std, full_matrices=False)
var_exp = (S_b**2) / np.sum(S_b**2)

L = 3.5
var_raw  = Vt_b[:3, :].T
var_norm = max(np.linalg.norm(v) for v in var_raw)
var_pts  = {name: L * var_raw[i] / var_norm
            for i, name in enumerate(['X1','X2','X3','Y','Yh'])}
p1 = var_pts['X1']; p2 = var_pts['X2']; p3 = var_pts['X3']
pY = var_pts['Y'];  pYh = var_pts['Yh']
R_orb = 1.8

# ── 5. FIGURE ─────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 8), facecolor='#0e0e1a')
ax1 = fig.add_subplot(121, projection='3d', facecolor='#0e0e1a')
ax2 = fig.add_subplot(122, facecolor='#0e0e1a')

# ══════════════════════════════════════════════════════════════════════════════
# PANEL 1 — Variable orbs
# ══════════════════════════════════════════════════════════════════════════════
def arrow(ax, tip, color, lw=2.5, ls='-'):
    ax.quiver(0,0,0, tip[0],tip[1],tip[2], color=color,
              linewidth=lw, linestyle=ls, arrow_length_ratio=0.12)

def orb(ax, centre, color, alpha=0.13):
    u, w = np.mgrid[0:2*np.pi:40j, 0:np.pi:30j]
    ax.plot_surface(centre[0]+R_orb*np.cos(u)*np.sin(w),
                    centre[1]+R_orb*np.sin(u)*np.sin(w),
                    centre[2]+R_orb*np.cos(w),
                    color=color, alpha=alpha, linewidth=0)

orb(ax1, p1,  '#4fc3f7', alpha=0.13)   # X1 — internship
orb(ax1, p2,  '#81c784', alpha=0.13)   # X2 — GPA
orb(ax1, p3,  '#ce93d8', alpha=0.13)   # X3 — parental income
orb(ax1, pY,  '#fff176', alpha=0.11)   # Y  — job score
orb(ax1, pYh, '#ffb74d', alpha=0.11)   # Ŷ  — fitted values

arrow(ax1, p1,  '#4fc3f7', lw=3)
arrow(ax1, p2,  '#81c784', lw=3)
arrow(ax1, p3,  '#ce93d8', lw=3)
arrow(ax1, pY,  '#fff176', lw=3)
arrow(ax1, pYh, '#ffb74d', lw=2, ls='--')

res = pY - pYh
ax1.quiver(*pYh, *res, color='#ef5350', lw=1.8, linestyle=':', arrow_length_ratio=0.2)

off = 0.3
ax1.text(*(p1+off), 'X₁  Internship',  color='#4fc3f7', fontsize=9, fontweight='bold', fontfamily='monospace')
ax1.text(*(p2+off), 'X₂  GPA',         color='#81c784', fontsize=9, fontweight='bold', fontfamily='monospace')
ax1.text(*(p3+off), 'X₃  Parental $',  color='#ce93d8', fontsize=9, fontweight='bold', fontfamily='monospace')
ax1.text(*(pY+off), 'Y   Job Score',   color='#fff176', fontsize=9, fontweight='bold', fontfamily='monospace')
ax1.text(*(pYh - np.array([0,0,0.6])), 'Ŷ (fitted)',  color='#ffb74d', fontsize=8, fontfamily='monospace')
ax1.text(*(pYh + res*0.55 + np.array([0.1,0,0])), 'e', color='#ef5350', fontsize=9, fontfamily='monospace')

# Annotation: angle ≈ correlation
ang12 = np.degrees(np.arccos(np.clip(r(treatment, GPA), -1, 1)))
ax1.text(-L*0.6, -L*0.5, -L*0.4,
         f'∠(X₁,X₂) = {ang12:.0f}°\n≈ arccos(r={r(treatment,GPA):.2f})\n→ near-orthogonal\n   = balanced RCT',
         color='#78909c', fontsize=7.5, fontfamily='monospace', alpha=0.9)

ax1.set_xlabel('PC 1', color='#444', labelpad=6)
ax1.set_ylabel('PC 2', color='#444', labelpad=6)
ax1.set_zlabel('PC 3', color='#444', labelpad=6)
ax1.tick_params(colors='#444')
for pane in [ax1.xaxis.pane, ax1.yaxis.pane, ax1.zaxis.pane]:
    pane.fill = False; pane.set_edgecolor('#1a1a33')
ax1.grid(True, color='#1a1a33', linewidth=0.5)
ax1.set_title(f'Regression geometry\n5 variables as vectors  (3 PCs = {sum(var_exp[:3]):.0%},  R²={R2:.3f})',
              color='white', fontsize=10, pad=10)

# ══════════════════════════════════════════════════════════════════════════════
# PANEL 2 — Students in covariate space with T/C assignment
# ══════════════════════════════════════════════════════════════════════════════
# Axes: GPA (x) vs parental income (y)
# Matched pairs connected by a line; T gets a filled dot, C an open dot.
# The PC1 boundary line shows the matched-pair sort axis.

T_COL = '#80cbc4'   # teal — treatment
C_COL = '#ef9a9a'   # coral — control

for pair_idx, (a, b) in enumerate(pairs):
    # Connect the matched pair with a thin bracket line
    ax2.plot([GPA[a], GPA[b]], [parental_income[a], parental_income[b]],
             color='#555577', lw=1.2, zorder=1, alpha=0.7)

for i in range(n):
    col  = T_COL if treatment[i] else C_COL
    label_arm = 'Internship (T)' if treatment[i] else 'Control (C)'
    ax2.scatter(GPA[i], parental_income[i],
                color=col, s=130, zorder=5,
                edgecolors='white', linewidths=0.8)
    ax2.text(GPA[i]+0.03, parental_income[i]+1.5, f'S{i+1}',
             color=col, fontsize=8, fontfamily='monospace', fontweight='bold')

# PC1 direction arrow — this is the sorting axis for matched pairing
pc1_dir = Vt_cov[0]        # unit vector in [GPA_std, income_std] space
# Convert back to data units for display
gpa_scale    = GPA.std()
inc_scale    = parental_income.std()
gpa_mid      = GPA.mean()
inc_mid      = parental_income.mean()
arrow_len_g  = pc1_dir[0] * gpa_scale * 0.6
arrow_len_i  = pc1_dir[1] * inc_scale * 0.6
ax2.annotate('', xy=(gpa_mid + arrow_len_g, inc_mid + arrow_len_i),
             xytext=(gpa_mid - arrow_len_g, inc_mid - arrow_len_i),
             arrowprops=dict(arrowstyle='->', color='#90a4ae', lw=2))
ax2.text(gpa_mid + arrow_len_g + 0.03, inc_mid + arrow_len_i + 1,
         'PC1\n(sort axis\nfor pairing)',
         color='#90a4ae', fontsize=7.5, fontfamily='monospace')

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0],[0], marker='o', color='w', markerfacecolor=T_COL,
           markersize=10, label='Internship (Treatment)'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor=C_COL,
           markersize=10, label='Control'),
    Line2D([0],[0], color='#555577', lw=1.5, label='Matched pair'),
]
ax2.legend(handles=legend_elements, loc='upper left',
           facecolor='#1a1a2e', edgecolor='#333355',
           labelcolor='white', fontsize=8)

ax2.set_xlabel('GPA', color='white', fontsize=10)
ax2.set_ylabel('Parental Income ($k)', color='white', fontsize=10)
ax2.tick_params(colors='#888')
ax2.spines[['top','right','bottom','left']].set_color('#333355')
ax2.set_facecolor('#0e0e1a')
ax2.set_title(
    f'Randomization: matched-pair assignment\n'
    f'r(X₁,GPA)={r(treatment,GPA):.3f}  r(X₁,income)={r(treatment,parental_income):.3f}  '
    f'(~0 = balanced)',
    color='white', fontsize=10, pad=10)

# ── OVERALL TITLE ─────────────────────────────────────────────────────────────
fig.suptitle('RCT Regression  ·  y = b₀ + b₁·internship + b₂·GPA + b₃·parental income',
             color='white', fontsize=12, y=1.01)

plt.tight_layout()
plt.savefig('regression_vectors.png', dpi=150, bbox_inches='tight', facecolor='#0e0e1a')
plt.show()
print("Saved → regression_vectors.png")