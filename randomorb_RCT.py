"""
RCT Regression — 3D Biplot with Randomization Surface
======================================================
A BIPLOT puts both VARIABLES and STUDENTS in the same PCA space:

  Data matrix X: 10 students × 5 variables  [X1, X2, X3, Y, Ŷ]
  SVD of X: X = U D V^T

  Variable arrows → rows of V^T (one arrow per variable)
    - Arrow direction = variable's dominant pattern across students
    - Angle between arrows ≈ arccos(correlation)
    - Orbs placed at arrow tips to show variable "reach"

  Student points → rows of U * D (one point per student)
    - Students near a variable arrow = high value on that variable
    - Teal dots = Treatment arm, coral = Control arm

  Randomization surface → separates T and C students in this space.
    With matched-pair randomization, the surface cuts covariate
    arrows (~50/50) because treatment is uncorrelated with covariates.

RANDOMIZATION METHOD: Matched-pair (gold standard for n=10)
  - Pair students by similarity on pre-treatment covariates (GPA, income)
  - Within each pair, flip a coin for T vs C
  - Guarantees exact 5T/5C and near-zero covariate–treatment correlation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(7)

# ── 1. PRE-TREATMENT DATA ─────────────────────────────────────────────────────
GPA             = np.array([3.8, 3.5, 3.9, 3.2, 2.8, 2.9, 3.5, 3.7, 2.5, 3.1])
parental_income = np.array([95,  60, 110,  45,  40,  55,  80,  75,  35,  50], dtype=float)
n = len(GPA)

# ── 2. MATCHED-PAIR RANDOMIZATION ────────────────────────────────────────────
cov_std = np.column_stack([
    (GPA            - GPA.mean())            / GPA.std(),
    (parental_income - parental_income.mean()) / parental_income.std()
])
_, _, Vt_cov = np.linalg.svd(cov_std, full_matrices=False)
pc1_score   = cov_std @ Vt_cov[0]
sorted_idx  = np.argsort(pc1_score)
pairs       = [(sorted_idx[i], sorted_idx[i+1]) for i in range(0, n, 2)]

treatment = np.zeros(n, dtype=float)
for a, b in pairs:
    if np.random.random() < 0.5:
        treatment[a] = 1
    else:
        treatment[b] = 1

def r(a, b): return np.corrcoef(a, b)[0, 1]

print("── Matched-Pair Assignment ────────────────────────")
df = pd.DataFrame({
    'Student': [f'S{i+1}' for i in range(n)],
    'GPA': GPA, 'Income($k)': parental_income.astype(int),
    'Arm': ['T' if t else 'C' for t in treatment]
})
print(df.to_string(index=False))
print(f"\n  r(treatment,GPA)    = {r(treatment,GPA):.3f}  <- ~0 = balanced")
print(f"  r(treatment,income) = {r(treatment,parental_income):.3f}  <- ~0 = balanced")

# ── 3. OUTCOME & OLS ─────────────────────────────────────────────────────────
job_score = (-52.4 + 24.8*treatment + 29.1*GPA
             + 0.08*parental_income + np.random.normal(0, 15, n))

X_mat   = np.column_stack([np.ones(n), treatment, GPA, parental_income])
b_hat   = np.linalg.lstsq(X_mat, job_score, rcond=None)[0]
Y_hat   = X_mat @ b_hat
R2      = 1 - np.sum((job_score-Y_hat)**2) / np.sum((job_score-job_score.mean())**2)

print(f"\n── OLS  b1(treatment)={b_hat[1]:.1f}  b2(GPA)={b_hat[2]:.1f}  "
      f"b3(income)={b_hat[3]:.2f}  R²={R2:.3f}")

# ── 4. BIPLOT SVD ─────────────────────────────────────────────────────────────
# Stack 5 variables as columns: [treatment, GPA, income, job_score, Y_hat]
# Standardise each column so all variables are on equal footing.
# SVD gives:
#   U  (10×5): student coordinates (rows = students)
#   S  (5,)  : singular values
#   Vt (5×5) : variable directions (rows = variables)
X_data = np.column_stack([treatment, GPA, parental_income, job_score, Y_hat])
X_std  = (X_data - X_data.mean(axis=0)) / X_data.std(axis=0)

U_b, S_b, Vt_b = np.linalg.svd(X_std, full_matrices=False)

var_exp = (S_b**2) / np.sum(S_b**2)
print(f"\n── Biplot PCA quality ─────────────────────────────")
print(f"  3 PCs capture {sum(var_exp[:3]):.1%} of total structure")

# Variable arrows: rows of Vt_b scaled by S so length ∝ variance explained
# Shape (5, 3) — one row per variable [X1, X2, X3, Y, Ŷ]
L = 3.5
var_raw  = Vt_b[:3, :].T                           # (5, 3)
var_norm = max(np.linalg.norm(v) for v in var_raw)
var_pts  = {name: L * var_raw[i] / var_norm
            for i, name in enumerate(['X1','X2','X3','Y','Yh'])}

# Student points: rows of U * D, then scale to fit inside orb cluster
stu_raw  = U_b[:, :3] * S_b[:3]                    # (10, 3)
stu_norm = max(np.linalg.norm(s) for s in stu_raw)
stu_pts  = stu_raw / stu_norm * L * 0.6            # sit inside orbs

p1  = var_pts['X1'];  p2  = var_pts['X2'];  p3  = var_pts['X3']
pY  = var_pts['Y'];   pYh = var_pts['Yh']
R_orb = 1.8   # orb radius

# ── 5. VISUALISE ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 10), facecolor='#0e0e1a')
ax  = fig.add_subplot(111, projection='3d', facecolor='#0e0e1a')

def arrow(tip, color, lw=2.5, ls='-'):
    ax.quiver(0,0,0, tip[0],tip[1],tip[2], color=color,
              linewidth=lw, linestyle=ls, arrow_length_ratio=0.10)

def orb(centre, color, alpha=0.13):
    u, w = np.mgrid[0:2*np.pi:40j, 0:np.pi:30j]
    ax.plot_surface(centre[0]+R_orb*np.cos(u)*np.sin(w),
                    centre[1]+R_orb*np.sin(u)*np.sin(w),
                    centre[2]+R_orb*np.cos(w),
                    color=color, alpha=alpha, linewidth=0)

# ── STUDENT DOTS ─────────────────────────────────────────────────────────────
# Each dot = one student in the biplot space.
# Position reflects that student's pattern across all 5 variables.
# Teal = Treatment, coral = Control.
z_floor = stu_pts[:, 2].min() - 0.4
for i, (pt, t) in enumerate(zip(stu_pts, treatment)):
    col = '#80cbc4' if t else '#ef9a9a'
    ax.scatter(*pt, color=col, s=90, zorder=6,
               edgecolors='white', linewidths=0.6)
    ax.text(pt[0]+0.1, pt[1]+0.1, pt[2]+0.1, f'S{i+1}',
            color=col, fontsize=7.5, fontfamily='monospace', alpha=0.9)
    ax.plot([pt[0],pt[0]], [pt[1],pt[1]], [pt[2], z_floor],
            color=col, linewidth=0.35, alpha=0.25)

# ── RANDOMIZATION SURFACE ────────────────────────────────────────────────────
# Perpendicular bisector between T-centroid and C-centroid in biplot space.
# This is the natural Voronoi boundary between the two arms.
t_cen = stu_pts[treatment==1].mean(axis=0)
c_cen = stu_pts[treatment==0].mean(axis=0)
mid   = (t_cen + c_cen) / 2                   # plane passes through midpoint
norm  = t_cen - c_cen                         # normal points T → C direction
norm  = norm / np.linalg.norm(norm)

arb = np.array([0,1,0]) if abs(norm[1]) < 0.9 else np.array([1,0,0])
f1  = arb - np.dot(arb, norm)*norm;  f1 /= np.linalg.norm(f1)
f2  = np.cross(norm, f1);            f2 /= np.linalg.norm(f2)

disc_r    = R_orb * 2.8
rhos, phis = np.linspace(0, disc_r, 35), np.linspace(0, 2*np.pi, 70)
RHO, PHI  = np.meshgrid(rhos, phis)
warp      = 0.25 * np.cos(2*PHI) * (RHO/disc_r)  # gentle curvature
SX = mid[0] + RHO*(np.cos(PHI)*f1[0]+np.sin(PHI)*f2[0]) + warp*norm[0]
SY = mid[1] + RHO*(np.cos(PHI)*f1[1]+np.sin(PHI)*f2[1]) + warp*norm[1]
SZ = mid[2] + RHO*(np.cos(PHI)*f1[2]+np.sin(PHI)*f2[2]) + warp*norm[2]
ax.plot_surface(SX, SY, SZ, alpha=0.18, color='#b0bec5', linewidth=0)

phi_r  = np.linspace(0, 2*np.pi, 150)
ring   = (mid + disc_r*(np.outer(np.cos(phi_r),f1)+np.outer(np.sin(phi_r),f2))
          + np.outer(0.25*np.cos(2*phi_r), norm))
ax.plot(ring[:,0], ring[:,1], ring[:,2], color='#90a4ae', lw=1.5, alpha=0.8)

ax.text(*(mid + norm*R_orb*1.3 + f1*disc_r*0.7),
        'TREATMENT →', color='#80cbc4', fontsize=8, fontfamily='monospace')
ax.text(*(mid - norm*R_orb*1.3 + f1*disc_r*0.7),
        '← CONTROL',   color='#ef9a9a', fontsize=8, fontfamily='monospace')

# ── VARIABLE ORBS & ARROWS ───────────────────────────────────────────────────
orb(pY,  '#fff176', alpha=0.11)
orb(pYh, '#ffb74d', alpha=0.11)
orb(p3,  '#ce93d8', alpha=0.13)
orb(p2,  '#81c784', alpha=0.13)
orb(p1,  '#4fc3f7', alpha=0.13)

arrow(p1,  '#4fc3f7', lw=3)
arrow(p2,  '#81c784', lw=3)
arrow(p3,  '#ce93d8', lw=3)
arrow(pY,  '#fff176', lw=3)
arrow(pYh, '#ffb74d', lw=2, ls='--')

res = pY - pYh
ax.quiver(*pYh, *res, color='#ef5350', lw=1.8, linestyle=':', arrow_length_ratio=0.2)

# ── LABELS ───────────────────────────────────────────────────────────────────
off = 0.28
ax.text(*(p1+off),  'X₁  Treatment\n    (RCT)',  color='#4fc3f7', fontsize=10,
        fontweight='bold', fontfamily='monospace')
ax.text(*(p2+off),  'X₂  GPA',                  color='#81c784', fontsize=10,
        fontweight='bold', fontfamily='monospace')
ax.text(*(p3+off),  'X₃  Parental $',            color='#ce93d8', fontsize=10,
        fontweight='bold', fontfamily='monospace')
ax.text(*(pY+off),  'Y   Job Score',              color='#fff176', fontsize=10,
        fontweight='bold', fontfamily='monospace')
ax.text(*(pYh - np.array([0,0,0.5])),
        'Ŷ (fitted)', color='#ffb74d', fontsize=9, fontfamily='monospace')
ax.text(*(pYh + res*0.6 + np.array([0.1,0,0])),
        'e', color='#ef5350', fontsize=9, fontfamily='monospace')

# ── STYLING ──────────────────────────────────────────────────────────────────
ax.set_xlabel('PC 1', color='#555', labelpad=8)
ax.set_ylabel('PC 2', color='#555', labelpad=8)
ax.set_zlabel('PC 3', color='#555', labelpad=8)
ax.tick_params(colors='#444')
for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
    pane.fill = False; pane.set_edgecolor('#1a1a33')
ax.grid(True, color='#1a1a33', linewidth=0.5)
ax.set_title(
    f'RCT Biplot  (R²={R2:.3f},  3D captures {sum(var_exp[:3]):.0%})\n'
    f'Teal = Treatment · Coral = Control · Surface = randomization boundary',
    color='white', fontsize=11, pad=12)

info = (f"r(X₁,X₂)={r(treatment,GPA):.3f}  r(X₁,X₃)={r(treatment,parental_income):.3f}  "
        f"r(X₂,X₃)={r(GPA,parental_income):.3f}  r(Y,X₁)={r(job_score,treatment):.3f}\n"
        f"Surface = perpendicular bisector of T/C centroids in biplot space")
fig.text(0.5, 0.01, info, ha='center', color='#aaaacc', fontsize=8,
         fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='#111133', alpha=0.7, edgecolor='#333366'))

plt.tight_layout()
plt.savefig('regression_vectors.png', dpi=150, bbox_inches='tight', facecolor='#0e0e1a')
plt.show()
print("\nSaved → regression_vectors.png")