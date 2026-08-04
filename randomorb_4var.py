"""
Regression Vector Geometry in 3D — with 4 variables
=====================================================
Each variable is a vector in R^10 (one dimension per observation).
We want to visualise Y, X1, X2, X3, and Ŷ as orbs in 3D.

THE KEY CONSTRAINT:
  3 vectors always fit exactly in 3D.
  4 or more vectors may need 4+ dimensions to preserve all pairwise angles.

So with 4 variables we use PCA to find the best 3D approximation:
  - Stack all centred, normalised variable vectors as rows of a matrix V (4 × 10)
  - The top 3 principal components of V give a 3D coordinate for each variable
    that preserves as much of the correlation structure as possible
  - Some information is lost (the 4th dimension), so angles are approximate
  - The % variance explained by the 3 PCs tells us how good the approximation is

Everything else (orbs, arrow lengths, overlap = correlation) is the same as before.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# ── 1. SYNTHETIC DATASET ─────────────────────────────────────────────────────
# We add parental_income as a 3rd predictor.
# It correlates positively with GPA and job_score (richer families → better
# schools → higher GPA; also some direct effect on job access).
# True DGP only used to generate numbers — OLS will estimate coefficients.

np.random.seed(7)

internship     = np.array([1, 1, 0, 1, 0, 1, 0, 1, 0, 0], dtype=float)
GPA            = np.array([3.8, 3.5, 3.9, 3.2, 2.8, 2.9, 3.5, 3.7, 2.5, 3.1])
parental_income = np.array([95, 60, 110, 45, 40, 55, 80, 75, 35, 50], dtype=float)  # $k
job_score      = (-52.4 + 24.8*internship + 29.1*GPA
                  + 0.08*parental_income   # small direct income effect
                  + np.random.normal(0, 15, 10))  # large noise → Y and Ŷ visibly differ

df = pd.DataFrame({
    'Internship':      internship,
    'GPA':             GPA,
    'ParentalIncome':  parental_income,
    'JobScore':        job_score.round(1)
})
print("── Synthetic Dataset ──────────────────────────────")
print(df.to_string(index=False))


# ── 2. FIT OLS ───────────────────────────────────────────────────────────────
X_mat = np.column_stack([np.ones(10), internship, GPA, parental_income])
b_hat = np.linalg.lstsq(X_mat, job_score, rcond=None)[0]
Y_hat_ols = X_mat @ b_hat

SS_res = np.sum((job_score - Y_hat_ols)**2)
SS_tot = np.sum((job_score - job_score.mean())**2)
R2 = 1 - SS_res / SS_tot

print("\n── OLS Estimates ──────────────────────────────────")
print(f"  b0 (intercept)       = {b_hat[0]:.2f}")
print(f"  b1 (internship)      = {b_hat[1]:.2f}")
print(f"  b2 (GPA)             = {b_hat[2]:.2f}")
print(f"  b3 (parental income) = {b_hat[3]:.2f}")
print(f"  R²                   = {R2:.3f}")


# ── 3. CENTER ALL VARIABLES ───────────────────────────────────────────────────
Y  = job_score       - job_score.mean()
X1 = internship      - internship.mean()
X2 = GPA             - GPA.mean()
X3 = parental_income - parental_income.mean()
Yh = Y_hat_ols       - Y_hat_ols.mean()   # centred fitted values


# ── 4. PAIRWISE CORRELATIONS ──────────────────────────────────────────────────
def r(a, b): return np.corrcoef(a, b)[0, 1]

print("\n── Correlations ───────────────────────────────────")
print(f"  r(X1,X2) = {r(X1,X2):.2f}  (internship ↔ GPA)")
print(f"  r(X1,X3) = {r(X1,X3):.2f}  (internship ↔ parental income)")
print(f"  r(X2,X3) = {r(X2,X3):.2f}  (GPA        ↔ parental income)")
print(f"  r(Y, X1) = {r(Y,X1):.2f}  (job score  ↔ internship)")
print(f"  r(Y, X2) = {r(Y,X2):.2f}  (job score  ↔ GPA)")
print(f"  r(Y, X3) = {r(Y,X3):.2f}  (job score  ↔ parental income)")
print(f"  r(Y, Yh) = {r(Y,Yh):.2f}  (job score  ↔ fitted values)")


# ── 5. PCA TO GET 3D COORDINATES FOR EACH VARIABLE ───────────────────────────
# Stack centred, unit-normalised variable vectors as rows: shape (5, 10)
# (Y, Ŷ, X1, X2, X3 — each is a length-10 observation vector)
# PCA finds the 3 directions in R^10 that capture the most variance across
# these 5 variable-vectors. Each variable's 3D coordinate is its projection
# onto these directions.
#
# Why does this preserve correlations approximately?
# Two variables that are highly correlated point in similar directions in R^10,
# so their PCA projections also point in similar directions in 3D.

def unit(v): return v / np.linalg.norm(v)

# Build the 5×10 matrix of unit variable vectors
V = np.vstack([unit(Y), unit(Yh), unit(X1), unit(X2), unit(X3)])  # (5, 10)

# SVD of V: V = U S W^T
# The rows of W^T (i.e. columns of W) are the principal directions in R^10.
# U gives the coordinates of each variable in PC space.
U, S, Wt = np.linalg.svd(V, full_matrices=False)

# Fraction of total "spread" captured by the top 3 PCs
var_explained = (S**2) / np.sum(S**2)
print(f"\n── PCA quality (3D approximation) ─────────────────")
print(f"  PC1 explains {var_explained[0]:.1%} of inter-variable structure")
print(f"  PC2 explains {var_explained[1]:.1%}")
print(f"  PC3 explains {var_explained[2]:.1%}")
print(f"  Total (3D)   {sum(var_explained[:3]):.1%}  — angles are this accurate")

# 3D coordinates: each row of U[:, :3] * S[:3] is one variable's position
coords = U[:, :3] * S[:3]   # shape (5, 3)
# Rows: [Y, Ŷ, X1, X2, X3]
c_Y, c_Yh, c_X1, c_X2, c_X3 = coords


# ── 6. VISUALISE ─────────────────────────────────────────────────────────────
# Normalise all coordinate vectors to the same display length L, then place
# an orb of radius R at each tip. Overlap ∝ correlation (approximately,
# with the accuracy given by the % variance explained above).

L = 3.5
R = 1.8

def disp(c): return L * c / np.linalg.norm(c)   # scale to display length

pY  = disp(c_Y)
pYh = disp(c_Yh)
p1  = disp(c_X1)
p2  = disp(c_X2)
p3  = disp(c_X3)

fig = plt.figure(figsize=(12, 9), facecolor='#0e0e1a')
ax  = fig.add_subplot(111, projection='3d', facecolor='#0e0e1a')


def arrow(tip, color, lw=2.5, ls='-'):
    ax.quiver(0, 0, 0, tip[0], tip[1], tip[2],
              color=color, linewidth=lw, linestyle=ls,
              arrow_length_ratio=0.10)


def orb(centre, color, alpha=0.13):
    u, w = np.mgrid[0:2*np.pi:40j, 0:np.pi:30j]
    xs = centre[0] + R * np.cos(u) * np.sin(w)
    ys = centre[1] + R * np.sin(u) * np.sin(w)
    zs = centre[2] + R * np.cos(w)
    ax.plot_surface(xs, ys, zs, color=color, alpha=alpha, linewidth=0)


# --- five orbs — one per variable --------------------------------------
# Draw the bigger / background ones first so foreground renders on top.
# Y and Ŷ nearly overlap (high R²); X2 and X3 overlap if GPA ↔ income corr is high.
orb(pY,  '#fff176', alpha=0.11)   # Y  job score       — yellow
orb(pYh, '#ffb74d', alpha=0.11)   # Ŷ  fitted values   — orange
orb(p3,  '#ce93d8', alpha=0.13)   # X3 parental income — purple
orb(p2,  '#81c784', alpha=0.13)   # X2 GPA             — green
orb(p1,  '#4fc3f7', alpha=0.13)   # X1 internship      — blue


# --- arrows -------------------------------------------------------------
arrow(p1,  '#4fc3f7', lw=3)
arrow(p2,  '#81c784', lw=3)
arrow(p3,  '#ce93d8', lw=3)
arrow(pY,  '#fff176', lw=3)
arrow(pYh, '#ffb74d', lw=2, ls='--')   # dashed = derived from data, not observed


# --- residual between Ŷ and Y ------------------------------------------
res_disp = pY - pYh
ax.quiver(pYh[0], pYh[1], pYh[2],
          res_disp[0], res_disp[1], res_disp[2],
          color='#ef5350', linewidth=1.8, linestyle=':',
          arrow_length_ratio=0.2)


# --- labels -------------------------------------------------------------
off = 0.28
ax.text(*(p1  + off), 'X₁  Internship',     color='#4fc3f7', fontsize=10,
        fontweight='bold', fontfamily='monospace')
ax.text(*(p2  + off), 'X₂  GPA',            color='#81c784', fontsize=10,
        fontweight='bold', fontfamily='monospace')
ax.text(*(p3  + off), 'X₃  Parental $',     color='#ce93d8', fontsize=10,
        fontweight='bold', fontfamily='monospace')
ax.text(*(pY  + off), 'Y   Job Score',       color='#fff176', fontsize=10,
        fontweight='bold', fontfamily='monospace')
ax.text(*(pYh - np.array([0, 0, 0.5])),
        'Ŷ  (fitted)', color='#ffb74d', fontsize=9, fontfamily='monospace')
ax.text(*(pYh + res_disp*0.6 + np.array([0.1, 0, 0])),
        'e', color='#ef5350', fontsize=9, fontfamily='monospace')


# --- axes & styling -----------------------------------------------------
ax.set_xlabel('PC 1', color='#555', labelpad=8)
ax.set_ylabel('PC 2', color='#555', labelpad=8)
ax.set_zlabel('PC 3', color='#555', labelpad=8)
ax.tick_params(colors='#444')
for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
    pane.fill = False
    pane.set_edgecolor('#1a1a33')
ax.grid(True, color='#1a1a33', linewidth=0.5)
ax.set_title(
    f'Vector Geometry of Regression  (R²={R2:.3f},  3D captures '
    f'{sum(var_explained[:3]):.0%} of structure)',
    color='white', fontsize=12, pad=14)

info = (f"r(X₁,X₂)={r(X1,X2):.2f}  r(X₁,X₃)={r(X1,X3):.2f}  "
        f"r(X₂,X₃)={r(X2,X3):.2f}  r(Y,X₁)={r(Y,X1):.2f}  "
        f"r(Y,X₂)={r(Y,X2):.2f}  r(Y,X₃)={r(Y,X3):.2f}\n"
        f"Axes = top 3 PCs of variable vectors.  Orb overlap ∝ correlation (approx).")
fig.text(0.5, 0.01, info, ha='center', color='#aaaacc', fontsize=8,
         fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='#111133',
                   alpha=0.7, edgecolor='#333366'))

plt.tight_layout()
plt.savefig('regression_vectors.png', dpi=150, bbox_inches='tight',
            facecolor='#0e0e1a')
plt.show()
print("\nSaved → regression_vectors.png")