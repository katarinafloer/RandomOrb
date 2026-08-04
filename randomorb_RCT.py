"""
RCT Regression — 3D Biplot with Randomization Surface
======================================================
STORY IN THREE ACTS:
  Act 1 — Pre-randomization: you observe X2 (GPA) and X3 (parental income).
           Students appear as dots in a biplot of those two covariates.
           That is ALL you need to randomize.

  Act 2 — Randomization: draw a surface through the student cloud.
           Which side you land on determines T vs C.
           The surface's perpendicular direction is where X1 (treatment)
           would point — orthogonal to X2 and X3 if balance is achieved.

  Act 3 — After the experiment: outcome Y and fitted values Ŷ appear.
           The angle between Y and Ŷ shows residual error.
           X1 still never gets its own orb — it was created by the surface.

BIPLOT MECHANICS:
  Data matrix: 10 students × 4 variables  [X2, X3, Y, Ŷ]  (no X1)
  SVD: X = U D V^T

  Variable arrows → rows of V^T   (one arrow + orb per variable)
    angle between arrows ≈ arccos(correlation)
  Student points  → rows of U * D  (one dot per student)
    students near an arrow = high on that variable

RANDOMIZATION: Matched-pair (gold standard)
  - Sort students by PC1 of standardised {GPA, income}
  - Pair consecutive students (most similar covariate profiles)
  - Coin-flip within each pair → 5T / 5C
  - Guarantees r(X1, X2) ≈ 0  and  r(X1, X3) ≈ 0
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(7)

# ── 1. PRE-TREATMENT COVARIATES ───────────────────────────────────────────────
GPA             = np.array([3.8, 3.5, 3.9, 3.2, 2.8, 2.9, 3.5, 3.7, 2.5, 3.1])
parental_income = np.array([95,  60, 110,  45,  40,  55,  80,  75,  35,  50], dtype=float)
n = len(GPA)

# ── 2. MATCHED-PAIR RANDOMIZATION ────────────────────────────────────────────
# Using only X2 and X3 — we don't need Y to randomize.
cov_std = np.column_stack([
    (GPA             - GPA.mean())             / GPA.std(),
    (parental_income - parental_income.mean()) / parental_income.std()
])
_, _, Vt_cov = np.linalg.svd(cov_std, full_matrices=False)
pc1_score  = cov_std @ Vt_cov[0]           # rank students by covariate similarity
sorted_idx = np.argsort(pc1_score)
pairs      = [(sorted_idx[i], sorted_idx[i+1]) for i in range(0, n, 2)]

treatment = np.zeros(n, dtype=float)
for a, b in pairs:
    if np.random.random() < 0.5:
        treatment[a] = 1
    else:
        treatment[b] = 1

def r(a, b): return np.corrcoef(a, b)[0, 1]

print("── Matched-pair assignment ────────────────────────")
df = pd.DataFrame({
    'Student'  : [f'S{i+1}' for i in range(n)],
    'GPA'      : GPA,
    'Income($k)': parental_income.astype(int),
    'Arm'      : ['T' if t else 'C' for t in treatment]
})
print(df.to_string(index=False))
print(f"\n  r(X1,X2) = {r(treatment,GPA):.3f}  <- ~0 means balanced on GPA")
print(f"  r(X1,X3) = {r(treatment,parental_income):.3f}  <- ~0 means balanced on income")

# ── 3. OUTCOME & OLS ─────────────────────────────────────────────────────────
job_score = (-52.4 + 24.8*treatment + 29.1*GPA
             + 0.08*parental_income + np.random.normal(0, 15, n))

X_mat = np.column_stack([np.ones(n), treatment, GPA, parental_income])
b_hat = np.linalg.lstsq(X_mat, job_score, rcond=None)[0]
Y_hat = X_mat @ b_hat
R2    = 1 - np.sum((job_score-Y_hat)**2) / np.sum((job_score-job_score.mean())**2)

print(f"\n── OLS results ────────────────────────────────────")
print(f"  b1(treatment)={b_hat[1]:.1f}  b2(GPA)={b_hat[2]:.1f}  "
      f"b3(income)={b_hat[3]:.2f}  R²={R2:.3f}")

# ── 4. BIPLOT SVD — all 5 regression variables: X1, X2, X3, Y, Ŷ
# X1 (internship) gets an orb as a regression variable.
# The surface shows HOW X1 was assigned (T/C split) — separate from the orb.
X_data = np.column_stack([treatment, GPA, parental_income, job_score, Y_hat])
X_std  = (X_data - X_data.mean(axis=0)) / X_data.std(axis=0)

U_b, S_b, Vt_b = np.linalg.svd(X_std, full_matrices=False)

var_exp = (S_b**2) / np.sum(S_b**2)
print(f"\n── Biplot quality ─────────────────────────────────")
print(f"  3 PCs capture {sum(var_exp[:3]):.1%} of total structure")

# Variable coordinates: scale to display length L
L = 3.5
var_raw  = Vt_b[:3, :].T                    # (5 variables, 3 dims)
var_norm = max(np.linalg.norm(v) for v in var_raw)
var_pts  = {name: L * var_raw[i] / var_norm
            for i, name in enumerate(['X1','X2','X3','Y','Yh'])}

# Student coordinates: scale so they sit inside the orb cluster
stu_raw  = U_b[:, :3] * S_b[:3]            # (10 students, 3 dims)
stu_norm = max(np.linalg.norm(s) for s in stu_raw)
stu_pts  = stu_raw / stu_norm * L * 0.6

p1  = var_pts['X1'];  p2  = var_pts['X2'];  p3  = var_pts['X3']
pY  = var_pts['Y'];   pYh = var_pts['Yh']
R_orb = 1.8   # orb radius

# ── 5. RANDOMIZATION SURFACE ─────────────────────────────────────────────────
# The surface is the perpendicular bisector between T-centroid and C-centroid
# in biplot space. Students on one side → Treatment, other side → Control.
#
# Key insight: the surface's NORMAL (perpendicular direction) is where X1
# would point. If the randomization is balanced, this normal is approximately
# orthogonal to the X2 and X3 arrows — that's what balance means geometrically.
t_pts = stu_pts[treatment == 1]
c_pts = stu_pts[treatment == 0]
t_cen = t_pts.mean(axis=0)
c_cen = c_pts.mean(axis=0)
mid   = (t_cen + c_cen) / 2
norm_vec = t_cen - c_cen
norm_vec = norm_vec / np.linalg.norm(norm_vec)   # unit normal = X1 direction

arb = np.array([0,1,0]) if abs(norm_vec[1]) < 0.9 else np.array([1,0,0])
f1  = arb - np.dot(arb, norm_vec)*norm_vec;  f1 /= np.linalg.norm(f1)
f2  = np.cross(norm_vec, f1);                f2 /= np.linalg.norm(f2)

disc_r     = R_orb * 2.8
rhos, phis = np.linspace(0, disc_r, 35), np.linspace(0, 2*np.pi, 70)
RHO, PHI   = np.meshgrid(rhos, phis)
warp       = 0.3 * np.cos(2*PHI) * (RHO/disc_r)
SX = mid[0] + RHO*(np.cos(PHI)*f1[0]+np.sin(PHI)*f2[0]) + warp*norm_vec[0]
SY = mid[1] + RHO*(np.cos(PHI)*f1[1]+np.sin(PHI)*f2[1]) + warp*norm_vec[1]
SZ = mid[2] + RHO*(np.cos(PHI)*f1[2]+np.sin(PHI)*f2[2]) + warp*norm_vec[2]

# Verify balance: angle between surface normal and covariate arrows
ang2 = np.degrees(np.arccos(np.clip(abs(np.dot(norm_vec, p2/np.linalg.norm(p2))), 0, 1)))
ang3 = np.degrees(np.arccos(np.clip(abs(np.dot(norm_vec, p3/np.linalg.norm(p3))), 0, 1)))
print(f"\n── Randomization balance (geometric check) ────────")
print(f"  Angle(surface normal, X2) = {ang2:.1f}°  (90° = perfectly balanced on GPA)")
print(f"  Angle(surface normal, X3) = {ang3:.1f}°  (90° = perfectly balanced on income)")

# ── 6. VISUALISE ─────────────────────────────────────────────────────────────
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
z_floor = stu_pts[:, 2].min() - 0.4
for i, (pt, t) in enumerate(zip(stu_pts, treatment)):
    col = '#80cbc4' if t else '#ef9a9a'          # teal = T, coral = C
    ax.scatter(*pt, color=col, s=90, zorder=6,
               edgecolors='white', linewidths=0.6)
    ax.text(pt[0]+0.1, pt[1]+0.1, pt[2]+0.1, f'S{i+1}',
            color=col, fontsize=7.5, fontfamily='monospace', alpha=0.9)
    ax.plot([pt[0],pt[0]], [pt[1],pt[1]], [pt[2], z_floor],
            color=col, linewidth=0.35, alpha=0.25)

# ── RANDOMIZATION SURFACE ────────────────────────────────────────────────────
ax.plot_surface(SX, SY, SZ, alpha=0.18, color='#b0bec5', linewidth=0)
phi_r = np.linspace(0, 2*np.pi, 150)
ring  = (mid + disc_r*(np.outer(np.cos(phi_r),f1)+np.outer(np.sin(phi_r),f2))
         + np.outer(0.3*np.cos(2*phi_r), norm_vec))
ax.plot(ring[:,0], ring[:,1], ring[:,2], color='#90a4ae', lw=1.5, alpha=0.8)

# X1 arrow: surface normal = internship assignment direction
# The surface CREATES X1 — so X1 is perpendicular to the surface by design.
# This is visible in the plot: the teal X1 arrow pokes out of the surface.

# Labels for T and C sides
ax.text(*(mid + norm_vec*R_orb*2.0 + f1*disc_r*0.5),
        'TREATMENT\narm', color='#80cbc4', fontsize=8.5, fontfamily='monospace',
        fontweight='bold')
ax.text(*(mid - norm_vec*R_orb*2.0 + f1*disc_r*0.5),
        'CONTROL\narm', color='#ef9a9a', fontsize=8.5, fontfamily='monospace',
        fontweight='bold')

# ── 5 REGRESSION VARIABLE ORBS: internship, GPA, parental $, Y, Ŷ ───────────
orb(p1,  '#4fc3f7', alpha=0.13)   # X1 — internship (RCT)
orb(p2,  '#81c784', alpha=0.13)   # X2 — GPA
orb(p3,  '#ce93d8', alpha=0.13)   # X3 — parental income
orb(pY,  '#fff176', alpha=0.11)   # Y  — job score
orb(pYh, '#ffb74d', alpha=0.11)   # Ŷ  — fitted values

arrow(p1,  '#4fc3f7', lw=3)
arrow(p2,  '#81c784', lw=3)
arrow(p3,  '#ce93d8', lw=3)
arrow(pY,  '#fff176', lw=3)
arrow(pYh, '#ffb74d', lw=2, ls='--')

# Residual vector  e = Y - Ŷ
res = pY - pYh
ax.quiver(*pYh, *res, color='#ef5350', lw=1.8, linestyle=':', arrow_length_ratio=0.2)

# ── LABELS ───────────────────────────────────────────────────────────────────
off = 0.28
ax.text(*(p1+off),  'X₁  Internship',   color='#4fc3f7', fontsize=10,
        fontweight='bold', fontfamily='monospace')
ax.text(*(p2+off),  'X₂  GPA',          color='#81c784', fontsize=10,
        fontweight='bold', fontfamily='monospace')
ax.text(*(p3+off),  'X₃  Parental $',   color='#ce93d8', fontsize=10,
        fontweight='bold', fontfamily='monospace')
ax.text(*(pY+off),  'Y   Job Score',     color='#fff176', fontsize=10,
        fontweight='bold', fontfamily='monospace')
ax.text(*(pYh - np.array([0,0,0.6])),
        'Ŷ (fitted)',  color='#ffb74d', fontsize=9, fontfamily='monospace')
ax.text(*(pYh + res*0.6 + np.array([0.1,0,0])),
        'e (residual)', color='#ef5350', fontsize=8, fontfamily='monospace')

# ── STYLING ──────────────────────────────────────────────────────────────────
ax.set_xlabel('PC 1', color='#555', labelpad=8)
ax.set_ylabel('PC 2', color='#555', labelpad=8)
ax.set_zlabel('PC 3', color='#555', labelpad=8)
ax.tick_params(colors='#444')
for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
    pane.fill = False; pane.set_edgecolor('#1a1a33')
ax.grid(True, color='#1a1a33', linewidth=0.5)
ax.set_title(
    f'RCT Regression Biplot  (R²={R2:.3f},  3 PCs = {sum(var_exp[:3]):.0%} of structure)\n'
    f'Orbs = regression variables · Dots = students · Surface = randomization boundary',
    color='white', fontsize=11, pad=12)

info = (f"Surface ⊥ to X₂ at {ang2:.0f}°  and  X₃ at {ang3:.0f}°  "
        f"(90° = perfectly balanced)  |  r(X₁,X₂)={r(treatment,GPA):.3f}  "
        f"r(X₁,X₃)={r(treatment,parental_income):.3f}")
fig.text(0.5, 0.01, info, ha='center', color='#aaaacc', fontsize=8,
         fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='#111133', alpha=0.7, edgecolor='#333366'))

plt.tight_layout()
plt.savefig('regression_vectors.png', dpi=150, bbox_inches='tight', facecolor='#0e0e1a')
plt.show()
print("\nSaved → regression_vectors.png")