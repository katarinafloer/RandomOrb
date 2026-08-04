"""
Regression Vector Geometry in 3D
=================================
Each variable (Y, X1, X2) is a vector in observation space (R^10).
We embed them in R^3 preserving:
  - vector length  = standard deviation
  - angle between  = arccos(correlation)

Then OLS projection = drop Y perpendicularly onto the plane of X1 & X2.
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ── 1. SYNTHETIC DATASET ────────────────────────────────────────────────────
np.random.seed(7)

internship = np.array([1, 1, 0, 1, 0, 1, 0, 1, 0, 0], dtype=float)
GPA        = np.array([3.8, 3.5, 3.9, 3.2, 2.8, 2.9, 3.5, 3.7, 2.5, 3.1])
job_score  = -52.4 + 24.8*internship + 29.1*GPA + np.random.normal(0, 3, 10)

df = pd.DataFrame({'Internship': internship, 'GPA': GPA, 'JobScore': job_score.round(1)})
print("── Synthetic Dataset ──────────────────────────────")
print(df.to_string(index=False))

# ── 2. CENTER VARIABLES (required for pure vector geometry) ──────────────────
Y  = job_score  - job_score.mean()
X1 = internship - internship.mean()
X2 = GPA        - GPA.mean()

# ── 3. STD DEVS & PAIRWISE CORRELATIONS ─────────────────────────────────────
sy, s1, s2 = np.std(Y), np.std(X1), np.std(X2)

r12 = np.corrcoef(X1, X2)[0, 1]   # internship ↔ GPA
ry1 = np.corrcoef(Y,  X1)[0, 1]   # job score  ↔ internship
ry2 = np.corrcoef(Y,  X2)[0, 1]   # job score  ↔ GPA

print("\n── Correlations ───────────────────────────────────")
print(f"  r(X1, X2) = {r12:.3f}  (internship ↔ GPA)")
print(f"  r(Y,  X1) = {ry1:.3f}  (job score  ↔ internship)")
print(f"  r(Y,  X2) = {ry2:.3f}  (job score  ↔ GPA)")

# ── 4. BUILD 3D VECTORS ──────────────────────────────────────────────────────
#
# Place X1 along the x-axis.
# Place X2 in the xy-plane so angle(X1,X2) = arccos(r12).
# Place Y  in full 3D        so angle(Y,X1) = arccos(ry1) and
#                               angle(Y,X2) = arccos(ry2).
#
v1 = s1 * np.array([1.0, 0.0, 0.0])

v2 = s2 * np.array([r12,
                     np.sqrt(max(0.0, 1 - r12**2)),
                     0.0])

# Solve for unit-vector components of vY:
#   a = ry1   (from dot with v1/|v1|)
#   a*r12 + b*sqrt(1-r12^2) = ry2   (from dot with v2/|v2|)
#   c = sqrt(1 - a^2 - b^2)
a = ry1
b = (ry2 - ry1 * r12) / np.sqrt(max(1e-12, 1 - r12**2))
c = np.sqrt(max(0.0, 1 - a**2 - b**2))
vY = sy * np.array([a, b, c])

# ── 5. OLS PROJECTION ────────────────────────────────────────────────────────
# Orthonormal basis for the plane spanned by v1, v2
e1 = v1 / np.linalg.norm(v1)
e2 = v2 - np.dot(v2, e1) * e1
e2 = e2 / np.linalg.norm(e2)

vY_hat   = np.dot(vY, e1)*e1 + np.dot(vY, e2)*e2   # fitted values vector
residual = vY - vY_hat                               # ⊥ to both X1 and X2

R_squared = (np.linalg.norm(vY_hat) / np.linalg.norm(vY))**2

print("\n── Regression Geometry ────────────────────────────")
print(f"  |Y|   = {np.linalg.norm(vY):.3f}  (std of Y)")
print(f"  |Ŷ|   = {np.linalg.norm(vY_hat):.3f}  (explained)")
print(f"  |e|   = {np.linalg.norm(residual):.3f}  (residual)")
print(f"  R²    = {R_squared:.3f}")
print(f"  e ⊥ X1: dot = {np.dot(residual, v1):.6f}  (≈ 0 ✓)")
print(f"  e ⊥ X2: dot = {np.dot(residual, v2):.6f}  (≈ 0 ✓)")

# ── 6. VISUALISE ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(11, 9), facecolor='#0e0e1a')
ax  = fig.add_subplot(111, projection='3d', facecolor='#0e0e1a')

# --- helper: draw an arrow from origin ---------------------------------
def arrow(v, color, lw=2.5, ls='-'):
    ax.quiver(0, 0, 0, v[0], v[1], v[2],
              color=color, linewidth=lw, linestyle=ls,
              arrow_length_ratio=0.12)

# --- helper: draw a transparent sphere at tip of v ---------------------
def orb(v, color, alpha=0.18, scale=0.9):
    u, w = np.mgrid[0:2*np.pi:30j, 0:np.pi:20j]
    r = np.linalg.norm(v) * scale * 0.28        # radius = fraction of length
    xs = v[0] + r * np.cos(u) * np.sin(w)
    ys = v[1] + r * np.sin(u) * np.sin(w)
    zs = v[2] + r * np.cos(w)
    ax.plot_surface(xs, ys, zs, color=color, alpha=alpha, linewidth=0)

# --- regression plane (spanned by e1, e2) ------------------------------
lim = max(s1, s2, sy) * 1.3
ss = np.linspace(-lim, lim, 12)
tt = np.linspace(-lim, lim, 12)
S, T = np.meshgrid(ss, tt)
PX = S*e1[0] + T*e2[0]
PY = S*e1[1] + T*e2[1]
PZ = S*e1[2] + T*e2[2]
ax.plot_surface(PX, PY, PZ, alpha=0.10, color='#4466dd', linewidth=0)
ax.plot_wireframe(PX, PY, PZ, alpha=0.08, color='#6688ff', linewidth=0.4)

# --- draw vectors -------------------------------------------------------
arrow(v1,       '#4fc3f7', lw=3)          # X1 internship — blue
arrow(v2,       '#81c784', lw=3)          # X2 GPA        — green
arrow(vY,       '#fff176', lw=3)          # Y job score   — yellow
arrow(vY_hat,   '#ffb74d', lw=2, ls='--') # Ŷ projection  — orange dashed

# residual from tip of vY_hat to tip of vY
ax.quiver(vY_hat[0], vY_hat[1], vY_hat[2],
          residual[0], residual[1], residual[2],
          color='#ef5350', linewidth=1.8, linestyle=':',
          arrow_length_ratio=0.18)

# right-angle marker at projection point (3 line segments)
tick = 0.12
perp_a = tick * e1
perp_b = tick * (residual / np.linalg.norm(residual))
p0, p1, p2 = vY_hat, vY_hat + perp_a, vY_hat + perp_b
corner_pts = np.array([p1, p1 + perp_b, p2])
ax.plot(corner_pts[:,0], corner_pts[:,1], corner_pts[:,2],
        color='#ef5350', linewidth=1, alpha=0.7)

# --- orbs at vector tips ------------------------------------------------
orb(v1,  '#4fc3f7', alpha=0.22)
orb(v2,  '#81c784', alpha=0.22)
orb(vY,  '#fff176', alpha=0.22)
orb(vY_hat, '#ffb74d', alpha=0.15)

# --- labels -------------------------------------------------------------
def label(v, txt, color, offset=(0.05, 0.05, 0.05)):
    ax.text(v[0]+offset[0], v[1]+offset[1], v[2]+offset[2],
            txt, color=color, fontsize=11, fontweight='bold',
            fontfamily='monospace')

label(v1,     'X₁\nInternship', '#4fc3f7')
label(v2,     'X₂\nGPA',        '#81c784')
label(vY,     'Y\nJob Score',   '#fff176')
label(vY_hat, 'Ŷ (fitted)',     '#ffb74d', offset=(0.05, 0.05, -0.15))
mid = vY_hat + residual*0.55
label(mid,    'e  (⊥ plane)',   '#ef5350', offset=(0.05,0,0))

# --- axes & styling -----------------------------------------------------
ax.set_xlabel('Dim 1', color='#888', labelpad=8)
ax.set_ylabel('Dim 2', color='#888', labelpad=8)
ax.set_zlabel('Dim 3', color='#888', labelpad=8)
ax.tick_params(colors='#555')
for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
    pane.fill = False
    pane.set_edgecolor('#222244')
ax.grid(True, color='#222244', linewidth=0.5)
ax.set_title(f'Vector Geometry of Regression  (R² = {R_squared:.3f})',
             color='white', fontsize=13, pad=14)

# --- annotation box -----------------------------------------------------
info = (f"r(X₁,X₂) = {r12:.2f}   r(Y,X₁) = {ry1:.2f}   r(Y,X₂) = {ry2:.2f}\n"
        f"Orb overlap ∝ correlation  |  e ⊥ both X vectors by construction")
fig.text(0.5, 0.02, info, ha='center', color='#aaaacc', fontsize=9,
         fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='#111133', alpha=0.7, edgecolor='#333366'))

plt.tight_layout()
plt.savefig('regression_vectors.png', dpi=150, bbox_inches='tight',
            facecolor='#0e0e1a')
plt.show()
print("\nSaved → regression_vectors.png")
