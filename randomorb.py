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
#
# Key idea: normalise all vectors to the same display length L.
# Place an orb of radius R at each tip.
# Distance between two tips = L * sqrt(2 - 2*r_ij), so overlap is driven purely
# by correlation: high r → tips close → orbs overlap a lot.
#
L = 3.5          # display length for all vectors
R = 2.0          # orb radius — chosen so pairs with r~0.3 still visibly overlap

u1  = v1  / np.linalg.norm(v1)   # unit vectors (direction only)
u2  = v2  / np.linalg.norm(v2)
uY  = vY  / np.linalg.norm(vY)
uYh = vY_hat / np.linalg.norm(vY_hat)

p1  = L * u1    # orb centres = vector tips at display length
p2  = L * u2
pY  = L * uY
pYh = L * uYh

fig = plt.figure(figsize=(11, 9), facecolor='#0e0e1a')
ax  = fig.add_subplot(111, projection='3d', facecolor='#0e0e1a')

# --- helper: arrow from origin to point --------------------------------
def arrow(tip, color, lw=2.5, ls='-'):
    ax.quiver(0, 0, 0, tip[0], tip[1], tip[2],
              color=color, linewidth=lw, linestyle=ls,
              arrow_length_ratio=0.10)

# --- helper: orb centred at a point ------------------------------------
def orb(centre, color, alpha=0.15):
    u, w = np.mgrid[0:2*np.pi:40j, 0:np.pi:30j]
    xs = centre[0] + R * np.cos(u) * np.sin(w)
    ys = centre[1] + R * np.sin(u) * np.sin(w)
    zs = centre[2] + R * np.cos(w)
    ax.plot_surface(xs, ys, zs, color=color, alpha=alpha, linewidth=0)

# --- regression plane in unit-vector space -----------------------------
lim = L * 1.4
ss = np.linspace(-lim, lim, 14)
tt = np.linspace(-lim, lim, 14)
S, T = np.meshgrid(ss, tt)
PX = S*u1[0] + T*(u2 - np.dot(u2,u1)*u1)[0] / np.linalg.norm(u2 - np.dot(u2,u1)*u1)
PY_p = S*u1[1] + T*(u2 - np.dot(u2,u1)*u1)[1] / np.linalg.norm(u2 - np.dot(u2,u1)*u1)
PZ = S*u1[2] + T*(u2 - np.dot(u2,u1)*u1)[2] / np.linalg.norm(u2 - np.dot(u2,u1)*u1)
ax.plot_surface(PX, PY_p, PZ, alpha=0.08, color='#4466dd', linewidth=0)
ax.plot_wireframe(PX, PY_p, PZ, alpha=0.06, color='#6688ff', linewidth=0.4)

# --- three orbs — centred at each vector tip ---------------------------
# Draw Y first (behind), then X1, X2 in front
orb(pY,  '#fff176', alpha=0.12)   # Y  — yellow
orb(p2,  '#81c784', alpha=0.14)   # X2 — green
orb(p1,  '#4fc3f7', alpha=0.14)   # X1 — blue

# --- arrows -------------------------------------------------------------
arrow(p1,  '#4fc3f7', lw=3)
arrow(p2,  '#81c784', lw=3)
arrow(pY,  '#fff176', lw=3)
arrow(pYh, '#ffb74d', lw=2, ls='--')   # Ŷ projection

# residual arrow: from pYh tip to pY tip
res_disp = pY - pYh
ax.quiver(pYh[0], pYh[1], pYh[2],
          res_disp[0], res_disp[1], res_disp[2],
          color='#ef5350', linewidth=1.8, linestyle=':',
          arrow_length_ratio=0.18)

# right-angle tick at projection foot
f_e1 = u1
f_e2 = (u2 - np.dot(u2,u1)*u1); f_e2 /= np.linalg.norm(f_e2)
res_dir = res_disp / np.linalg.norm(res_disp)
tick = 0.18
corner_pts = np.array([pYh + tick*f_e1,
                        pYh + tick*f_e1 + tick*res_dir,
                        pYh + tick*res_dir])
ax.plot(corner_pts[:,0], corner_pts[:,1], corner_pts[:,2],
        color='#ef5350', linewidth=1, alpha=0.6)

# --- labels -------------------------------------------------------------
off = 0.25
ax.text(*(p1 + off), 'X₁\nInternship', color='#4fc3f7',
        fontsize=11, fontweight='bold', fontfamily='monospace')
ax.text(*(p2 + off), 'X₂\nGPA',        color='#81c784',
        fontsize=11, fontweight='bold', fontfamily='monospace')
ax.text(*(pY + off), 'Y\nJob Score',   color='#fff176',
        fontsize=11, fontweight='bold', fontfamily='monospace')
ax.text(*(pYh - np.array([0,0,0.4])),  'Ŷ',  color='#ffb74d',
        fontsize=10, fontfamily='monospace')
ax.text(*(pYh + res_disp*0.55 + np.array([0.1,0,0])), 'e (⊥)',
        color='#ef5350', fontsize=9, fontfamily='monospace')

# --- axes & styling -----------------------------------------------------
ax.set_xlabel('Dim 1', color='#666', labelpad=8)
ax.set_ylabel('Dim 2', color='#666', labelpad=8)
ax.set_zlabel('Dim 3', color='#666', labelpad=8)
ax.tick_params(colors='#444')
for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
    pane.fill = False
    pane.set_edgecolor('#1a1a33')
ax.grid(True, color='#1a1a33', linewidth=0.5)
ax.set_title(f'Vector Geometry of Regression  (R² = {R_squared:.3f})',
             color='white', fontsize=13, pad=14)

info = (f"r(X₁,X₂)={r12:.2f}   r(Y,X₁)={ry1:.2f}   r(Y,X₂)={ry2:.2f}\n"
        f"Orb centres = vector tips (normalised).  Overlap ∝ correlation.")
fig.text(0.5, 0.02, info, ha='center', color='#aaaacc', fontsize=9,
         fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='#111133', alpha=0.7, edgecolor='#333366'))

plt.tight_layout()
plt.savefig('regression_vectors.png', dpi=150, bbox_inches='tight',
            facecolor='#0e0e1a')
plt.show()
print("\nSaved → regression_vectors.png")