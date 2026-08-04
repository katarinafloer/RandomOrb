"""
Regression Vector Geometry in 3D
=================================
In OLS regression, we usually think of variables as columns of numbers.
But there's a geometric view: each variable is a VECTOR in n-dimensional
space (one dimension per observation). With n=10 students, each variable
lives in R^10.

The key facts that make this useful:
  - The LENGTH of a variable's vector = its standard deviation
  - The ANGLE between two vectors = arccos(their correlation)
    → perfectly correlated = same direction (angle = 0)
    → uncorrelated         = perpendicular  (angle = 90°)
  - OLS fitted values Ŷ = the PROJECTION of Y onto the plane of X1 & X2
  - The residual e = Y - Ŷ, and it is always PERPENDICULAR to that plane

We can't draw R^10, but for 3 variables we only need R^3:
three vectors always fit in 3D while preserving all angles and lengths.

The ORBS (spheres) around each vector tip visualise the variable's "reach"
in space. Two orbs overlap when their vectors point in similar directions
— i.e. when the two variables are correlated.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# ── 1. SYNTHETIC DATASET ─────────────────────────────────────────────────────
# We invent 10 students with known internship status and GPA.
# The "true" population model we use to generate scores is:
#   job_score = -52.4 + 24.8*internship + 29.1*GPA + noise
# These coefficients are just a device to produce realistic numbers.
# We never use them again — all the geometry below comes from the DATA.

np.random.seed(7)

internship = np.array([1, 1, 0, 1, 0, 1, 0, 1, 0, 0], dtype=float)
GPA        = np.array([3.8, 3.5, 3.9, 3.2, 2.8, 2.9, 3.5, 3.7, 2.5, 3.1])
job_score  = -52.4 + 24.8*internship + 29.1*GPA + np.random.normal(0, 3, 10)

df = pd.DataFrame({'Internship': internship, 'GPA': GPA, 'JobScore': job_score.round(1)})
print("── Synthetic Dataset ──────────────────────────────")
print(df.to_string(index=False))


# ── 2. CENTER VARIABLES ───────────────────────────────────────────────────────
# We subtract each variable's mean so that all vectors pass through the origin.
# This is required for the geometric interpretation: the inner product of two
# centred vectors equals their covariance, and dividing by lengths gives the
# correlation. Without centering, a non-zero mean would add an uninformative
# "intercept direction" that clutters the geometry.

Y  = job_score  - job_score.mean()
X1 = internship - internship.mean()
X2 = GPA        - GPA.mean()


# ── 3. STANDARD DEVIATIONS & PAIRWISE CORRELATIONS ───────────────────────────
# These two numbers fully describe each vector in our 3D embedding:
#   std dev   → how long the vector is
#   correlation → the angle between any two vectors

sy, s1, s2 = np.std(Y), np.std(X1), np.std(X2)

r12 = np.corrcoef(X1, X2)[0, 1]   # internship ↔ GPA
ry1 = np.corrcoef(Y,  X1)[0, 1]   # job score  ↔ internship
ry2 = np.corrcoef(Y,  X2)[0, 1]   # job score  ↔ GPA

print("\n── Correlations ───────────────────────────────────")
print(f"  r(X1, X2) = {r12:.3f}  (internship ↔ GPA)")
print(f"  r(Y,  X1) = {ry1:.3f}  (job score  ↔ internship)")
print(f"  r(Y,  X2) = {ry2:.3f}  (job score  ↔ GPA)")


# ── 4. BUILD THE 3D VECTORS ───────────────────────────────────────────────────
# We place the three vectors so that every length and every angle is preserved.
#
# Step 1 — X1 goes along the x-axis (arbitrary; we pick this as our reference).
v1 = s1 * np.array([1.0, 0.0, 0.0])

# Step 2 — X2 goes in the xy-plane.
# Its x-component must satisfy: dot(u1, u2) = cos(angle) = r12
# Its y-component fills out the unit length: sqrt(1 - r12^2)
v2 = s2 * np.array([r12,
                     np.sqrt(max(0.0, 1 - r12**2)),
                     0.0])

# Step 3 — Y goes into full 3D (it may point out of the X1-X2 plane).
# We solve for a unit vector [a, b, c] such that:
#   dot with u1 = ry1   →  a = ry1
#   dot with u2 = ry2   →  a*r12 + b*sqrt(1-r12^2) = ry2  →  solve for b
#   c fills out unit length: c = sqrt(1 - a^2 - b^2)
# The c component is non-zero when Y is NOT fully explained by X1 and X2
# (i.e. when R² < 1). It represents the residual "direction."
a = ry1
b = (ry2 - ry1 * r12) / np.sqrt(max(1e-12, 1 - r12**2))
c = np.sqrt(max(0.0, 1 - a**2 - b**2))
vY = sy * np.array([a, b, c])


# ── 5. OLS PROJECTION ────────────────────────────────────────────────────────
# OLS asks: "what point in the plane of X1 & X2 is closest to Y?"
# Geometrically that is the perpendicular projection of vY onto the plane.
#
# We build an orthonormal basis for the X1-X2 plane using Gram-Schmidt:
#   e1 = unit vector along X1
#   e2 = unit vector along X2, after removing its X1 component
# Then project vY onto each basis vector and add up.

e1 = v1 / np.linalg.norm(v1)
e2 = v2 - np.dot(v2, e1) * e1        # remove X1 component from X2
e2 = e2 / np.linalg.norm(e2)         # normalise

vY_hat   = np.dot(vY, e1)*e1 + np.dot(vY, e2)*e2   # Ŷ: projection of Y onto plane
residual = vY - vY_hat                               # e: what's left over, ⊥ to plane

# R² = (length of Ŷ)² / (length of Y)²
# = fraction of Y's "reach" that is explained by the X1-X2 plane
R_squared = (np.linalg.norm(vY_hat) / np.linalg.norm(vY))**2

print("\n── Regression Geometry ────────────────────────────")
print(f"  |Y|   = {np.linalg.norm(vY):.3f}  (total spread of job score)")
print(f"  |Ŷ|   = {np.linalg.norm(vY_hat):.3f}  (spread explained by internship + GPA)")
print(f"  |e|   = {np.linalg.norm(residual):.3f}  (unexplained residual)")
print(f"  R²    = {R_squared:.3f}  (= |Ŷ|² / |Y|²)")
print(f"  e ⊥ X1: dot = {np.dot(residual, v1):.6f}  (must be 0 by OLS construction ✓)")
print(f"  e ⊥ X2: dot = {np.dot(residual, v2):.6f}  (must be 0 by OLS construction ✓)")


# ── 6. VISUALISE ─────────────────────────────────────────────────────────────
# The actual vectors (v1, v2, vY) have very different lengths because the
# variables have different scales (s1 ≈ 0.5, s2 ≈ 0.37, sy ≈ 20).
# For a readable picture we normalise all vectors to the same display length L,
# then place an orb (sphere) of radius R at each tip.
#
# What does the orb represent?
#   - The centre is the "direction" of that variable in vector space.
#   - Two orbs overlap when their vector tips are close — i.e. when the
#     variables point in similar directions, i.e. when they are correlated.
#   - The MORE correlated two variables are, the MORE their orbs overlap.
#   - The overlap between X1 and X2's orbs is the shared variance that
#     causes omitted-variable bias when one of them is left out.

L = 3.5    # all vectors drawn at this display length
R = 2.0    # orb radius — tuned so pairs with r ≈ 0.3 still visibly intersect

# Convert to unit vectors (direction only), then scale to display length
u1  = v1      / np.linalg.norm(v1)
u2  = v2      / np.linalg.norm(v2)
uY  = vY      / np.linalg.norm(vY)
uYh = vY_hat  / np.linalg.norm(vY_hat)

# Orb centres: tips of the scaled vectors
p1  = L * u1
p2  = L * u2
pY  = L * uY
pYh = L * uYh

fig = plt.figure(figsize=(11, 9), facecolor='#0e0e1a')
ax  = fig.add_subplot(111, projection='3d', facecolor='#0e0e1a')


# --- helper: draw an arrow from the origin to a point ------------------
def arrow(tip, color, lw=2.5, ls='-'):
    ax.quiver(0, 0, 0, tip[0], tip[1], tip[2],
              color=color, linewidth=lw, linestyle=ls,
              arrow_length_ratio=0.10)


# --- helper: draw a transparent sphere centred at a point --------------
def orb(centre, color, alpha=0.15):
    u, w = np.mgrid[0:2*np.pi:40j, 0:np.pi:30j]
    xs = centre[0] + R * np.cos(u) * np.sin(w)
    ys = centre[1] + R * np.sin(u) * np.sin(w)
    zs = centre[2] + R * np.cos(w)
    ax.plot_surface(xs, ys, zs, color=color, alpha=alpha, linewidth=0)


# --- regression plane: the surface spanned by X1 and X2 ---------------
# OLS finds the point on this plane closest to Y (= the projection Ŷ).
# We parameterise the plane with the same orthonormal basis e1, e2.
lim = L * 1.4
ss = np.linspace(-lim, lim, 14)
tt = np.linspace(-lim, lim, 14)
S, T = np.meshgrid(ss, tt)
# Each point on the plane = s*e1 + t*e2 (in unit-vector coordinates)
e2_unit = (u2 - np.dot(u2, u1)*u1)
e2_unit = e2_unit / np.linalg.norm(e2_unit)
PX   = S*u1[0]    + T*e2_unit[0]
PY_p = S*u1[1]    + T*e2_unit[1]
PZ   = S*u1[2]    + T*e2_unit[2]
ax.plot_surface(PX, PY_p, PZ, alpha=0.08, color='#4466dd', linewidth=0)
ax.plot_wireframe(PX, PY_p, PZ, alpha=0.06, color='#6688ff', linewidth=0.4)


# --- four variable orbs ------------------------------------------------
# Each variable gets its own orb centred at its vector tip.
# Y and Ŷ will nearly overlap because R²=0.982 — their vectors point in
# almost the same direction. The small gap between them IS the residual.
orb(pY,  '#fff176', alpha=0.12)   # Y  job score      — yellow
orb(pYh, '#ffb74d', alpha=0.12)   # Ŷ  fitted values  — orange (almost same spot as Y)
orb(p2,  '#81c784', alpha=0.14)   # X2 GPA            — green
orb(p1,  '#4fc3f7', alpha=0.14)   # X1 internship     — blue


# --- variable arrows ----------------------------------------------------
arrow(p1,  '#4fc3f7', lw=3)           # X1
arrow(p2,  '#81c784', lw=3)           # X2
arrow(pY,  '#fff176', lw=3)           # Y
arrow(pYh, '#ffb74d', lw=2, ls='--')  # Ŷ (dashed — it's derived, not observed)


# --- residual: the gap between Ŷ and Y ---------------------------------
# This arrow is perpendicular to the X1-X2 plane by OLS construction.
res_disp = pY - pYh
ax.quiver(pYh[0], pYh[1], pYh[2],
          res_disp[0], res_disp[1], res_disp[2],
          color='#ef5350', linewidth=1.8, linestyle=':',
          arrow_length_ratio=0.18)

# Small right-angle marker at the foot of the perpendicular
res_dir = res_disp / np.linalg.norm(res_disp)
tick = 0.18
corner_pts = np.array([pYh + tick*u1,
                        pYh + tick*u1 + tick*res_dir,
                        pYh            + tick*res_dir])
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
ax.text(*(pYh - np.array([0, 0, 0.4])), 'Ŷ (projection)', color='#ffb74d',
        fontsize=9, fontfamily='monospace')
ax.text(*(pYh + res_disp*0.55 + np.array([0.1, 0, 0])), 'e  (⊥ to plane)',
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

# Annotation summarising the key numbers
info = (f"r(X₁,X₂)={r12:.2f}   r(Y,X₁)={ry1:.2f}   r(Y,X₂)={ry2:.2f}\n"
        f"Orb centres = vector tips (all normalised to same length). "
        f"Overlap ∝ correlation.")
fig.text(0.5, 0.02, info, ha='center', color='#aaaacc', fontsize=9,
         fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='#111133',
                   alpha=0.7, edgecolor='#333366'))

plt.tight_layout()
plt.savefig('regression_vectors.png', dpi=150, bbox_inches='tight',
            facecolor='#0e0e1a')
plt.show()
print("\nSaved → regression_vectors.png")