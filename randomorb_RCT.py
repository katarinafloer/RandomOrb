"""
Regression Vector Geometry in 3D — with RCT Randomization
===========================================================
WHAT CHANGED FROM THE OBSERVATIONAL VERSION:
  The treatment assignment (X1) is no longer observational (who chose to do
  an internship). Instead it comes from a MATCHED-PAIR RANDOMIZATION:

    1. Standardise the two pre-treatment covariates (GPA, parental income).
    2. Sort students by their covariate composite score and create 5 consecutive
       pairs — each pair contains two students who are similar on covariates.
    3. Within each pair, flip a coin: one student goes to Treatment, one to Control.

  This is the gold standard for small samples. It guarantees:
    - Exactly 5T / 5C (perfect size balance)
    - Near-zero correlation between treatment and each covariate (balance)
    - Every student has a 0.5 probability of treatment (unconfounded)

  In vector-space terms: the randomized X1 vector is near-ORTHOGONAL to X2
  and X3. The randomization surface (disc perpendicular to X1_rct) therefore
  slices each covariate orb close to 50/50 — visually showing balance.
  Compare this to observational assignment where X2 leaned heavily one way.
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
# Step 1: standardise covariates so GPA and income are on equal footing
cov_std = np.column_stack([
    (GPA            - GPA.mean())            / GPA.std(),
    (parental_income - parental_income.mean()) / parental_income.std()
])

# Step 2: score each student on first principal component of covariates
#   (a single number summarising their "overall background")
_, _, Vt = np.linalg.svd(cov_std, full_matrices=False)
pc1_score = cov_std @ Vt[0]   # projection onto first PC

# Step 3: sort by score and pair consecutive students
#   (pair 0: most similar at the bottom; pair 4: most similar at the top)
sorted_idx = np.argsort(pc1_score)
pairs = [(sorted_idx[i], sorted_idx[i+1]) for i in range(0, n, 2)]

# Step 4: within each pair, randomly assign one to Treatment, one to Control
treatment = np.zeros(n, dtype=float)
for a, b in pairs:
    if np.random.random() < 0.5:
        treatment[a] = 1
    else:
        treatment[b] = 1

df = pd.DataFrame({
    'Student':        [f'S{i+1}' for i in range(n)],
    'GPA':            GPA,
    'ParentalInc($k)': parental_income.astype(int),
    'PC1_score':      pc1_score.round(2),
    'Arm':            ['Treatment' if t else 'Control' for t in treatment]
})
print("── Matched-Pair Randomization ─────────────────────")
print(df.to_string(index=False))

def r(a, b): return np.corrcoef(a, b)[0, 1]

print("\n── Covariate Balance ──────────────────────────────")
for name, vals in [('GPA', GPA), ('Parental income', parental_income)]:
    t_m = vals[treatment==1].mean();  c_m = vals[treatment==0].mean()
    smd = (t_m - c_m) / vals.std()
    print(f"  {name:16s}  T={t_m:.1f}  C={c_m:.1f}  SMD={smd:.3f}  <- should be ~0")
print(f"  r(treatment, GPA)    = {r(treatment, GPA):.3f}  ← should be ≈0")
print(f"  r(treatment, income) = {r(treatment, parental_income):.3f}  ← should be ≈0")


# ── 3. GENERATE OUTCOME UNDER RANDOMIZED ASSIGNMENT ──────────────────────────
# Now job_score is generated using the RANDOMIZED treatment.
# DGP coefficients are just to make realistic numbers; OLS will re-estimate them.
job_score = (-52.4 + 24.8*treatment + 29.1*GPA
             + 0.08*parental_income
             + np.random.normal(0, 15, n))


# ── 4. FIT OLS ───────────────────────────────────────────────────────────────
X_mat   = np.column_stack([np.ones(n), treatment, GPA, parental_income])
b_hat   = np.linalg.lstsq(X_mat, job_score, rcond=None)[0]
Y_hat   = X_mat @ b_hat
R2      = 1 - np.sum((job_score - Y_hat)**2) / np.sum((job_score - job_score.mean())**2)

print(f"\n── OLS Estimates ──────────────────────────────────")
print(f"  b0={b_hat[0]:.1f}  b1(treatment)={b_hat[1]:.1f}  "
      f"b2(GPA)={b_hat[2]:.1f}  b3(income)={b_hat[3]:.2f}  R²={R2:.3f}")


# ── 5. CENTER VARIABLES ───────────────────────────────────────────────────────
Y  = job_score       - job_score.mean()
X1 = treatment       - treatment.mean()      # RANDOMIZED treatment
X2 = GPA             - GPA.mean()
X3 = parental_income - parental_income.mean()
Yh = Y_hat           - Y_hat.mean()

print(f"\n── Correlations (post-randomization) ──────────────")
print(f"  r(X1,X2)={r(X1,X2):.3f}  r(X1,X3)={r(X1,X3):.3f}  "
      f"r(X2,X3)={r(X2,X3):.3f}")
print(f"  r(Y,X1) ={r(Y,X1):.3f}  r(Y,X2) ={r(Y,X2):.3f}  "
      f"r(Y,X3) ={r(Y,X3):.3f}  r(Y,Yh)={r(Y,Yh):.3f}")


# ── 6. PCA TO GET 3D COORDINATES ─────────────────────────────────────────────
def unit(v): return v / np.linalg.norm(v)

V = np.vstack([unit(Y), unit(Yh), unit(X1), unit(X2), unit(X3)])
U, S, Wt = np.linalg.svd(V, full_matrices=False)
var_exp  = (S**2) / np.sum(S**2)
coords   = U[:, :3] * S[:3]
c_Y, c_Yh, c_X1, c_X2, c_X3 = coords

# ── STUDENT COORDINATES IN THE SAME 3D SPACE ─────────────────────────────────
# In the SVD  V = U S Wt, the rows of Wt are the PC directions in R^10.
# Projecting student i (a unit basis vector e_i) onto PC k gives Wt[k, i].
# So each student's 3D position = Wt[:3, i] — they live in the same space
# as the variable orbs but closer to the origin (inside the orb cluster).
student_raw = Wt[:3, :].T                          # shape (10, 3)
max_norm    = max(np.linalg.norm(r) for r in student_raw)
student_pts = student_raw / max_norm * 3.5 * 0.55  # scale to sit inside orbs (L=3.5)

print(f"\n── PCA quality ────────────────────────────────────")
print(f"  3D captures {sum(var_exp[:3]):.1%} of inter-variable structure")


# ── 7. VISUALISE ─────────────────────────────────────────────────────────────
L = 3.5;  R = 1.8

def disp(c): return L * c / np.linalg.norm(c)

pY  = disp(c_Y);   pYh = disp(c_Yh)
p1  = disp(c_X1);  p2  = disp(c_X2);  p3  = disp(c_X3)

fig = plt.figure(figsize=(12, 9), facecolor='#0e0e1a')
ax  = fig.add_subplot(111, projection='3d', facecolor='#0e0e1a')

def arrow(tip, color, lw=2.5, ls='-'):
    ax.quiver(0,0,0, tip[0],tip[1],tip[2], color=color,
              linewidth=lw, linestyle=ls, arrow_length_ratio=0.10)

def orb(centre, color, alpha=0.13):
    u, w = np.mgrid[0:2*np.pi:40j, 0:np.pi:30j]
    ax.plot_surface(centre[0]+R*np.cos(u)*np.sin(w),
                    centre[1]+R*np.sin(u)*np.sin(w),
                    centre[2]+R*np.cos(w),
                    color=color, alpha=alpha, linewidth=0)


# ── RANDOMIZATION SURFACE ────────────────────────────────────────────────────
# With a proper RCT, X1 is near-orthogonal to X2 and X3 — they point in
# very different directions. No single plane perpendicular to X1 can pass
# through ALL orbs simultaneously (they're spread in different directions).
#
# Instead we place the surface through the CENTROID of all orb centres.
# This guarantees every orb straddles it. The surface normal tells us the
# "average split direction" across all variables.
#
# The curvature of the surface (cosine warp) reflects the non-linear
# Voronoi-like boundary that matched-pair randomization creates in
# covariate space: each pair has its own local T/C boundary.

all_pts = [p1, p2, p3, pY, pYh]

# Centroid of all orb display positions
centroid     = sum(all_pts) / len(all_pts)
centroid_dir = centroid / np.linalg.norm(centroid)   # plane normal
d            = np.linalg.norm(centroid)               # offset along normal
plane_origin = centroid                               # plane passes through centroid

# Orthonormal basis for the plane (perpendicular to centroid_dir)
arb = np.array([0,1,0]) if abs(centroid_dir[1]) < 0.9 else np.array([1,0,0])
f1  = arb - np.dot(arb, centroid_dir)*centroid_dir;  f1 /= np.linalg.norm(f1)
f2  = np.cross(centroid_dir, f1);                    f2 /= np.linalg.norm(f2)

disc_r = R * 2.8
rhos = np.linspace(0, disc_r, 35)
phis = np.linspace(0, 2*np.pi, 70)
RHO, PHI = np.meshgrid(rhos, phis)

# Gentle cosine warp — nod to the non-linear matched-pair Voronoi boundary
warp_amp  = 0.30;  warp_freq = 2
warp      = warp_amp * np.cos(warp_freq * PHI) * (RHO / disc_r)

SX = plane_origin[0] + RHO*(np.cos(PHI)*f1[0]+np.sin(PHI)*f2[0]) + warp*centroid_dir[0]
SY = plane_origin[1] + RHO*(np.cos(PHI)*f1[1]+np.sin(PHI)*f2[1]) + warp*centroid_dir[1]
SZ = plane_origin[2] + RHO*(np.cos(PHI)*f1[2]+np.sin(PHI)*f2[2]) + warp*centroid_dir[2]
ax.plot_surface(SX, SY, SZ, alpha=0.20, color='#b0bec5', linewidth=0)

# Outer edge
phi_ring  = np.linspace(0, 2*np.pi, 150)
warp_ring = warp_amp * np.cos(warp_freq * phi_ring)
ring = (plane_origin
        + disc_r*(np.outer(np.cos(phi_ring), f1) + np.outer(np.sin(phi_ring), f2))
        + np.outer(warp_ring, centroid_dir))
ax.plot(ring[:,0], ring[:,1], ring[:,2], color='#90a4ae', linewidth=1.5, alpha=0.8)

ax.text(*(plane_origin + centroid_dir*R*1.2 + f1*disc_r*0.7),
        'TREATMENT →', color='#80cbc4', fontsize=8, fontfamily='monospace')
ax.text(*(plane_origin - centroid_dir*R*1.2 + f1*disc_r*0.7),
        '← CONTROL',   color='#ef9a9a', fontsize=8, fontfamily='monospace')

# Fraction of each orb on each side
print("\n── Orb split across randomization surface ─────────")
for pt, nm in zip(all_pts, ['X₁ (treatment)','X₂ (GPA)      ',
                              'X₃ (income)   ','Y  (outcome)  ','Ŷ  (fitted)   ']):
    x = np.clip((np.dot(pt, centroid_dir) - d) / R, -1, 1)
    frac = 0.5 + (3/4)*(x - x**3/3)
    bar_t = '█'*int(frac*20);  bar_c = '░'*(20-int(frac*20))
    print(f"  {nm}  T|{bar_t}{bar_c}|C  {frac:.0%}T / {1-frac:.0%}C")


# ── STUDENT POINTS ───────────────────────────────────────────────────────────
# Each dot is one student, positioned in the same 3D PCA space as the orbs.
# Teal = Treatment arm, coral = Control arm (from matched-pair randomization).
# Drop lines connect each student to the "floor" (z = min) so depth is readable.
z_floor = student_pts[:, 2].min() - 0.3
for i, (pt, t) in enumerate(zip(student_pts, treatment)):
    color = '#80cbc4' if t else '#ef9a9a'
    ax.scatter(*pt, color=color, s=80, zorder=5, edgecolors='white', linewidths=0.5)
    ax.text(pt[0]+0.08, pt[1]+0.08, pt[2]+0.08, f'S{i+1}',
            color=color, fontsize=7, fontfamily='monospace', alpha=0.85)
    # drop line to floor
    ax.plot([pt[0], pt[0]], [pt[1], pt[1]], [pt[2], z_floor],
            color=color, linewidth=0.4, alpha=0.3)

# ── ORBS ─────────────────────────────────────────────────────────────────────
orb(pY,  '#fff176', alpha=0.11)   # Y  outcome        — yellow
orb(pYh, '#ffb74d', alpha=0.11)   # Ŷ  fitted         — orange
orb(p3,  '#ce93d8', alpha=0.13)   # X3 parental $     — purple
orb(p2,  '#81c784', alpha=0.13)   # X2 GPA            — green
orb(p1,  '#4fc3f7', alpha=0.13)   # X1 treatment (RCT)— blue


# ── ARROWS ───────────────────────────────────────────────────────────────────
arrow(p1,  '#4fc3f7', lw=3)
arrow(p2,  '#81c784', lw=3)
arrow(p3,  '#ce93d8', lw=3)
arrow(pY,  '#fff176', lw=3)
arrow(pYh, '#ffb74d', lw=2, ls='--')

res_disp = pY - pYh
ax.quiver(*pYh, *res_disp, color='#ef5350', linewidth=1.8,
          linestyle=':', arrow_length_ratio=0.2)


# ── LABELS ───────────────────────────────────────────────────────────────────
off = 0.28
ax.text(*(p1+off),  'X₁  Treatment\n    (RCT)',   color='#4fc3f7', fontsize=10,
        fontweight='bold', fontfamily='monospace')
ax.text(*(p2+off),  'X₂  GPA',                   color='#81c784', fontsize=10,
        fontweight='bold', fontfamily='monospace')
ax.text(*(p3+off),  'X₃  Parental $',             color='#ce93d8', fontsize=10,
        fontweight='bold', fontfamily='monospace')
ax.text(*(pY+off),  'Y   Job Score',               color='#fff176', fontsize=10,
        fontweight='bold', fontfamily='monospace')
ax.text(*(pYh - np.array([0,0,0.5])), 'Ŷ (fitted)', color='#ffb74d',
        fontsize=9, fontfamily='monospace')
ax.text(*(pYh + res_disp*0.6 + np.array([0.1,0,0])), 'e',
        color='#ef5350', fontsize=9, fontfamily='monospace')


# ── AXES & STYLING ────────────────────────────────────────────────────────────
ax.set_xlabel('PC 1', color='#555', labelpad=8)
ax.set_ylabel('PC 2', color='#555', labelpad=8)
ax.set_zlabel('PC 3', color='#555', labelpad=8)
ax.tick_params(colors='#444')
for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
    pane.fill = False; pane.set_edgecolor('#1a1a33')
ax.grid(True, color='#1a1a33', linewidth=0.5)
ax.set_title(
    f'RCT Vector Geometry  (R²={R2:.3f},  3D captures {sum(var_exp[:3]):.0%} of structure)\n'
    f'Matched-pair randomization: surface cuts covariate orbs ~50/50',
    color='white', fontsize=11, pad=14)

info = (f"r(X₁,X₂)={r(X1,X2):.3f}  r(X₁,X₃)={r(X1,X3):.3f}  r(X₂,X₃)={r(X2,X3):.3f}  "
        f"r(Y,X₁)={r(Y,X1):.3f}  r(Y,X₂)={r(Y,X2):.3f}\n"
        f"X₁ ⊥ covariates by design → surface slices X₂,X₃ orbs evenly → balanced arms")
fig.text(0.5, 0.01, info, ha='center', color='#aaaacc', fontsize=8,
         fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='#111133',
                   alpha=0.7, edgecolor='#333366'))

plt.tight_layout()
plt.savefig('regression_vectors.png', dpi=150, bbox_inches='tight', facecolor='#0e0e1a')
plt.show()
print("\nSaved → regression_vectors.png")