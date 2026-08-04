"""
RCT Regression — Orb Plot  (n=100)
===================================

VARIABLES (5 orbs):
  X1 = had internship (0/1)
  X2 = GPA
  X3 = parental income
  Y  = job score (outcome)
  Ŷ  = OLS fitted values

ORB SIZE = partial R² of that variable (for X1/X2/X3) or R² (for Ŷ).
  Partial R² is unit-free: it measures how much of Y's variance each
  variable uniquely explains after controlling for the others.
  All X orbs overlap with Y's orb — that's what correlation looks like.
  More overlap = stronger relationship with Y.

ORB POSITION = direction in PCA space (biplot).
  Angle between two arrows ≈ arccos(correlation between those variables).
  Near-parallel arrows = highly correlated.
  Near-perpendicular arrows = uncorrelated.

RANDOMIZATION SURFACE
  Fit to pass through all 5 orb centers — this is what geometric balance
  looks like. Each orb is half in Treatment and half in Control.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

rng = np.random.default_rng(42)
n   = 100

# ── 1. BASELINE DATA (n=100) ──────────────────────────────────────────────────
GPA             = rng.normal(3.3, 0.4, n).clip(2.0, 4.0)
parental_income = rng.normal(65,  25,  n).clip(15, 150)
# Internship probability increases with parental income (realistic: wealthier
# families have more connections). Creates visible multicollinearity with X3.
p_internship = 1 / (1 + np.exp(-(parental_income - 65) / 18))
internship   = (rng.random(n) < p_internship).astype(float)

def r(a, b): return np.corrcoef(a, b)[0, 1]

# ── 2. OLS: Y ~ internship + GPA + parental_income ───────────────────────────
noise     = rng.normal(0, 15, n)
job_score = -52.4 + 24.8*internship + 29.1*GPA + 0.08*parental_income + noise
X_mat     = np.column_stack([np.ones(n), internship, GPA, parental_income])
b_hat     = np.linalg.lstsq(X_mat, job_score, rcond=None)[0]
Y_hat     = X_mat @ b_hat
R2        = 1 - np.sum((job_score-Y_hat)**2) / np.sum((job_score-job_score.mean())**2)

# Partial R² for each predictor (annotation only — not used for orb size)
def partial_r2(y, X_full, drop_col):
    X_red = np.delete(X_full, drop_col, axis=1)
    b_red = np.linalg.lstsq(X_red, y, rcond=None)[0]
    ss_res_red = np.sum((y - X_red @ b_red)**2)
    ss_res_full = np.sum((y - X_full @ np.linalg.lstsq(X_full, y, rcond=None)[0])**2)
    return (ss_res_red - ss_res_full) / ss_res_red

pR2 = {
    'X1': partial_r2(job_score, X_mat, 1),
    'X2': partial_r2(job_score, X_mat, 2),
    'X3': partial_r2(job_score, X_mat, 3),
}

print(f"OLS (n={n}):  b1(internship)={b_hat[1]:.1f}  b2(GPA)={b_hat[2]:.1f}  "
      f"b3(income)={b_hat[3]:.2f}  R²={R2:.3f}")
print(f"Partial R²:  internship={pR2['X1']:.3f}  GPA={pR2['X2']:.3f}  "
      f"income={pR2['X3']:.3f}")

# ── 3. MATCHED-PAIR RANDOMIZATION — MENTORSHIP RCT ───────────────────────────
# Sort 100 students by PC1 of all baseline covariates, pair consecutively,
# coin-flip within each pair → exactly 50T / 50C, balanced on everything.
cov_std = np.column_stack([
    (internship      - internship.mean())      / (internship.std() + 1e-9),
    (GPA             - GPA.mean())             / GPA.std(),
    (parental_income - parental_income.mean()) / parental_income.std(),
    (job_score       - job_score.mean())       / job_score.std(),
])
_, _, Vt_cov = np.linalg.svd(cov_std, full_matrices=False)
pc1        = cov_std @ Vt_cov[0]
sorted_idx = np.argsort(pc1)
pairs      = [(sorted_idx[i], sorted_idx[i+1]) for i in range(0, n, 2)]
mentorship = np.zeros(n, dtype=float)
for a, b in pairs:
    if rng.random() < 0.5:
        mentorship[a] = 1
    else:
        mentorship[b] = 1

print(f"\nMentorship balance (n={n}):")
for label, var in [('internship', internship), ('GPA', GPA),
                   ('income', parental_income), ('job score', job_score)]:
    print(f"  r(mentorship, {label:10s}) = {r(mentorship, var):.3f}")

# ── 4. BIPLOT SVD ─────────────────────────────────────────────────────────────
X_data = np.column_stack([internship, GPA, parental_income, job_score, Y_hat])
X_std  = (X_data - X_data.mean(axis=0)) / X_data.std(axis=0)
U_b, S_b, Vt_b = np.linalg.svd(X_std, full_matrices=False)
var_exp = (S_b**2) / np.sum(S_b**2)
print(f"\nBiplot: 3 PCs = {sum(var_exp[:3]):.1%} of structure")

# Variable arrow positions (all normalized to same display length L)
L = 3.5
var_raw  = Vt_b[:3, :].T                          # (5, 3)
var_norm = max(np.linalg.norm(v) for v in var_raw)
names    = ['X1','X2','X3','Y','Yh']
var_pts  = {nm: L * var_raw[i] / var_norm for i, nm in enumerate(names)}
p1=var_pts['X1']; p2=var_pts['X2']; p3=var_pts['X3']
pY=var_pts['Y'];  pYh=var_pts['Yh']

# ── 5. ORB SIZES = partial R² (unit-free importance measure) ─────────────────
# partial R² measures how much of Y's variance each variable uniquely explains.
# R² of the full model is used for Ŷ.
# All radii are set large enough that every X orb overlaps with Y's orb,
# because every X in the regression must correlate with Y to be included.
#
# Minimum radius guarantee: for orbs at tips of arrows (length L) with angle θ
# between them, the tips are distance 2L·sin(θ/2) apart. For overlap we need
# R_i + R_Y > 2L·sin(θ/2). We compute this for each variable vs Y.

importance = {
    'X1': pR2['X1'],
    'X2': pR2['X2'],
    'X3': pR2['X3'],
    'Y' : 1.0,           # outcome itself
    'Yh': R2,            # fitted values explain R² of Y
}

# Scale importance values to a radius range [R_base, R_top]
R_base, R_top = 1.2, 2.8
imp_vals = np.array([importance[nm] for nm in names])
R_orbs   = R_base + (imp_vals / imp_vals.max()) * (R_top - R_base)

# Enforce overlap 1: each X orb must overlap Y's orb
R_Y = R_orbs[names.index('Y')]
for i, nm in enumerate(names):
    if nm in ('Y', 'Yh'):
        continue
    dist_to_Y = float(np.linalg.norm(var_pts[nm] - pY))
    min_r = max(0.5, dist_to_Y - R_Y + 0.4)
    if R_orbs[i] < min_r:
        R_orbs[i] = min_r

# Enforce overlap 2: correlated X pairs must also overlap each other.
# Multicollinearity (X-X correlation) should be visible as orb intersection.
x_names = ['X1', 'X2', 'X3']
x_vars  = {'X1': internship, 'X2': GPA, 'X3': parental_income}
OVERLAP_THRESHOLD = 0.25   # show overlap for |r| > this
for ni in x_names:
    for nj in x_names:
        if nj <= ni:
            continue
        rij = abs(r(x_vars[ni], x_vars[nj]))
        if rij > OVERLAP_THRESHOLD:
            dist_ij      = float(np.linalg.norm(var_pts[ni] - var_pts[nj]))
            needed_sum   = dist_ij + 0.3 + 0.5 * rij   # more correlation → more overlap
            current_sum  = R_orbs[names.index(ni)] + R_orbs[names.index(nj)]
            if current_sum < needed_sum:
                bump = (needed_sum - current_sum) / 2   # split deficit evenly
                R_orbs[names.index(ni)] += bump
                R_orbs[names.index(nj)] += bump

orb_r = dict(zip(names, R_orbs))
print("\nOrb radii (partial R² + overlap guarantee):")
for nm in names:
    print(f"  {nm}: importance={importance[nm]:.3f}  →  radius={orb_r[nm]:.2f}")

# ── 6. RANDOMIZATION SURFACE ──────────────────────────────────────────────────
# Best-fit plane through all 5 orb centers — the plane that passes through
# as much of each orb as possible. This is the geometric ideal for balance.
orb_centers  = np.array([p1, p2, p3, pY, pYh])
orb_centroid = orb_centers.mean(axis=0)
centered_pts = orb_centers - orb_centroid
_, _, Vt_plane = np.linalg.svd(centered_pts)
norm_vec = Vt_plane[-1] / np.linalg.norm(Vt_plane[-1])

arb = np.array([0,1,0]) if abs(norm_vec[1]) < 0.9 else np.array([1,0,0])
f1  = arb - np.dot(arb, norm_vec)*norm_vec;  f1 /= np.linalg.norm(f1)
f2  = np.cross(norm_vec, f1);                f2 /= np.linalg.norm(f2)

disc_r     = R_top * 3.5
rhos, phis = np.linspace(0, disc_r, 35), np.linspace(0, 2*np.pi, 70)
RHO, PHI   = np.meshgrid(rhos, phis)
warp       = 0.3 * np.cos(2*PHI) * (RHO/disc_r)
mid        = orb_centroid
SX = mid[0] + RHO*(np.cos(PHI)*f1[0]+np.sin(PHI)*f2[0]) + warp*norm_vec[0]
SY = mid[1] + RHO*(np.cos(PHI)*f1[1]+np.sin(PHI)*f2[1]) + warp*norm_vec[1]
SZ = mid[2] + RHO*(np.cos(PHI)*f1[2]+np.sin(PHI)*f2[2]) + warp*norm_vec[2]

# ── 7. PLOT ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(12, 10), facecolor='#0e0e1a')
ax  = fig.add_subplot(111, projection='3d', facecolor='#0e0e1a')

COLORS = {'X1':'#4fc3f7', 'X2':'#81c784', 'X3':'#ce93d8', 'Y':'#fff176', 'Yh':'#ffb74d'}

def draw_orb(centre, color, radius, alpha=0.13):
    u, w = np.mgrid[0:2*np.pi:40j, 0:np.pi:30j]
    ax.plot_surface(centre[0]+radius*np.cos(u)*np.sin(w),
                    centre[1]+radius*np.sin(u)*np.sin(w),
                    centre[2]+radius*np.cos(w),
                    color=color, alpha=alpha, linewidth=0)

def draw_arrow(tip, color, lw=2.8, ls='-'):
    ax.quiver(0,0,0, tip[0],tip[1],tip[2], color=color,
              linewidth=lw, linestyle=ls, arrow_length_ratio=0.12)

# Draw orbs (size = std dev) and arrows
for nm, pt in var_pts.items():
    ls = '--' if nm == 'Yh' else '-'
    draw_orb(pt, COLORS[nm], orb_r[nm])
    draw_arrow(pt, COLORS[nm], ls=ls)

# Residual  e = Y − Ŷ
res = pY - pYh
ax.quiver(*pYh, *res, color='#ef5350', lw=1.5, linestyle=':', arrow_length_ratio=0.2)

# Randomization surface
ax.plot_surface(SX, SY, SZ, alpha=0.18, color='#b0bec5', linewidth=0)
phi_r = np.linspace(0, 2*np.pi, 150)
ring  = (mid + disc_r*(np.outer(np.cos(phi_r),f1) + np.outer(np.sin(phi_r),f2))
         + np.outer(0.3*np.cos(2*phi_r), norm_vec))
ax.plot(ring[:,0], ring[:,1], ring[:,2], color='#90a4ae', lw=1.8, alpha=0.9)

ax.text(*(mid + norm_vec*R_top*1.8 + f1*disc_r*0.55),
        'MENTORSHIP\n(Treatment)', color='#80cbc4', fontsize=9,
        fontfamily='monospace', fontweight='bold')
ax.text(*(mid - norm_vec*R_top*1.8 + f1*disc_r*0.55),
        'NO MENTORSHIP\n(Control)', color='#ef9a9a', fontsize=9,
        fontfamily='monospace', fontweight='bold')

# Labels — show std dev and partial R² where available
off = 0.25
ax.text(*(p1+off), f'X₁  Internship\n    pR²={pR2["X1"]:.2f}',
        color='#4fc3f7', fontsize=9, fontweight='bold', fontfamily='monospace')
ax.text(*(p2+off), f'X₂  GPA\n    pR²={pR2["X2"]:.2f}',
        color='#81c784', fontsize=9, fontweight='bold', fontfamily='monospace')
ax.text(*(p3+off), f'X₃  Parental $\n    pR²={pR2["X3"]:.2f}',
        color='#ce93d8', fontsize=9, fontweight='bold', fontfamily='monospace')
ax.text(*(pY+off), f'Y   Job Score',
        color='#fff176', fontsize=9, fontweight='bold', fontfamily='monospace')
ax.text(*(pYh - np.array([0,0,0.8])), f'Ŷ   fitted  (R²={R2:.2f})',
        color='#ffb74d', fontsize=8.5, fontfamily='monospace')
ax.text(*(pYh + res*0.55), 'e', color='#ef5350', fontsize=9, fontfamily='monospace')

# Styling — force equal axis ranges so spheres look spherical
max_val = max(abs(c) for pt in var_pts.values() for c in pt) + max(orb_r.values())
ax.set_xlim(-max_val, max_val)
ax.set_ylim(-max_val, max_val)
ax.set_zlim(-max_val, max_val)
ax.set_box_aspect([1, 1, 1])
ax.set_xlabel('PC 1', color='#444', labelpad=6)
ax.set_ylabel('PC 2', color='#444', labelpad=6)
ax.set_zlabel('PC 3', color='#444', labelpad=6)
ax.tick_params(colors='#333')
for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
    pane.fill = False; pane.set_edgecolor('#1a1a33')
ax.grid(True, color='#1a1a33', linewidth=0.5)
ax.set_title(
    f'Mentorship RCT  (n={n},  R²={R2:.3f},  3 PCs={sum(var_exp[:3]):.0%})\n'
    f'Orb size = std dev  ·  Labels show partial R²  ·  Surface = randomization boundary',
    color='white', fontsize=10, pad=12)

info = (f"r(mentorship, internship)={r(mentorship,internship):.3f}  "
        f"r(mentorship, GPA)={r(mentorship,GPA):.3f}  "
        f"r(mentorship, income)={r(mentorship,parental_income):.3f}  "
        f"r(mentorship, Y)={r(mentorship,job_score):.3f}")
fig.text(0.5, 0.01, info, ha='center', color='#aaaacc', fontsize=8,
         fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='#111133', alpha=0.7, edgecolor='#333366'))

plt.tight_layout()
plt.savefig('regression_vectors.png', dpi=150, bbox_inches='tight', facecolor='#0e0e1a')
plt.show()
print("Saved → regression_vectors.png")
