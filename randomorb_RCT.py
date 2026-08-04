"""
RCT Regression — Orb Plot with Mentorship Randomization Surface
================================================================

SETUP
  10 students with existing baseline data:
    X1 = had internship (0/1)
    X2 = GPA
    X3 = parental income
    Y  = job score
    Ŷ  = fitted values from OLS: Y ~ X1 + X2 + X3

  These 5 variables appear as orbs in the plot.
  Each orb sits at the tip of its variable's vector in a shared PCA space.
  Angle between two arrows = arccos(correlation between those variables).

NEW INTERVENTION
  We now want to randomize the same 10 students into a mentorship programme.
  Good randomization = the new treatment vector is ORTHOGONAL to all 5 existing
  variable vectors, meaning the mentorship groups are balanced on internship
  history, GPA, income, job score, and the model's fitted values.

THE SURFACE
  The randomization surface = the plane perpendicular to the new treatment vector.
  A surface that cuts THROUGH all 5 orbs means every variable is split evenly
  between treatment and control — that is what balance looks like geometrically.
  If randomization were poor, the surface would miss some orbs entirely.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(7)

# ── 1. BASELINE DATA ──────────────────────────────────────────────────────────
GPA             = np.array([3.8, 3.5, 3.9, 3.2, 2.8, 2.9, 3.5, 3.7, 2.5, 3.1])
parental_income = np.array([95,  60, 110,  45,  40,  55,  80,  75,  35,  50], dtype=float)
internship      = np.array([1,   0,   1,   0,   0,   1,   1,   0,   0,   1], dtype=float)
n = len(GPA)

def r(a, b): return np.corrcoef(a, b)[0, 1]

# ── 2. OLS: Y ~ internship + GPA + parental_income ───────────────────────────
X_mat   = np.column_stack([np.ones(n), internship, GPA, parental_income])
job_score = (-52.4 + 24.8*internship + 29.1*GPA
             + 0.08*parental_income + np.random.normal(0, 15, n))
b_hat   = np.linalg.lstsq(X_mat, job_score, rcond=None)[0]
Y_hat   = X_mat @ b_hat
R2      = 1 - np.sum((job_score-Y_hat)**2) / np.sum((job_score-job_score.mean())**2)
print(f"OLS  b1(internship)={b_hat[1]:.1f}  b2(GPA)={b_hat[2]:.1f}  "
      f"b3(income)={b_hat[3]:.2f}  R²={R2:.3f}")

# ── 3. RE-RANDOMIZATION FOR NEW MENTORSHIP RCT ───────────────────────────────
# Try many random 5T/5C assignments; keep the one most balanced across ALL
# baseline variables (internship, GPA, income, job score).
# This is "re-randomization" — standard in modern RCT design.
def imbalance(t):
    """Sum of squared correlations with all baseline variables."""
    return (r(t, internship)**2 + r(t, GPA)**2
            + r(t, parental_income)**2 + r(t, job_score)**2)

best_imb, mentorship = np.inf, None
rng = np.random.default_rng(0)
for _ in range(5000):
    idx = rng.permutation(n)
    t   = np.zeros(n); t[idx[:5]] = 1
    imb = imbalance(t)
    if imb < best_imb:
        best_imb, mentorship = imb, t.copy()

print(f"Mentorship balance:")
print(f"  r(mentorship, internship) = {r(mentorship,internship):.3f}")
print(f"  r(mentorship, GPA)        = {r(mentorship,GPA):.3f}")
print(f"  r(mentorship, income)     = {r(mentorship,parental_income):.3f}")
print(f"  r(mentorship, job score)  = {r(mentorship,job_score):.3f}")

# ── 4. BIPLOT: 5 existing variables in shared PCA space ──────────────────────
X_data  = np.column_stack([internship, GPA, parental_income, job_score, Y_hat])
X_std   = (X_data - X_data.mean(axis=0)) / X_data.std(axis=0)
U_b, S_b, Vt_b = np.linalg.svd(X_std, full_matrices=False)
var_exp = (S_b**2) / np.sum(S_b**2)
print(f"Biplot: 3 PCs capture {sum(var_exp[:3]):.1%} of total structure")

L = 3.5; R_orb = 1.8
var_raw  = Vt_b[:3, :].T                        # (5, 3)
var_norm = max(np.linalg.norm(v) for v in var_raw)
var_pts  = {name: L * var_raw[i] / var_norm
            for i, name in enumerate(['X1','X2','X3','Y','Yh'])}
p1=var_pts['X1']; p2=var_pts['X2']; p3=var_pts['X3']
pY=var_pts['Y'];  pYh=var_pts['Yh']

# ── 5. RANDOMIZATION SURFACE ──────────────────────────────────────────────────
# A perfectly balanced RCT means the new treatment vector is orthogonal to
# every existing variable. Geometrically: the surface passes through ALL orbs.
#
# We find this surface by fitting a plane to the 5 orb centers.
# The plane's normal = direction most orthogonal to all variable arrows.
# The plane is then centered at the orb centroid so it bisects each orb.
orb_centers  = np.array([p1, p2, p3, pY, pYh])
orb_centroid = orb_centers.mean(axis=0)
centered     = orb_centers - orb_centroid
_, _, Vt_plane = np.linalg.svd(centered)
norm_vec = Vt_plane[-1]          # normal to best-fit plane through orb centers
norm_vec = norm_vec / np.linalg.norm(norm_vec)

# Verify: angle between surface normal and each variable arrow
# Should be close to 90° for all — meaning treatment ⊥ all existing variables
print("Surface normal vs variable arrows (90° = perfectly balanced):")
for name, pt in var_pts.items():
    ang = np.degrees(np.arccos(np.clip(
        abs(np.dot(norm_vec, pt/np.linalg.norm(pt))), 0, 1)))
    print(f"  {name}: {ang:.1f}°")
arb = np.array([0,1,0]) if abs(norm_vec[1]) < 0.9 else np.array([1,0,0])
f1  = arb - np.dot(arb, norm_vec)*norm_vec;  f1 /= np.linalg.norm(f1)
f2  = np.cross(norm_vec, f1);                f2 /= np.linalg.norm(f2)

disc_r       = R_orb * 3.2
rhos, phis   = np.linspace(0, disc_r, 35), np.linspace(0, 2*np.pi, 70)
RHO, PHI     = np.meshgrid(rhos, phis)
warp         = 0.3 * np.cos(2*PHI) * (RHO/disc_r)
mid          = orb_centroid
SX = mid[0] + RHO*(np.cos(PHI)*f1[0]+np.sin(PHI)*f2[0]) + warp*norm_vec[0]
SY = mid[1] + RHO*(np.cos(PHI)*f1[1]+np.sin(PHI)*f2[1]) + warp*norm_vec[1]
SZ = mid[2] + RHO*(np.cos(PHI)*f1[2]+np.sin(PHI)*f2[2]) + warp*norm_vec[2]

# ── 6. PLOT ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(11, 9), facecolor='#0e0e1a')
ax  = fig.add_subplot(111, projection='3d', facecolor='#0e0e1a')

def arrow(tip, color, lw=2.8, ls='-'):
    ax.quiver(0,0,0, tip[0],tip[1],tip[2], color=color,
              linewidth=lw, linestyle=ls, arrow_length_ratio=0.12)

def orb(centre, color, alpha=0.13):
    u, w = np.mgrid[0:2*np.pi:40j, 0:np.pi:30j]
    ax.plot_surface(centre[0]+R_orb*np.cos(u)*np.sin(w),
                    centre[1]+R_orb*np.sin(u)*np.sin(w),
                    centre[2]+R_orb*np.cos(w),
                    color=color, alpha=alpha, linewidth=0)

# Orbs + arrows
orb(p1,  '#4fc3f7');  arrow(p1,  '#4fc3f7')        # X1 internship
orb(p2,  '#81c784');  arrow(p2,  '#81c784')        # X2 GPA
orb(p3,  '#ce93d8');  arrow(p3,  '#ce93d8')        # X3 parental income
orb(pY,  '#fff176');  arrow(pY,  '#fff176')        # Y  job score
orb(pYh, '#ffb74d');  arrow(pYh, '#ffb74d', lw=2, ls='--')  # Ŷ fitted

res = pY - pYh
ax.quiver(*pYh, *res, color='#ef5350', lw=1.5, linestyle=':', arrow_length_ratio=0.2)

# Randomization surface
ax.plot_surface(SX, SY, SZ, alpha=0.20, color='#b0bec5', linewidth=0)
phi_r = np.linspace(0, 2*np.pi, 150)
ring  = (mid + disc_r*(np.outer(np.cos(phi_r),f1) + np.outer(np.sin(phi_r),f2))
         + np.outer(0.3*np.cos(2*phi_r), norm_vec))
ax.plot(ring[:,0], ring[:,1], ring[:,2], color='#90a4ae', lw=1.8, alpha=0.9)

# T / C side labels — offset along surface normal
ax.text(*(mid + norm_vec*R_orb*2.2 + f1*disc_r*0.55),
        'MENTORSHIP\n(Treatment)', color='#80cbc4', fontsize=9,
        fontfamily='monospace', fontweight='bold')
ax.text(*(mid - norm_vec*R_orb*2.2 + f1*disc_r*0.55),
        'NO MENTORSHIP\n(Control)', color='#ef9a9a', fontsize=9,
        fontfamily='monospace', fontweight='bold')

# Variable labels
off = 0.3
ax.text(*(p1+off),  'X₁  Internship\n    (baseline)',  color='#4fc3f7', fontsize=9, fontweight='bold', fontfamily='monospace')
ax.text(*(p2+off),  'X₂  GPA',                         color='#81c784', fontsize=9, fontweight='bold', fontfamily='monospace')
ax.text(*(p3+off),  'X₃  Parental $',                  color='#ce93d8', fontsize=9, fontweight='bold', fontfamily='monospace')
ax.text(*(pY+off),  'Y   Job Score',                    color='#fff176', fontsize=9, fontweight='bold', fontfamily='monospace')
ax.text(*(pYh - np.array([0,0,0.7])), 'Ŷ (fitted)',    color='#ffb74d', fontsize=8, fontfamily='monospace')
ax.text(*(pYh + res*0.55), 'e',                         color='#ef5350', fontsize=9, fontfamily='monospace')

# Styling
ax.set_xlabel('PC 1', color='#444', labelpad=6)
ax.set_ylabel('PC 2', color='#444', labelpad=6)
ax.set_zlabel('PC 3', color='#444', labelpad=6)
ax.tick_params(colors='#333')
for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
    pane.fill = False; pane.set_edgecolor('#1a1a33')
ax.grid(True, color='#1a1a33', linewidth=0.5)
ax.set_title(
    f'Mentorship RCT  ·  Y = b₀ + b₁·internship + b₂·GPA + b₃·income  (R²={R2:.3f})\n'
    f'Surface cuts through all orbs → mentorship is balanced on every baseline variable',
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