"""
Visualise Fourier / harmonic representations of closed surfaces.

Left  : genus-0 closed surface via spherical-harmonic expansion
        r(theta, phi) = sum_{l,m} a_{lm} Y_{lm}(theta, phi)
        x = r * sin(theta)cos(phi), y = r*sin(theta)sin(phi), z = r*cos(theta)
        -> star-shaped closed surface (topological sphere).

Right : genus-1 closed surface via two-variable Fourier series on the torus
        X(u,v) = sum_{k,l} c_{kl} e^{i(ku+lv)},  (u,v) in [0,2pi)^2
        with a torus base + a few low-frequency modes that perturb it
        while keeping it closed (periodic in both u and v).

Each panel shows the incremental reconstruction as more harmonics / Fourier
modes are added, demonstrating the multiscale nature of the representation.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import sph_harm_y
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

OUT = r"D:\Code\hktex\latex\spline_tex_draft\fourier_closed_surface.png"

# ----------------------------------------------------------------------
# 1. Genus-0: spherical-harmonic closed surface
# ----------------------------------------------------------------------
TH, PH = np.meshgrid(np.linspace(0, np.pi, 220),
                     np.linspace(0, 2*np.pi, 440), indexing='ij')

# Hand-picked low-order coefficients (l, m, a_lm, label)
coeffs = [
    (0,  0, 1.00, r"$Y_{0}^{0}$ (sphere)"),
    (2,  0, 0.35, r"$+Y_{2}^{0}$ (oblate)"),
    (3,  2, 0.18, r"$+Y_{3}^{2}$"),
    (4,  0, 0.12, r"$+Y_{4}^{0}$"),
    (4,  4, 0.10, r"$+Y_{4}^{4}$ (detail)"),
]

def r_of(coeffs_subset):
    r = np.zeros_like(TH)
    for l, m, a, _ in coeffs_subset:
        # sph_harm_y(S, theta, phi) returns complex; S = array of (l,m)
        Y = sph_harm_y(np.array([l]), np.array([m]), TH, PH)[0, 0]
        # add a*Re(Y) and an equal-amplitude imaginary part rotated by phase
        r = r + a * np.real(Y) + 0.6*a * np.imag(Y)
    return 1.0 + r

def to_cart(r):
    x = r * np.sin(TH) * np.cos(PH)
    y = r * np.sin(TH) * np.sin(PH)
    z = r * np.cos(TH)
    return x, y, z

# ----------------------------------------------------------------------
# 2. Genus-1: two-variable Fourier series on the torus
# ----------------------------------------------------------------------
R0, r0 = 1.6, 0.55           # major / minor radius of base torus
U, V = np.meshgrid(np.linspace(0, 2*np.pi, 240),
                   np.linspace(0, 2*np.pi, 240), indexing='ij')

# Each mode perturbs the torus. (k,l,amp_x,amp_y,amp_z, label)
# Only low-frequency modes -> stays a smooth genus-1 closed surface.
tmodes = [
    (0, 0, 0, 0, 0, r"base torus"),
    (1, 0, 0.20, 0.0, 0.0, r"$+e^{\pm iu}$ (bend)"),
    (2, 0, 0.06, 0.06, 0.0, r"$+e^{\pm 2iu}$"),
    (0, 3, 0.0, 0.0, 0.10, r"$+e^{\pm 3iv}$ (ripple)"),
    (2, 1, 0.05, 0.05, 0.05, r"$+e^{\pm(2u+v)}$ (twist)"),
]

def torus_of(modes_subset):
    # base torus centred at origin, axis = z
    x = (R0 + r0*np.cos(V)) * np.cos(U)
    y = (R0 + r0*np.cos(V)) * np.sin(U)
    z = r0 * np.sin(V)
    for k, l, ax, ay, az, _ in modes_subset[1:]:   # skip base entry
        # real cosine + sine terms => equivalent to e^{i(ku+lv)} pairs
        cu, su = np.cos(k*U), np.sin(k*U)
        cv, sv = np.cos(l*V), np.sin(l*V)
        x = x + ax * (cu*cv - su*sv)
        y = y + ay * (cu*cv + su*sv)
        z = z + az * (cu*sv + su*cv)
    return x, y, z

# ----------------------------------------------------------------------
# 3. Plot grid: rows = selected accumulation stages, cols = (genus0, genus1)
# Show 3 representative stages: base, intermediate, full reconstruction.
# (5 rows overflowed the single-column text height by ~88pt.)
# ----------------------------------------------------------------------
stage_idx = [0, 2, 4]   # indices into coeffs / tmodes -> base, mid, full
nrows = len(stage_idx)
fig = plt.figure(figsize=(8.4, 3.0*nrows))

for row, i in enumerate(stage_idx):
    # genus 0
    axL = fig.add_subplot(nrows, 2, 2*row+1, projection='3d')
    sub = coeffs[:i+1]
    x, y, z = to_cart(r_of(sub))
    axL.plot_surface(x, y, z, color='#c8d8ff', edgecolor='none',
                     alpha=0.9, linewidth=0, antialiased=True, shade=True)
    axL.set_title(sub[-1][3], fontsize=9, pad=2)
    axL.set_xlim(-1.7,1.7); axL.set_ylim(-1.7,1.7); axL.set_zlim(-1.7,1.7)
    axL.set_box_aspect((1,1,1))
    axL.set_xticks([]); axL.set_yticks([]); axL.set_zticks([])
    axL.view_init(elev=22, azim=-55)

    # genus 1
    axR = fig.add_subplot(nrows, 2, 2*row+2, projection='3d')
    sub = tmodes[:i+1]
    x, y, z = torus_of(sub)
    axR.plot_surface(x, y, z, color='#ffd8c8', edgecolor='none',
                     alpha=0.9, linewidth=0, antialiased=True, shade=True)
    axR.set_title(sub[-1][5], fontsize=9, pad=2)
    axR.set_xlim(-2.6,2.6); axR.set_ylim(-2.6,2.6); axR.set_zlim(-1.2,1.2)
    axR.set_box_aspect((2.6,2.6,1.2))
    axR.set_xticks([]); axR.set_yticks([]); axR.set_zticks([])
    axR.view_init(elev=24, azim=-55)

fig.suptitle("Fourier / harmonic representations of closed surfaces\n"
             "Left: genus-0  $r(\\theta,\\varphi)=\\sum a_{\\ell m}Y_{\\ell m}$   "
             "Right: genus-1  $\\mathbf{X}(u,v)=\\sum \\mathbf{c}_{kl}\\,e^{i(ku+lv)}$",
             fontsize=11, y=0.995)
plt.tight_layout(rect=(0,0,1,0.95))
plt.savefig(OUT, dpi=150, bbox_inches='tight')
print("saved:", OUT)
