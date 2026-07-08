# -*- coding: utf-8 -*-
"""
Periodic Homogenization GUI
===========================
Computes the 3x3 plane homogenized stiffness tensor Ch for 2D microstructures.

Microstructures supported:
  - EllipseCentered : centered rotated ellipse (matches apply_ellipse)
  - Lattice         : diagonal lattice struts
  - Biotruss        : quadratic Bezier curves (matches apply_bio_truss)

"""

from __future__ import annotations

import os
import csv
import time
import traceback
import math
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from numpy.linalg import solve as np_solve
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, filedialog, messagebox


# ============================================================
#  Constitutive matrix  (engineering Voigt, plane stress/strain)
# ============================================================

def build_C(E, nu, plane_stress=True):
    """
    3x3 isotropic plane constitutive matrix with engineering shear.
      plane_stress: lam = E*nu/(1-nu^2),  C = [[lam+2mu,lam,0],[lam,lam+2mu,0],[0,0,mu]]
      plane_strain: lam = E*nu/((1+nu)*(1-2nu))
    where mu = E/(2*(1+nu)).
    Voigt order: [sigma11, sigma22, sigma12],  strain = [eps11, eps22, gamma12]
    C[2,2] = mu  (not 2mu) because gamma12 = 2*eps12 is the engineering shear.
    """
    E, nu = float(E), float(nu)
    mu  = E / (2.0*(1.0+nu))
    lam = E*nu/(1.0-nu**2) if plane_stress else E*nu/((1.0+nu)*(1.0-2.0*nu))
    return np.array([[lam+2*mu, lam,      0.0],
                     [lam,      lam+2*mu, 0.0],
                     [0.0,      0.0,      mu ]])


# ============================================================
#  Q1 element  (bilinear quadrilateral, matches CPE4/CPS4)
# ============================================================

def _shape_Q1(xi, eta):
    """Bilinear Q1 shape functions N (4,) and derivatives DN (2,4) at (xi,eta)."""
    N  = 0.25*np.array([(1-xi)*(1-eta), (1+xi)*(1-eta),
                         (1+xi)*(1+eta), (1-xi)*(1+eta)])
    DN = 0.25*np.array([[-(1-eta), (1-eta), (1+eta), -(1+eta)],
                         [-(1-xi), -(1+xi), (1+xi),   (1-xi) ]])
    return N, DN

def _gauss_Q1():
    """2x2 Gauss quadrature for Q1 (4 points, weights=1)."""
    s = 1.0/math.sqrt(3.0)
    return np.array([[-s,-s],[s,-s],[s,s],[-s,s]]), np.ones(4)

def elem_KP(x_el, C, strain_id):
    """
    Element stiffness K (8x8) and force P (8x1) for one Q1 element.
    x_el  : (2,4) node coordinates  (columns = nodes, rows = x,y)
    C     : (3,3) constitutive matrix
    strain_id : 0=eps11, 1=eps22, 2=gamma12  (engineering shear unit load)

    P = int B^T C e_macro dOmega_e
    where e_macro[strain_id] = 1.0, all others = 0.

    """
    gauss_pts, gauss_w = _gauss_Q1()
    K = np.zeros((8, 8))
    P = np.zeros((8, 1))
    e_macro = np.zeros((3, 1))
    e_macro[strain_id, 0] = 1.0

    for r, w in zip(gauss_pts, gauss_w):
        N, DN = _shape_Q1(r[0], r[1])
        J     = x_el @ DN.T            # (2,2) Jacobian
        detJ  = np.linalg.det(J)
        dNdx  = np.linalg.inv(J) @ DN  # (2,4) shape function physical derivatives

        # B matrix (3x8): engineering-Voigt strain = B * u_el
        B = np.zeros((3, 8))
        for a in range(4):
            B[0, 2*a  ] = dNdx[0, a]   # eps11 = du/dx
            B[1, 2*a+1] = dNdx[1, a]   # eps22 = dv/dy
            B[2, 2*a  ] = dNdx[1, a]   # gamma12 = du/dy + dv/dx
            B[2, 2*a+1] = dNdx[0, a]

        K += B.T @ C @ B * detJ * w
        P += B.T @ (C @ e_macro) * detJ * w

    return K, P


# ============================================================
#  Periodic homogenization 
# ============================================================

def homogenize_numpy(m, elem_ids, E_list, nu_list, plane_stress=True):
    """
    Compute the 3x3 homogenized stiffness tensor Ch using the energy formula.

    Parameters
    ----------
    m         : int   - mesh is m x m elements
    elem_ids  : array - length m*m, elem_ids[j*m+i] = material at column i, row j
                        0 -> E_list[0] (matrix/outside), 1 -> E_list[1] (inclusion/inside)
    E_list    : sequence of Young's moduli for each material index
    nu_list   : sequence of Poisson's ratios
    plane_stress : bool (default True)

    Returns
    -------
    Ch : (3,3) ndarray
         [[C1111, C1122, C1112],
          [C1122, C2222, C2212],
          [C1112, C2212, C1212]]
    """
    m = int(m)
    elem_ids = np.asarray(elem_ids, dtype=int).ravel()
    assert elem_ids.size == m*m

    Ce    = [build_C(E, nu, plane_stress) for E, nu in zip(E_list, nu_list)]
    d     = 1.0 / m
    n_node = (m+1)**2
    n_pre  = 2        # two prescribed (pinned) DOFs at the gauge node

    # --- Node coordinates: node j*(m+1)+i is at (i*d, j*d) ---
    coords = np.array([[i*d, j*d]
                        for j in range(m+1)
                        for i in range(m+1)], dtype=float)

    # --- Element connectivity: elem j*m+i has CCW nodes [b, b+1, b+m+2, b+m+1] ---
    elem_nodes = np.array([
        [j*(m+1)+i, j*(m+1)+i+1, (j+1)*(m+1)+i+1, (j+1)*(m+1)+i]
        for j in range(m) for i in range(m)
    ], dtype=int)
    # Verify: elem_nodes[j*m+i] is the element in column i, row j  OK

    # --- Equation numbering (exact replica of assign_periodic_BC2d) ---
    eq_num = np.zeros((2, n_node), dtype=float)
    npn    = m + 2        # non_periodic_node = m1+2, at coords (d, d)
    eq_num[0, npn] = -1
    eq_num[1, npn] = -2
    row = 0
    for i in range(n_node):
        for j in range(2):
            if eq_num[j, i] == 0:
                row += 1
                eq_num[j, i] = row
    n_dof = 2*m*m - 2    # number of free DOFs

    # Periodicity: top row = bottom row
    eq_num[:, m*(m+1):(m+1)*(m+1)] = eq_num[:, 0:m+1]
    # Periodicity: right column = left column
    for k in range(m+1):
        eq_num[:, k*(m+1)+m] = eq_num[:, k*(m+1)]
    # Shift equation numbers (exact replica)
    sub = []
    for sf in range(2, 2*m, 2):
        sub += [sf] * (m+1)
    sub = np.array(sub)
    eq_num[0][m+1 : m*(m+1)] -= sub
    eq_num[1][m+1 : m*(m+1)] -= sub
    eq_num[0][npn] += 2
    eq_num[1][npn] += 2

    # Global equation numbers (0-indexed) for assembly
    gq = eq_num.copy()
    for idx in range(gq.size):
        if gq.flat[idx] < 0:
            gq.flat[idx] += 1
    gq = (gq - gq.min()).T.astype(int)   # shape (n_node, 2)

    # --- Precompute K and P per material type (uniform mesh: geometry identical) ---
    x_ref = coords[elem_nodes[0]].T   # (2,4) reference element coords
    Km = [elem_KP(x_ref, Ce[mid], 0)[0]        for mid in range(len(E_list))]
    Pm = [[elem_KP(x_ref, Ce[mid], s)[1]
            for mid in range(len(E_list))]
           for s in range(3)]

    # --- Solve for periodic correctors and accumulate Ch ---
    total_size = n_pre + n_dof
    UURs  = []    # list of (2, n_node) displacement arrays, one per strain case
    chi0s = []    # list of dicts {mat_id: (8,1) array}, one per strain case
    KFF   = None  # assembled once, reused for all 3 RHS

    for sid in range(3):
        PF_full = np.zeros(total_size)
        if sid == 0:
            Kdata, Krow, Kcol = [], [], []

        for ei in range(m*m):
            mat    = int(elem_ids[ei])
            K_el   = Km[mat]
            P_el   = Pm[sid][mat].ravel()
            cn     = elem_nodes[ei]
            for a in range(4):
                for da in range(2):
                    ga = gq[cn[a], da]
                    PF_full[ga] += P_el[2*a + da]
                    if sid == 0:
                        for b in range(4):
                            for db in range(2):
                                gb = gq[cn[b], db]
                                Krow.append(ga)
                                Kcol.append(gb)
                                Kdata.append(K_el[2*a+da, 2*b+db])

        if sid == 0:
            KUR = scipy.sparse.csc_matrix(
                (Kdata, (Krow, Kcol)), shape=(total_size, total_size))
            KFF = KUR[n_pre:, n_pre:].tocsc()

        UF = scipy.sparse.linalg.spsolve(KFF, PF_full[n_pre:])

        UUR = np.zeros((2, n_node))
        for i_node in range(n_node):
            for i_dof in range(2):
                r = int(eq_num[i_dof, i_node])
                if r > 0:
                    UUR[i_dof, i_node] = UF[r - 1]
                # prescribed DOFs (gauge node) stay 0  OK
        UURs.append(UUR)

        # chi0[mat] = K_e^{-1} P_e  (element-local linear macro-displacement)
        # Use safe_solve: K + small_perturb * I  
        c0 = {}
        for mid in range(len(E_list)):
            K_reg = Km[mid] + 1e-12 * max(1.0, E_list[mid]) * np.eye(8)
            c0[mid] = np_solve(K_reg, Pm[sid][mid])
        chi0s.append(c0)

    # --- Energy formula for Ch ---
    Ch = np.zeros((3, 3))
    for ei in range(m*m):
        mat = int(elem_ids[ei])
        K_el = Km[mat]
        cn   = elem_nodes[ei]
        for ii in range(3):
            c0i = chi0s[ii][mat].ravel()
            chi_i = np.array([UURs[ii][dd, cn[a]]
                               for a in range(4) for dd in range(2)])
            Di = c0i - chi_i
            for jj in range(3):
                c0j = chi0s[jj][mat].ravel()
                chi_j = np.array([UURs[jj][dd, cn[a]]
                                   for a in range(4) for dd in range(2)])
                Ch[ii, jj] += Di @ (K_el @ (c0j - chi_j))

    Ch /= (m*m * d*d)   # normalize by |Y| = m*m*d*d = 1  (unit cell)
    return Ch


# ============================================================
#  Microstructure generators 
# ============================================================

class BaseMicrostructure:
    def __init__(self, m):
        self.m = int(m)

    def feature_names(self):       raise NotImplementedError
    def default_fixed_features(self): raise NotImplementedError
    def generate_random_input(self): raise NotImplementedError
    def generate_ids(self, *args): raise NotImplementedError

    def _centers(self):
        """
        Element centroids: elem j*m+i  ->  x=(i+0.5)/m, y=(j+0.5)/m
        """
        m = self.m
        i_idx = np.tile(np.arange(m), m)         # x index (fast)
        j_idx = np.repeat(np.arange(m), m)        # y index (slow)
        return (i_idx + 0.5)/m, (j_idx + 0.5)/m


class EllipseCentered(BaseMicrostructure):
    """
    Centered rotated ellipse. Replicates GeometryBuilder.apply_ellipse().
    elem_id=0 outside (matrix), elem_id=1 inside (inclusion, hole_id=1).
    """
    def feature_names(self):        return ["a", "b", "theta"]
    def default_fixed_features(self): return [0.1, 0.2, 0.4]

    def generate_random_input(self):
        return (np.random.uniform(0.06, 0.45),
                np.random.uniform(0.06, 0.45),
                np.random.uniform(0.0, math.pi))

    def generate_ids(self, a, b, theta):
        x, y = self._centers()
        a, b, theta = float(a), float(b), float(theta)
        h, k = 0.5, 0.5
        c, s = math.cos(theta), math.sin(theta)
        tx = (((x-h)*c + (y-k)*s) / a)**2
        ty = (((x-h)*s - (y-k)*c) / b)**2
        ids = np.zeros(self.m**2, dtype=int)
        ids[(tx + ty - 1.0) < 0.0] = 1
        return ids


class Lattice(BaseMicrostructure):
    """Diagonal lattice struts. elem_id=1 inside struts."""
    def feature_names(self):        return ["t"]
    def default_fixed_features(self): return [0.08]
    def generate_random_input(self): return (np.random.uniform(0.04, 0.45),)

    def generate_ids(self, t):
        x, y = self._centers()
        t = float(t)
        ids = np.zeros(self.m**2, dtype=int)
        mask = ((np.abs(y + 0.5*x - 1.0)/math.sqrt(1.25) <= t)
                | (np.abs(y - 0.5*x)/math.sqrt(1.25) <= t)
                | (np.abs(x) <= t)
                | (np.abs(x - 1.0) <= t))
        ids[mask] = 1
        return ids


class Biotruss(BaseMicrostructure):
    """
    Four quadratic Bezier curves defining void regions at each edge.

    Parameters (10 total):
      a        : half-width of bottom/top arch span, centered at x=0.5
      b        : half-width of left/right arch span, centered at y=0.5
      p1_x, p1_y : control-point for bottom void arch (curve 1)
      p2_x, p2_y : control-point for right void arch  (curve 2)
      p3_x, p3_y : control-point for top void arch    (curve 3)
      p4_x, p4_y : control-point for left void arch   (curve 4)

    p_x values control lateral position of the arch apex.
    p_y (or p_x for vertical curves) controls the depth of the void bite.

    Void convention (hole_id=1):
      elem_id = 0  ->  solid background   ->  E_list[0]
      elem_id = 1  ->  void / opening     ->  E_list[1]  (set E_list[1] << E_list[0])
    """
    def feature_names(self):
        return ["a", "b",
                "p1_x", "p1_y",
                "p2_x", "p2_y",
                "p3_x", "p3_y",
                "p4_x", "p4_y"]

    def default_fixed_features(self):
        # a=b=0.5  (arch spans half the cell width)
        # p_x=0.5  (arch apex centred laterally)
        # p1_y=0.75, p3_y=0.25 -> bottom/top void depth ~0.75 of half-cell
        # p2_x=0.75, p4_x=0.25 -> right/left void depth ~0.75 of half-cell
        return [0.5, 0.5,
                0.5, 0.75,
                0.75, 0.5,
                0.5, 0.25,
                0.25, 0.5]

    def generate_random_input(self):
        #   a = b = 0.5  (fixed arch spans)
        #   all p values: uniform(0.01, 0.6)  
        a  = 0.5
        b  = 0.5
        p1_x = np.random.uniform(0.01, 0.6)
        p1_y = np.random.uniform(0.01, 0.6)
        p2_x = np.random.uniform(0.01, 0.6)
        p2_y = np.random.uniform(0.01, 0.6)
        p3_x = np.random.uniform(0.01, 0.6)
        p3_y = np.random.uniform(0.01, 0.6)
        p4_x = np.random.uniform(0.01, 0.6)
        p4_y = np.random.uniform(0.01, 0.6)
        return a, b, p1_x, p1_y, p2_x, p2_y, p3_x, p3_y, p4_x, p4_y

    @staticmethod
    def _bezier(px, py, m):
        """
        Quadratic Bezier curve sampled at m+1 points (t=0 to t=1).

        Returns lists, not numpy arrays.
        """
        step_inc = 1.0 / m
        t = np.arange(0, 1 + step_inc, step_inc)  
        x_vals = [(1-ti)**2*px[0] + 2*ti*(1-ti)*px[1] + ti**2*px[2] for ti in t]
        y_vals = [(1-ti)**2*py[0] + 2*ti*(1-ti)*py[1] + ti**2*py[2] for ti in t]
        return x_vals, y_vals

    def generate_ids(self, a, b, p1_x, p1_y, p2_x, p2_y, p3_x, p3_y, p4_x, p4_y):
        """
        The i variable is shared within the horizontal block (curves 1 and 3)
        and reset to 0 before the vertical block (curves 2 and 4), then shared
        within the vertical block.
        """
        m   = self.m
        ids = np.zeros(m * m, dtype=int)
        x_c, y_c = self._centers()


        px_1 = [0.5-a/2, 0.5-a/2+a*p1_x, 0.5+a/2];  py_1 = [0,   2*p1_y, 0  ]
        px_2 = [1,        1-2*p2_x,        1       ];  py_2 = [0.5-b/2, 0.5-b/2+b*p2_y, 0.5+b/2]
        px_3 = [0.5-a/2, 0.5-a/2+a*p3_x, 0.5+a/2];  py_3 = [1,   1-2*p3_y, 1]
        px_4 = [0,        2*p4_x,          0       ];  py_4 = [0.5-b/2, 0.5-b/2+b*p4_y, 0.5+b/2]

        x1, y1 = self._bezier(px_1, py_1, m)
        x2, y2 = self._bezier(px_2, py_2, m)
        x3, y3 = self._bezier(px_3, py_3, m)
        x4, y4 = self._bezier(px_4, py_4, m)

        for eid, (x, y) in enumerate(zip(x_c, y_c)):
            i         = 0     # shared across horizontal block 
            tolerance = 0

            # --- Horizontal curves (bottom arch 1, top arch 3) ---
            if x >= px_1[0] and x <= px_1[2]:
                if y <= p1_y:                        # bottom void region
                    while x >= x1[i] and i < len(x1)-1: i += 1
                    if y <= y1[i]: tolerance = -1
                if y >= (1 - p3_y):                  # top void region (i NOT reset)
                    while x >= x3[i] and i < len(x3)-1: i += 1
                    if y >= y3[i]: tolerance = -1

            # --- Vertical curves (right arch 2, left arch 4) ---
            i = 0             # reset before vertical block
            if y >= py_2[0] and y <= py_2[2]:
                if x >= (1 - p2_x):                  # right void region
                    while y >= y2[i] and i < len(y2)-1: i += 1
                    if x >= x2[i]: tolerance = -1
                if x <= p4_x:                        # left void region (i NOT reset)
                    while y >= y4[i] and i < len(y4)-1: i += 1
                    if x <= x4[i]: tolerance = -1

            if tolerance < 0:
                ids[eid] = 1   # void  (hole_id = 1)

        return ids


def get_microstructure(name, m):
    n = str(name).lower()
    if n in ("ellipsecentered","ellipsevoid","ellipseinclusion"):
        return EllipseCentered(m)
    if n == "lattice":   return Lattice(m)
    if n == "biotruss":  return Biotruss(m)
    raise ValueError(f"Unknown microstructure '{name}'.")


# ============================================================
#  Geometry plotting
# ============================================================

def plot_geometry(elem_ids, m, title="Geometry", cmap="binary",
                  save_path=None, show_nonblocking=True,
                  pause_time=0.25, close_after_show=True):
    arr = np.asarray(elem_ids).reshape((m, m))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(arr, cmap=cmap, origin="lower",
              vmin=int(arr.min()), vmax=int(arr.max()))
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xticks(np.arange(-0.5, m, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, m, 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.25)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show_nonblocking:
        plt.show(block=False); plt.pause(pause_time)
    else:
        plt.show()
    if close_after_show:
        plt.close(fig)


# ============================================================
#  Verification test
# ============================================================

def test_homogeneous(m=8, E=1.0, nu=0.3, plane_stress=True):
    """
    All elements = material 0 (solid). Must recover the single-material C exactly.
    Returns (Ch_computed, Ch_expected, abs_error).
    """
    ids = np.zeros(m*m, dtype=int)
    Ch  = homogenize_numpy(m, ids, [E, 1e-6], [nu, nu], plane_stress)
    Ce  = build_C(E, nu, plane_stress)
    return Ch, Ce, np.abs(Ch - Ce), np.max(np.abs(Ch - Ce)) / max(np.max(np.abs(Ce)), 1.0)


# ============================================================
#  Data generation
# ============================================================

def save_row(filename, row, mode="a"):
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    with open(filename, mode, newline="") as f:
        csv.writer(f).writerow(row)


def generate_dataset(params, log=print):
    ms_name  = params["microstructure_name"]
    m        = params["m"]
    num_data = params["num_data"]
    E0, E1   = params["E0"], params["E1"]
    nu0,nu1  = params["nu0"],params["nu1"]
    ps       = params["plane_stress"]
    seed     = params["seed"]
    out_dir  = params["output_dir"]
    rand     = params["random_geometry"]
    fixed    = params["fixed_features"]
    save_png = params["save_png"]
    show_plt = params["show_plots"]
    pause    = params["pause_time"]
    close    = params["close_after_show"]

    np.random.seed(seed)
    ms = get_microstructure(ms_name, m)
    E_list, nu_list = [E0,E1], [nu0,nu1]

    os.makedirs(out_dir, exist_ok=True)
    if save_png:
        os.makedirs(os.path.join(out_dir,"geometry_plots",ms_name), exist_ok=True)

    fname = os.path.join(out_dir, f"{ms_name}_Ch.csv")
    save_row(fname, ms.feature_names()+["C1111","C1122","C1112","C2222","C2212","C1212"], mode="w")

    start = time.time()
    for i in range(num_data):
        feats = ms.generate_random_input() if rand else tuple(fixed)
        ids   = ms.generate_ids(*feats)

        log(f"\n-- Sample {i+1}/{num_data} " + "-"*36)
        log(f"  features = {[round(float(v),6) for v in feats]}")
        log(f"  mat0 fraction (matrix)    = {float(np.mean(ids==0)):.6f}")
        log(f"  mat1 fraction (inclusion) = {float(np.mean(ids==1)):.6f}")

        png_path = None
        if save_png:
            fs = "_".join(f"{float(v):.4g}" for v in feats)
            png_path = os.path.join(out_dir,"geometry_plots",ms_name,
                                    f"iter_{i+1:04d}_{fs}.png")
        if save_png or show_plt:
            plot_geometry(ids,m,title=f"{ms_name} - sample {i+1}",cmap="viridis",
                         save_path=png_path,show_nonblocking=show_plt,
                         pause_time=pause,close_after_show=close)

        Ch = homogenize_numpy(m, ids, E_list, nu_list, ps)

        save_row(fname, list(feats)+[Ch[0,0],Ch[0,1],Ch[0,2],Ch[1,1],Ch[1,2],Ch[2,2]])
        log(f"  elapsed = {time.time()-start:.1f}s")
        log("  Ch =")
        log(f"    {Ch[0,0]:+.6e}  {Ch[0,1]:+.6e}  {Ch[0,2]:+.6e}")
        log(f"    {Ch[1,0]:+.6e}  {Ch[1,1]:+.6e}  {Ch[1,2]:+.6e}")
        log(f"    {Ch[2,0]:+.6e}  {Ch[2,1]:+.6e}  {Ch[2,2]:+.6e}")
        if png_path: log(f"  PNG: {png_path}")

    log(f"\nDone. CSV: {fname}")
    return fname


# ============================================================
#  GUI
# ============================================================

_FONT_TITLE = ("DejaVu Sans Mono", 11, "bold")
_FONT_LABEL = ("DejaVu Sans Mono",  9)
_FONT_SMALL = ("DejaVu Sans Mono",  8)
_FONT_LOG   = ("DejaVu Sans Mono",  9)


class HomogenizationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Periodic Homogenization")
        self.root.geometry("920x900")
        self.root.resizable(True, True)

        self.param_entries = []
        self.ms_var       = tk.StringVar(value="EllipseCentered")
        self.out_dir_var  = tk.StringVar(value=os.path.abspath("./data_homog"))
        self.rand_var     = tk.BooleanVar(value=True)
        self.ps_var       = tk.BooleanVar(value=True)
        self.save_png_var = tk.BooleanVar(value=True)
        self.show_plt_var = tk.BooleanVar(value=True)
        self.close_var    = tk.BooleanVar(value=True)

        self._build()
        self._refresh_geom()

    def _build(self):
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill="both", expand=True)

        # Title
        tk.Label(main, text="Periodic Homogenization",
                 font=_FONT_TITLE, anchor="center").pack(fill="x", pady=(0,2))

        form = ttk.Frame(main)
        form.pack(fill="x")
        r = [0]

        def row_entry(lbl, default):
            tk.Label(form, text=lbl, font=_FONT_LABEL, anchor="w").grid(
                row=r[0], column=0, sticky="w", padx=8, pady=3)
            v = tk.StringVar(value=str(default))
            ttk.Entry(form, textvariable=v, width=24).grid(
                row=r[0], column=1, sticky="w", padx=8, pady=3)
            r[0]+=1; return v

        def sep():
            ttk.Separator(form, orient="horizontal").grid(
                row=r[0], column=0, columnspan=5, sticky="ew", pady=5); r[0]+=1

        def note(txt, color="#222"):
            tk.Label(form, text=txt, font=_FONT_SMALL, fg=color, anchor="w").grid(
                row=r[0], column=0, columnspan=5, sticky="w", padx=8, pady=1); r[0]+=1

        # Microstructure
        tk.Label(form, text="Microstructure", font=_FONT_LABEL, anchor="w").grid(
            row=r[0], column=0, sticky="w", padx=8, pady=3)
        cb = ttk.Combobox(form, textvariable=self.ms_var,
                          values=["EllipseCentered","Biotruss","Lattice"],
                          state="readonly", width=22)
        cb.grid(row=r[0], column=1, sticky="w", padx=8, pady=3)
        cb.bind("<<ComboboxSelected>>", lambda _: self._refresh_geom())
        r[0]+=1

        self.m_var    = row_entry("Mesh size m  (m x m elements)", 32)
        self.nd_var   = row_entry("Number of data points", 5)
        self.seed_var = row_entry("Random seed", 42)

        sep()
        # Material 0
        note("E_list[0]  ->  material index 0  ->  MATRIX / outside the inclusion", "#003080")
        self.E0_var  = row_entry("E0  [Pa]  (matrix / outside)", 0.5e9)
        self.nu0_var = row_entry("nu0        (matrix / outside)", 0.3)
        # Material 1
        note("E_list[1]  ->  material index 1  ->  INCLUSION / inside  (hole_id=1)", "#800000")
        self.E1_var  = row_entry("E1  [Pa]  (inclusion / inside)", 1e-6)
        self.nu1_var = row_entry("nu1        (inclusion / inside)", 0.3)

        ttk.Checkbutton(form,
            text="Plane stress  (default: True)",
            variable=self.ps_var).grid(
            row=r[0], column=0, columnspan=3, sticky="w", padx=8, pady=3); r[0]+=1

        sep()
        ttk.Checkbutton(form,
            text="Use random geometry   (uncheck to use fixed values below)",
            variable=self.rand_var).grid(
            row=r[0], column=0, columnspan=3, sticky="w", padx=8, pady=3); r[0]+=1

        self.gf = ttk.LabelFrame(form,
            text="Fixed geometry parameters  (used when random geometry is OFF)")
        self.gf.grid(row=r[0], column=0, columnspan=5, sticky="ew", padx=8, pady=6)
        r[0]+=1

        sep()
        tk.Label(form, text="Output folder", font=_FONT_LABEL, anchor="w").grid(
            row=r[0], column=0, sticky="w", padx=8, pady=3)
        ttk.Entry(form, textvariable=self.out_dir_var, width=48).grid(
            row=r[0], column=1, columnspan=2, sticky="w", padx=8, pady=3)
        ttk.Button(form, text="Browse", command=self._browse).grid(
            row=r[0], column=3, sticky="w", padx=8, pady=3); r[0]+=1

        self.pause_var = row_entry("Plot pause  [s]", 0.25)

        chkf = ttk.Frame(form)
        chkf.grid(row=r[0], column=0, columnspan=5, sticky="w", pady=3); r[0]+=1
        ttk.Checkbutton(chkf, text="Save PNG",
                        variable=self.save_png_var).pack(side="left", padx=10)
        ttk.Checkbutton(chkf, text="Show plots",
                        variable=self.show_plt_var).pack(side="left", padx=10)
        ttk.Checkbutton(chkf, text="Close plot after show",
                        variable=self.close_var).pack(side="left", padx=10)

        # Buttons
        bf = ttk.Frame(main)
        bf.pack(fill="x", pady=8)
        ttk.Button(bf, text="Run",
                   command=self._run).pack(side="left", padx=6)
        ttk.Button(bf, text="Test homogeneous material",
                   command=self._test).pack(side="left", padx=6)
        ttk.Button(bf, text="Quit",
                   command=self.root.destroy).pack(side="left", padx=6)

        # Log
        lf = ttk.Frame(main)
        lf.pack(fill="both", expand=True, pady=(4,0))
        self.log = tk.Text(lf, height=14, wrap="word", font=_FONT_LOG)
        sb = ttk.Scrollbar(lf, command=self.log.yview)
        self.log.configure(yscrollcommand=sb.set)
        self.log.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        self._log("Homogenization GUI ready.")


    def _browse(self):
        d = filedialog.askdirectory(initialdir=self.out_dir_var.get())
        if d: self.out_dir_var.set(d)

    def _refresh_geom(self):
        for c in self.gf.winfo_children(): c.destroy()
        self.param_entries = []
        ms = get_microstructure(self.ms_var.get(), 32)
        names, defs = ms.feature_names(), ms.default_fixed_features()
        n = len(names); two_col = n > 4
        for i,(nm,df) in enumerate(zip(names,defs)):
            gr = i%(( n+1)//2) if two_col else i
            co = (i//((n+1)//2))*2 if two_col else 0
            tk.Label(self.gf, text=nm, font=_FONT_LABEL, anchor="w").grid(
                row=gr, column=co, sticky="w", padx=8, pady=2)
            v = tk.StringVar(value=str(df))
            ttk.Entry(self.gf, textvariable=v, width=14).grid(
                row=gr, column=co+1, sticky="w", padx=8, pady=2)
            self.param_entries.append((nm,v))

    def _log(self, txt):
        self.log.insert("end", str(txt)+"\n")
        self.log.see("end")
        self.root.update_idletasks()

    def _params(self):
        m = int(self.m_var.get())
        n = int(self.nd_var.get())
        if m < 2: raise ValueError("m must be >= 2")
        if n < 1: raise ValueError("num_data must be >= 1")
        return {
            "microstructure_name": self.ms_var.get(),
            "m": m, "num_data": n, "seed": int(self.seed_var.get()),
            "E0": float(self.E0_var.get()),  "E1": float(self.E1_var.get()),
            "nu0": float(self.nu0_var.get()), "nu1": float(self.nu1_var.get()),
            "plane_stress": bool(self.ps_var.get()),
            "output_dir":   self.out_dir_var.get(),
            "random_geometry": bool(self.rand_var.get()),
            "fixed_features": [float(v.get()) for _,v in self.param_entries],
            "save_png":    bool(self.save_png_var.get()),
            "show_plots":  bool(self.show_plt_var.get()),
            "pause_time":  float(self.pause_var.get()),
            "close_after_show": bool(self.close_var.get()),
        }

    def _test(self):
        try: p = self._params()
        except Exception as e: messagebox.showerror("Input error", str(e)); return
        mt = min(p["m"], 16)
        self._log(f"\nHomogeneous material test  (m={mt}, E=1.0, nu=0.3) ...")
        try:
            Ch, Ce, err = test_homogeneous(mt, 1.0, 0.3, p["plane_stress"])
            self._log("Computed Ch:")
            for row in Ch: self._log("  "+"  ".join(f"{v:+.6e}" for v in row))
            self._log("Exact C:")
            for row in Ce: self._log("  "+"  ".join(f"{v:+.6e}" for v in row))
            self._log("Abs error (matrix):")
            for row in err: self._log("  "+"  ".join(f"{v:.2e}" for v in row))
            mx = float(np.max(err))
            self._log(f"Max abs error : {mx:.3e}")
            self._log(f"Max rel error : {rel_err:.3e}  ->  {'PASSED' if rel_err<1e-8 else 'FAILED'}")
            self._log("-"*60)
        except Exception:
            s=traceback.format_exc(); self._log(s); messagebox.showerror("Failed",s)

    def _run(self):
        try: p = self._params()
        except Exception as e: messagebox.showerror("Input error",str(e)); return
        self._log("\nStarting run ...")
        self._log(f"  ms={p['microstructure_name']}  m={p['m']}  n={p['num_data']}")
        self._log(f"  E_list=[{p['E0']:.4g}, {p['E1']:.4g}]  nu_list=[{p['nu0']},{p['nu1']}]")
        self._log(f"  plane_stress={p['plane_stress']}  rand={p['random_geometry']}")
        self._log("-"*60)
        try:
            f = generate_dataset(p, log=self._log)
            messagebox.showinfo("Done", f"Finished.\nCSV: {f}")
        except Exception:
            s=traceback.format_exc(); self._log(s); messagebox.showerror("Failed",s)


# ============================================================
#  Entry point
# ============================================================

def main():
    root = tk.Tk()
    HomogenizationGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()