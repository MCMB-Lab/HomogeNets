# -*- coding: utf-8 -*-
"""Finite-element routines for periodic homogenization."""

from __future__ import annotations

import math
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from numpy.linalg import solve as np_solve

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


def test_homogeneous(m=8, E=1.0, nu=0.3, plane_stress=True):
    """
    All elements = material 0 (solid). Must recover the single-material C exactly.
    Returns (Ch_computed, Ch_expected, abs_error).
    """
    ids = np.zeros(m*m, dtype=int)
    Ch  = homogenize_numpy(m, ids, [E, 1e-6], [nu, nu], plane_stress)
    Ce  = build_C(E, nu, plane_stress)
    return Ch, Ce, np.abs(Ch - Ce), np.max(np.abs(Ch - Ce)) / max(np.max(np.abs(Ce)), 1.0)

