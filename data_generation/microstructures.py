# -*- coding: utf-8 -*-
"""Microstructure geometry generators."""

from __future__ import annotations

import math
import numpy as np

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

