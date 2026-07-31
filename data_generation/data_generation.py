# -*- coding: utf-8 -*-
"""Dataset generation and CSV output."""

from __future__ import annotations

import csv
import os
import time
import numpy as np

from fem import homogenize_numpy
from microstructures import get_microstructure
from plotting import plot_geometry

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

