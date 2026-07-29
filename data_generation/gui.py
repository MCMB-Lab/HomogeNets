# -*- coding: utf-8 -*-
"""Tkinter graphical interface for periodic homogenization."""

from __future__ import annotations

import os
import traceback
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np

from data_generation import generate_dataset
from fem import test_homogeneous
from microstructures import get_microstructure

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
            Ch, Ce, err, rel_err = test_homogeneous(mt, 1.0, 0.3, p["plane_stress"])
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

