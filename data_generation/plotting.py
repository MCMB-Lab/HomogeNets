# -*- coding: utf-8 -*-
"""Geometry plotting utilities."""

from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

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

