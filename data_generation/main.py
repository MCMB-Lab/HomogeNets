# -*- coding: utf-8 -*-
"""Application entry point."""

import tkinter as tk

from gui import HomogenizationGUI


def main():
    root = tk.Tk()
    HomogenizationGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
