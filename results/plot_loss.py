import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# pattern = 
def plt_loss_curves(seq_len, plot_extended=False):
    if plot_extended:
        fig_width = 20
        skip_points = 1
    else:
        fig_width = 8
        skip_points = 4
    
    csv_files_to_style = {
        # CSV File Name:                                    ( Color,        lw,     Label)
        f"bf16/valid-base-torch-{seq_len}-1.csv":           ('orange',      3.00,   "Pure Pytorch w/o Restart"),
        f"bf16/valid-base-torch-{seq_len}-2.csv":           ('gold',        0.75,   None),
        f"bf16/valid-base-flash-{seq_len}-1.csv":           ('green',       3.00,   "FAv3 w/o Restart"),
        f"bf16/valid-base-flash-{seq_len}-2.csv":           ('lime',        0.75,   None),
        f"bf16/valid-pccheck-flash-{seq_len}-1.csv":        ('purple',      3.00,   "FAv3 + PCcheck w/o Restart"),
        f"bf16/valid-pccheck-flash-{seq_len}-2.csv":        ('fuchsia',     0.75,   None),
        f"bf16/valid-pccheck-flash-{seq_len}-part1.csv":    ('deepskyblue', 1.50,   "FAv3 + PCcheck w/ Restart"),
        f"bf16/valid-pccheck-flash-{seq_len}-part2.csv":    ('deepskyblue', 1.50,   None),
    }

    csv_files = list(csv_files_to_style.keys())
    # Plot the loss curves
    plt.figure(figsize=(fig_width, 5))
    restart_steps = {}
    for fname in csv_files:
        df = pd.read_csv(fname)
        if {"step", "loss"} <= set(df.columns):
            color, lw, label = csv_files_to_style[fname]
            dashes = (2, 2) if "part" in fname else (5, 0)
            linestyle = 'dashed' if "part" in fname else 'solid'
            
            if "part1" in fname:
                restart_steps[label] = df["step"].max()

            plt.plot(df["step"][::skip_points], 
                     df["loss"][::skip_points], 
                     label=label, linestyle=linestyle, dashes=dashes, lw=lw, color=color)

    # Check if all elements in restart_steps are the same
    if all(x == list(restart_steps.values())[0] for x in restart_steps.values()):
        plt.axvline(x=list(restart_steps.values())[0], color='red', label='Restart', linestyle='dashed')
    else:
        for key, step in restart_steps.items():
            plt.axvline(x=step, label=f'Restart {key}', linestyle='dashed')
    
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss Across Runs")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"loss_curves_{seq_len}.png", dpi=600)

plt_loss_curves(2048)