import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# pattern = 
def plt_loss_curves(seq_len):
    csv_files = glob.glob(f"bf16/loss-*-{seq_len}-part*.csv")

    # Plot the loss curves
    plt.figure(figsize=(8, 5))
    restart_steps = {}
    for fname in sorted(csv_files):
        df = pd.read_csv(fname)
        if {"step", "loss"} <= set(df.columns):
            label = os.path.splitext(os.path.basename(fname))[0]

            if "torch" in label:
                color = 'gold'
            else:
                color = 'lime'
            
            if "pccheck" in label:
                lw = 1.5
                dashes = (5, 5)
                linestyle = 'dashed'
                color = 'darkorange' if color == 'gold' else 'darkgreen'
            else:
                lw = 1.5
                dashes = (5, 0)
                linestyle = 'solid'
                    
            
            if "part1" in label:
                label = label.replace(f"-{seq_len}-part1", "").replace("loss-", "")
                restart_steps[label] = df["step"].max()
            else:
                label = None

            plt.plot(df["step"][::4], df["loss"][::4], label=label, linestyle=linestyle, dashes=dashes, lw=lw, color=color)

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

for seq_len in [256, 512, 1024, 2048]:
    plt_loss_curves(seq_len)