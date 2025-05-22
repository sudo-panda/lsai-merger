import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# pattern = 
def plt_loss_curves(seq_len):
    csv_files = [f"bf16/loss-base-torch-{seq_len}-part1.csv", f"bf16/loss-base-torch-{seq_len}-part2.csv", f"bf16/loss-pccheck-flash-{seq_len}-part1.csv", f"bf16/loss-pccheck-flash-{seq_len}-part2.csv"]

    # Plot the loss curves
    plt.figure(figsize=(8, 5))
    for fname in sorted(csv_files):
        df = pd.read_csv(fname)
        if {"step", "loss"} <= set(df.columns):
            label = os.path.splitext(os.path.basename(fname))[0]
            if "pccheck" in label:
                lw = 1.5
                dashes = (5, 5)
                linestyle = 'dashed'
                color = 'darkorange'
                if "part1" in label:
                    label = "PCcheck + FlashAttentionV3"
                    pccheck_restart = df["step"].max()
                else:
                    label = None
            else:
                lw = 1.5
                dashes = (5, 0)
                linestyle = 'solid'
                color = 'darkgreen'
                if "part1" in label:
                    label = "torch.save + SDPA"
                    base_restart = df["step"].max()
                else:
                    label = None

            plt.plot(df["step"][::4], df["loss"][::4], label=label, linestyle=linestyle, dashes=dashes, lw=lw, color=color)

    if base_restart == pccheck_restart:
        plt.axvline(x = base_restart, color = 'red', label = 'Restart', linestyle='dashed')
    else:
        plt.axvline(x = base_restart, color = 'orange', label = 'Restart (Base)', linestyle='dashed')
        plt.axvline(x = pccheck_restart, color = 'green', label = 'Restart (PCcheck + FAv3)', linestyle='dashed')
    
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss Across Runs")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"loss_curves_{seq_len}.png", dpi=600)

for seq_len in [256, 512, 1024, 2048]:
    plt_loss_curves(seq_len)