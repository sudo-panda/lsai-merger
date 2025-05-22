import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# pattern = 
csv_files = ["bf16/loss-base-torch-2048-part1.csv", "bf16/loss-base-torch-2048-part2.csv", "bf16/loss-pccheck-flash-2048-part1.csv", "bf16/loss-pccheck-flash-2048-part2.csv"]

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
            label = "PCcheck + FlashAttentionV3" if "part1" in label else None
        else:
            lw = 1.5
            dashes = (5, 0)
            linestyle = 'solid'
            color = 'darkgreen'
            label = "torch.save + SDPA" if "part1" in label else None

        plt.plot(df["step"][::4], df["loss"][::4], label=label, linestyle=linestyle, dashes=dashes, lw=lw, color=color)

plt.axvline(x = 300, color = 'red', label = 'Restart', linestyle='dotted')
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training Loss Across Runs")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("loss_curves.png", dpi=600)