import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

IN = Path("results/NSGA2.csv")
OUT_PNG = Path("results/NSGA2_plot.png")

if not IN.exists():
    raise SystemExit(f"Input file not found: {IN}")

print(f"Reading {IN}...")
df = pd.read_csv(IN)

# Add ID column if missing
if 'ID' not in df.columns:
    df.insert(0, 'ID', range(1, len(df) + 1))

# Colors for consistency
colors = {'Latency': 'tab:blue', 'Spread': 'tab:orange', 'UnderUtil': 'tab:green'}

plt.figure(figsize=(10, 6))
for col in ('Latency', 'Spread', 'UnderUtil'):
    if col in df.columns:
        x = df['ID'].to_numpy()
        y = df[col].to_numpy()

        # plot raw data with low visibility
        plt.plot(
            x,
            y,
            marker='o',
            markersize=2,
            linestyle='-',
            linewidth=0.7,
            markerfacecolor='none',
            markeredgewidth=0.6,
            alpha=0.35,
            label=col,
            color=colors.get(col),
        )

        # linear fit (emphasized)
        if len(x) >= 2:
            coeffs = np.polyfit(x, y, 1)
            p = np.poly1d(coeffs)
            yhat = p(x)
            ss_res = np.sum((y - yhat) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0
            plt.plot(
                x,
                yhat,
                linestyle='--',
                linewidth=2.0,
                alpha=0.95,
                color=colors.get(col),
                label=f"{col} trend (R²={r2:.3f})",
            )

plt.xlabel('ID')
plt.ylabel('Value')
plt.title('NSGA2 results')
plt.legend(fontsize='small')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=200)
print(f"Saved plot: {OUT_PNG}")
