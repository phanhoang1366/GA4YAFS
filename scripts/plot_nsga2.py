import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

FILES = {
    "NSGA2": Path("results/NSGA2.csv"),
    "Heuristic": Path("results/Memetic.csv"),
}
OUTPUT = Path("results/weighted_sum_comparison.png")
TARGET_GEN = 50
WEIGHTS = {"Latency": 1 / 3, "Spread": 1 / 3, "UnderUtil": 1 / 3}
OBJECTIVES = tuple(WEIGHTS.keys())
COLORS = {"Latency": "tab:blue", "Spread": "tab:orange", "UnderUtil": "tab:green"}


def load_frame(name: str, path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing file for {name}: {path}")
    frame = pd.read_csv(path)
    if "ID" not in frame.columns:
        frame.insert(0, "ID", range(len(frame)))
    missing = [col for col in OBJECTIVES if col not in frame.columns]
    if missing:
        raise SystemExit(f"{name} missing columns {missing} in {path}")
    frame["Weighted"] = sum(frame[col] * WEIGHTS[col] for col in OBJECTIVES)
    return frame


def pick_generation_rows(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    ordered = frame.sort_values("ID")
    gen_row = ordered.loc[ordered["ID"] == TARGET_GEN]
    if gen_row.empty:
        max_id = int(ordered["ID"].max()) if not ordered.empty else -1
        raise SystemExit(f"Generation {TARGET_GEN} not found (available IDs up to {max_id})")
    final_row = ordered.iloc[-1]
    return gen_row.iloc[0], final_row


def plot_series(ax, name: str, frame: pd.DataFrame, ymax: float | None = None) -> None:
    x = frame["ID"].to_numpy()

    for col in OBJECTIVES:
        y = frame[col].to_numpy()

        # Plot raw data with low visibility
        ax.plot(
            x,
            y,
            marker="o",
            markersize=2,
            linestyle="-",
            linewidth=0.7,
            markerfacecolor="none",
            markeredgewidth=0.6,
            alpha=0.35,
            color=COLORS[col],
            label=col,
        )

        # Linear fit (emphasized)
        if len(x) >= 2:
            coeffs = np.polyfit(x, y, 1)
            p = np.poly1d(coeffs)
            yhat = p(x)
            ss_res = np.sum((y - yhat) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0
            ax.plot(x, yhat, linestyle="--", linewidth=2.0, alpha=0.95, color=COLORS[col], 
                   label=f"{col} trend (R²={r2:.3f})")

    ax.set_xlabel("Generation (ID)")
    ax.set_ylabel("Value")
    ax.set_title(f"{name} raw objectives over generations")
    if ymax is not None:
        ax.set_ylim(0, ymax)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize="small")


def plot_diff(ax, label: str, baseline: float, challenger: float) -> None:
    delta = challenger - baseline  # negative means challenger better (lower)
    improvement_pct = ((baseline - challenger) / baseline * 100.0) if baseline != 0 else 0.0
    x_pos = 0
    height = delta
    ax.bar(x_pos, height, width=0.6, color="grey", alpha=0.7)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ymax = max(abs(delta), 1e-6) * 1.4
    ax.set_ylim(-ymax, ymax)
    ax.set_xticks([x_pos])
    ax.set_xticklabels([label])
    ax.set_ylabel("Δ weighted (Heuristic - NSGA2)")
    ax.set_title(f"Difference {label}")
    ax.text(x_pos, height + (0.05 * ymax if height >= 0 else -0.05 * ymax),
            f"Δ={delta:.3f}\nImprovement={improvement_pct:.1f}%",
            ha="center", va="bottom" if height >= 0 else "top", fontsize=9)


def plot_panels(frames: dict[str, pd.DataFrame], nsga_gen: pd.Series, nsga_final: pd.Series, heu_gen: pd.Series, heu_final: pd.Series):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Find max value across all objectives for consistent y-scale
    max_y = 0.0
    for frame in frames.values():
        for col in OBJECTIVES:
            max_y = max(max_y, frame[col].max())
    max_y *= 1.1
    
    plot_series(axes[0, 0], "NSGA2", frames["NSGA2"], ymax=max_y)
    plot_series(axes[0, 1], "Heuristic", frames["Heuristic"], ymax=max_y)

    plot_diff(axes[1, 0], f"Gen {TARGET_GEN}", baseline=nsga_gen["Weighted"], challenger=heu_gen["Weighted"])
    plot_diff(axes[1, 1], "Final", baseline=nsga_final["Weighted"], challenger=heu_final["Weighted"])

    fig.suptitle(f"Weighted sum comparison (Gen {TARGET_GEN} vs Final)", y=1.02)
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    fig.savefig(OUTPUT, dpi=200)
    print(f"Saved {OUTPUT}")


def main() -> None:
    frames: dict[str, pd.DataFrame] = {name: load_frame(name, path) for name, path in FILES.items()}
    nsga_gen, nsga_final = pick_generation_rows(frames["NSGA2"])
    heu_gen, heu_final = pick_generation_rows(frames["Heuristic"])
    plot_panels(frames, nsga_gen, nsga_final, heu_gen, heu_final)


if __name__ == "__main__":
    main()
