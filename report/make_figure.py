"""Figure 1: Round 1 replayed with all our high quotes set to one rate (numbers from the exact replay)."""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
out = sys.argv[1]
q = [40.00, 50.00, 60.00, 70.00, 80.00, 89.99, 91.89, 95.00, 97.89, 99.99]
p = [29.92, 39.75, 50.20, 61.12, 72.01, 74.55, 76.40, 60.40, 62.51, 64.07]
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 8, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6})
fig, ax = plt.subplots(figsize=(3.35, 2.2))
ax.plot(q, p, color="0.6", lw=1.3, marker="o", ms=2.8, zorder=2)
best = max(range(len(p)), key=lambda i: p[i]); sub = q.index(97.89)
ax.plot(q[best], p[best], "o", ms=5, color="#1f5fbf", zorder=3)
ax.plot(q[sub], p[sub], "o", ms=5, color="black", zorder=3)
ax.axhline(73.31, color="0.4", ls="--", lw=0.7, zorder=1)
ax.text(40.5, 74.8, "Round 1 winner: €73.3M", fontsize=7, color="0.3")
ax.plot([91.99, 91.99], [0, 52], color="0.4", ls=":", lw=0.9, zorder=1)
ax.text(91.2, 3, "rival's 91.99% quote", fontsize=7, color="0.3", ha="right")
ax.annotate(f"best: {q[best]:.2f}%, €{p[best]:.1f}M", (q[best], p[best]), xytext=(66.5, 81.5), fontsize=7,
            color="#1f5fbf", arrowprops=dict(arrowstyle="-", lw=0.5, color="#1f5fbf"))
ax.annotate(f"submitted: {q[sub]:.2f}%, €{p[sub]:.1f}M", (q[sub], p[sub]), xytext=(57, 36), fontsize=7,
            arrowprops=dict(arrowstyle="-", lw=0.5, color="0.2"))
ax.set_xlabel("Single rate on all our high quotes (%)")
ax.set_ylabel("Round 1 profit (€M)")
ax.set_xlim(38, 102); ax.set_ylim(0, 88)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(pad=0.3)
fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"), dpi=200)
print("saved", out)
