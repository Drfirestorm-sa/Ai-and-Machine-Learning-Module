"""

In that file's `guess_distribution`, unsolved games are counted under "6", so
solved-in-6 = dist["6"] - failures and failures get their own "fail" bar
(matching the Dataset A/B charts).
"""
import json, sys, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src = sys.argv[1] if len(sys.argv) > 1 else "results_datasetC.json"
dst = sys.argv[2] if len(sys.argv) > 2 else "guess_distribution_C.png"
d = json.load(open(src))

cats = ["1", "2", "3", "4", "5", "6", "fail"]
fig, ax = plt.subplots(figsize=(7, 4))
w = 0.16
for k, r in enumerate(d["results"]):
    dist, f, n = r["guess_distribution"], r["failures"], r["games"]
    counts = [dist.get(c, 0) for c in cats[:5]] + [dist.get("6", 0) - f, f]
    assert sum(counts) == n, (r["agent"], sum(counts), n)
    ax.bar([i + (k - 2) * w for i in range(len(cats))],
           [100 * c / n for c in counts], w, label=r["agent"])
ax.set_xticks(range(len(cats))); ax.set_xticklabels(cats)
ax.set_xlabel("Guesses needed"); ax.set_ylabel("% of games")
ax.set_title(f"Dataset C: guess distribution ({d['evaluation_games']} games)")
ax.legend(fontsize=7); fig.tight_layout()
fig.savefig(dst, dpi=150)
print("saved", dst)
