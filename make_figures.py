import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
for ds in "AB":
    d = json.load(open(f"results/results_dataset{ds}.json"))
    fig, ax = plt.subplots(figsize=(7, 4))
    cats = ["1", "2", "3", "4", "5", "6", "fail"]
    w = 0.16
    for k, r in enumerate(d["results"]):
        dist = r["distribution"]
        vals = [100 * dist.get(c, 0) / r["games"] for c in cats]
        ax.bar([i + (k - 2) * w for i in range(len(cats))], vals, w, label=r["agent"])
    ax.set_xticks(range(len(cats))); ax.set_xticklabels(cats)
    ax.set_xlabel("Guesses needed"); ax.set_ylabel("% of games")
    ax.set_title(f"Dataset {ds}: guess distribution ({d['meta']['evaluation_games']} games)")
    ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(f"results/guess_distribution_{ds}.png", dpi=150)
