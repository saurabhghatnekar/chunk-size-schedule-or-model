"""Apply the criteria of docs/go_no_go_architectural.md to results/pilot_architectural.csv.

    uv run python -m src.pilot.analyze_architectural

Writes results/tables/pilot_architectural_summary.csv, results/figures/pilot_architectural.{png,pdf}
and docs/pilot_architectural.md. The decision rule was written down before the pilot ran.
"""
from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
DECISION_CELLS = (32, 128, 512)
ANCHOR = 1024
# 1% critical values of F with 2 numerator degrees of freedom, by denominator degrees of freedom
F_CRIT = {12: 6.93, 11: 7.21, 10: 7.56, 9: 8.02}


def anova(groups: dict[int, list[float]]) -> tuple[float, float, int]:
    """One-way analysis of variance. Returns (F, pooled within-cell standard deviation, denominator df)."""
    values = [v for g in groups.values() for v in g]
    grand = sum(values) / len(values)
    between = sum(len(g) * (sum(g) / len(g) - grand) ** 2 for g in groups.values())
    within = sum((v - sum(g) / len(g)) ** 2 for g in groups.values() for v in g)
    df_between, df_within = len(groups) - 1, len(values) - len(groups)
    s_within = math.sqrt(within / df_within)
    f = (between / df_between) / (within / df_within) if within > 0 else math.inf
    return f, s_within, df_within


def decide(groups: dict[int, list[float]], anchor: list[float]) -> dict:
    f, s_within, df = anova(groups)
    means = {c: sum(g) / len(g) for c, g in groups.items()}
    spread = max(means.values()) - min(means.values())
    crit = F_CRIT.get(df)
    anchor_mean = sum(anchor) / len(anchor) if anchor else float("nan")
    gap_to_anchor = max(abs(m - anchor_mean) for m in means.values()) if anchor else float("nan")
    if crit is None:
        verdict = "INCOMPLETE"
    elif anchor and gap_to_anchor <= 2 * s_within:
        verdict = "UNINFORMATIVE"
    elif f > crit and spread > 2 * s_within:
        verdict = "EFFECT"
    else:
        verdict = "NO EFFECT DETECTED"
    return {"F": f, "F_crit": crit, "df_within": df, "s_within": s_within, "range_of_means": spread,
            "range_over_s_within": spread / s_within if s_within else math.inf, "anchor_mean": anchor_mean,
            "largest_gap_to_anchor": gap_to_anchor, "verdict": verdict, **{f"mean_chunk_{c}": m for c, m in means.items()}}


def figure(final: pd.DataFrame, path_stem: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, ink2, muted, grid, axis = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
    blue, grey = "#2a78d6", "#898781"
    plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "axes.edgecolor": axis, "axes.labelcolor": ink2,
                         "xtick.color": muted, "ytick.color": muted, "text.color": ink,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(6.2, 3.8), facecolor=surface)
    ax.set_facecolor(surface)
    cells = [c for c in (*DECISION_CELLS, ANCHOR) if c in set(final["chunk"])]
    by_seed = final.pivot(index="seed", columns="chunk", values="val_loss")
    for seed, row in by_seed.iterrows():  # one line per seed: same initial weights and data order across cells
        ax.plot(range(len(cells)), [row[c] for c in cells], color=axis, lw=1.0, zorder=1)
        ax.annotate(f"seed {seed}", (len(cells) - 1, row[cells[-1]]), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=7, color=ink2)
    for x, chunk in enumerate(cells):
        values = by_seed[chunk].tolist()
        ax.scatter([x] * len(values), values, s=38, color=grey if chunk == ANCHOR else blue, edgecolor=surface,
                   linewidth=1.2, zorder=3)
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels([f"chunk {c}" + ("\n(no test-time training)" if c == ANCHOR else "") for c in cells])
    ax.set_xlim(-0.4, len(cells) - 0.35)
    ax.grid(axis="y", color=grid, lw=0.6)
    ax.set_ylabel("final held-out loss (nats per byte)")
    ax.set_title("One dot per run; lines join the runs that share a seed", loc="left", fontsize=9, color=ink)
    for ext in ("png", "pdf"):
        fig.savefig(f"{path_stem}.{ext}", dpi=200, bbox_inches="tight", facecolor=surface)
    plt.close(fig)


def paired_section(final: pd.DataFrame, result: dict) -> str:
    """Seed-paired differences. This is NOT the pre-registered analysis and is labelled as such."""
    by_seed = final.pivot(index="seed", columns="chunk", values="val_loss").dropna()
    cells = [c for c in (*DECISION_CELLS, ANCHOR) if c in by_seed]
    table = ["| Seed | " + " | ".join(f"chunk {c}" for c in cells) + " |", "|---|" + "---|" * len(cells)]
    for seed, row in by_seed.iterrows():
        table.append(f"| {seed} | " + " | ".join(f"{row[c]:.4f}" for c in cells) + " |")
    ordered = sum(all(row[a] <= row[b] + 1e-12 for a, b in zip(cells, cells[1:])) for _, row in by_seed.iterrows())
    n = len(by_seed)
    diffs = ["| Comparison | Difference per seed | Mean | t (" + str(n - 1) + " d.f.) | Seeds where the larger chunk is worse |",
             "|---|---|---|---|---|"]
    for a, b in ((32, 128), (32, 512), (128, 512), (32, ANCHOR), (512, ANCHOR)):
        if a not in by_seed or b not in by_seed:
            continue
        d = (by_seed[b] - by_seed[a]).tolist()
        mean = sum(d) / n
        sd = math.sqrt(sum((x - mean) ** 2 for x in d) / (n - 1))
        t = mean / (sd / math.sqrt(n)) if sd > 0 else math.inf
        diffs.append(f"| chunk {b} minus chunk {a} | {' '.join(f'{x:+.4f}' for x in d)} | {mean:+.4f} | {t:+.2f} | "
                     f"{sum(x > 0 for x in d)} of {n} |")
    return f"""## Outside the pre-registered rule: the same data paired by seed

The rule above treats runs as independent draws. That choice was made because, in the precision pilot,
rerunning one seed moved the loss as much as changing the seed. In this model the seed moves the loss far more
than the chunk length does (within-cell standard deviation {result['s_within']:.4f}), and within each seed the losses
line up by chunk length:

{chr(10).join(table)}

The loss is non-decreasing in chunk length in {ordered} of {n} seeds across all {len(cells)} cells.

{chr(10).join(diffs)}

For reference, the two-sided critical values of t with {n - 1} degrees of freedom are {({4: '2.78 (5%) and 4.60 (1%)', 3: '3.18 (5%) and 5.84 (1%)', 2: '4.30 (5%) and 9.92 (1%)'}).get(n - 1, 'not tabulated here')}.

**How to read this.** It is a post-hoc analysis chosen after seeing the data, so it does not count toward go
and is not a result. It does say the pre-registered noise model was the wrong one for this layer, and it gives
a specific, testable prediction: smaller chunks (more fast-weight updates) give lower loss at equal steps.
What it needs is a fresh run with new seeds, a paired test fixed in advance, and same-seed reruns to measure
the run-to-run floor, which this pilot does not have.
"""


def document(frame: pd.DataFrame, final: pd.DataFrame, result: dict) -> str:
    row0 = final.iloc[0]
    sha_now = hashlib.sha256((ROOT / "docs" / "go_no_go_architectural.md").read_bytes()).hexdigest()
    sha_run = str(frame["thresholds_sha256"].dropna().iloc[0])
    cells = ["| Chunk length | Fast-weight updates per sequence | Runs | Final loss of each run | Mean | Std | Tokens/s |",
             "|---|---|---|---|---|---|---|"]
    for chunk in (*DECISION_CELLS, ANCHOR):
        g = final[final["chunk"] == chunk].sort_values("seed")
        if g.empty:
            continue
        label = f"{chunk} (anchor, no test-time training)" if chunk == ANCHOR else str(chunk)
        cells.append(f"| {label} | {int(g['fast_weight_updates_per_sequence'].iloc[0])} | {len(g)} | "
                     f"{' '.join(f'{v:.4f}' for v in g['val_loss'])} | {g['val_loss'].mean():.4f} | "
                     f"{g['val_loss'].std(ddof=1):.4f} | {g['tokens_per_s'].mean() / 1e3:.0f}k |")
    cross = ["| Trained at chunk | " + " | ".join(f"evaluated at {c}" for c in (*DECISION_CELLS, ANCHOR)) + " |",
             "|---|" + "---|" * 4]
    for chunk in (*DECISION_CELLS, ANCHOR):
        g = final[final["chunk"] == chunk]
        if g.empty:
            continue
        vals = [g[f"val_loss_at_chunk_{c}"].mean() for c in (*DECISION_CELLS, ANCHOR)]
        cross.append(f"| {chunk} | " + " | ".join(f"**{v:.4f}**" if c == chunk else f"{v:.4f}"
                                                   for c, v in zip((*DECISION_CELLS, ANCHOR), vals)) + " |")
    failed = frame[frame["status"] != "ok"]
    meaning = {
        "EFFECT": "Chunk length moves trained quality beyond run-to-run noise. **This counts toward go** under the rule fixed beforehand.",
        "NO EFFECT DETECTED": "Chunk length does not move trained quality beyond run-to-run noise at this scale. This does **not** count toward go.",
        "UNINFORMATIVE": "No decision cell differs from the anchor with test-time training switched off, so the layer is not "
                         "using test-time training at this scale and the pilot says nothing about chunk length either way.",
        "INCOMPLETE": "Too few runs completed to apply the rule.",
    }[result["verdict"]]
    best = min(DECISION_CELLS, key=lambda c: result.get(f"mean_chunk_{c}", math.inf))
    return f"""# Small pilot on the architectural class

Generated by `uv run python -m src.pilot.analyze_architectural` from `results/pilot_architectural.csv`
(run {row0['run_utc']}, {row0['gpu']}, torch {row0['torch']}). Criteria: `docs/go_no_go_architectural.md`.

Criteria file hash stored in the result rows: `{sha_run[:16]}`. Hash of the file now: `{sha_now[:16]}`
({"unchanged since the pilot was launched" if sha_run == sha_now else "the file has been edited since the pilot was launched"}).

## Setup

Byte-level language model on enwik8, {int(row0['params']):,} parameters ({int(row0['n_layers'])} layers, width {int(row0['d_model'])},
{int(row0['n_heads'])} heads). Each block sums a causal sliding-window attention (window {int(row0['window'])}, fixed) and the
official LaCT fast-weight op under its own bfloat16 autocast. {int(row0['steps'])} steps of {int(row0['batch'])} windows of
{int(row0['seq_len'])} bytes. Only the test-time-training chunk length differs between cells. The loss is the held-out loss on
{int(row0['eval_windows'])} fixed validation windows, each model evaluated at the chunk length it was trained with.

## Result

{chr(10).join(cells)}

| Quantity | Value |
|---|---|
| F across chunk {', '.join(map(str, DECISION_CELLS))} | {result['F']:.2f} (1% critical value {result['F_crit']}, {result['df_within']} denominator degrees of freedom) |
| Within-cell standard deviation | {result['s_within']:.4f} |
| Range of the three cell means | {result['range_of_means']:.4f} ({result['range_over_s_within']:.1f} within-cell standard deviations; the rule needs more than 2) |
| Largest gap between a decision cell and the anchor | {result['largest_gap_to_anchor']:.4f} ({result['largest_gap_to_anchor'] / result['s_within']:.1f} within-cell standard deviations; below 2 would make the pilot uninformative) |

**Verdict: {result['verdict']}.** {meaning}

Lowest mean loss among the decision cells: chunk {best}. The direction is reported, not part of the criterion.

![Pilot](../results/figures/pilot_architectural.png)

{paired_section(final, result)}
## The same trained models evaluated at other chunk lengths

Mean final loss over the runs of each row; bold is the chunk length the row was trained with. This is the
deployment question (inference-time chunk length differing from training-time) and is not part of the decision.

{chr(10).join(cross)}

{"## Runs that did not complete" + chr(10) + chr(10) + failed[["chunk", "seed", "status"]].to_string(index=False) if len(failed) else ""}

## What this does and does not say

- It is a model of a few million parameters trained for minutes at a context of 1024 bytes, far below the chunk
  lengths LaCT is designed for. It shows whether chunk length can matter for trained quality at all, not how
  much it matters at scale or which chunk length is best there.
- The attention window is fixed at {int(row0['window'])} for every cell so that only one thing changes. LaCT usually ties the window
  to the chunk; that coupled comparison is a different experiment.
- Chunk length also changes throughput several-fold (last column of the first table), so equal steps is not
  equal wall-clock. The roadmap's wall-clock question is separate and is not answered here.
- Five runs per cell, treated as independent draws because same-seed reruns vary as much as different seeds.
"""


def main() -> int:
    frame = pd.read_csv(RESULTS / "pilot_architectural.csv")
    final = frame[(frame["status"] == "ok") & (frame["final"] == True)]  # noqa: E712
    groups = {c: final[final["chunk"] == c]["val_loss"].tolist() for c in DECISION_CELLS}
    groups = {c: g for c, g in groups.items() if g}
    anchor = final[final["chunk"] == ANCHOR]["val_loss"].tolist()
    if len(groups) < len(DECISION_CELLS) or any(len(g) < 2 for g in groups.values()):
        print("too few completed runs to analyse")
        return 1
    result = decide(groups, anchor)
    (RESULTS / "tables").mkdir(exist_ok=True)
    pd.DataFrame([result]).to_csv(RESULTS / "tables" / "pilot_architectural_summary.csv", index=False)
    figure(final, RESULTS / "figures" / "pilot_architectural")
    (ROOT / "docs" / "pilot_architectural.md").write_text(document(frame, final, result))
    for key, value in result.items():
        print(f"{key:26s} {value:.5f}" if isinstance(value, float) else f"{key:26s} {value}")
    print("wrote docs/pilot_architectural.md, results/tables/pilot_architectural_summary.csv, results/figures/pilot_architectural.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
