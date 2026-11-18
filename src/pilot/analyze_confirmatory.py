"""Apply docs/go_no_go_confirmatory.md to results/pilot_confirmatory.csv.

    uv run python -m src.pilot.analyze_confirmatory

Writes results/tables/pilot_confirmatory_summary.csv, results/figures/pilot_confirmatory.{png,pdf}
and docs/pilot_confirmatory.md. Every rule here was written down before the pilot ran.
"""
from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
T_CRIT_ONE_SIDED_1PCT = {5: 3.365, 4: 3.747, 3: 4.541}  # by degrees of freedom
RUNG_WIDTH = {"A": 256, "B": 512}
CELLS = (32, 512, 1024)
SCIENCE = {
    ("EFFECT", "CONSISTENT"): ("GO", "chunk length moves trained quality on the architectural class, and the direction "
                                     "holds at the first step up in size"),
    ("EFFECT", "NOT CONSISTENT"): ("WEAK GO", "the effect exists at small scale and may not survive scale; proceed only as "
                                              "far as the roadmap-scale pilot, not the grid"),
    ("EFFECT", "INCOMPLETE"): ("WEAK GO", "the effect exists at small scale; the larger rung did not complete, so its "
                                          "direction at scale is unknown"),
}


def paired(d: list[float]) -> tuple[float, float, float]:
    n = len(d)
    mean = sum(d) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in d) / (n - 1)) if n > 1 else float("nan")
    t = mean / (sd / math.sqrt(n)) if sd and sd > 0 else math.inf
    return mean, sd, t


def decide_a(d: list[float], floor: float) -> str:
    crit = T_CRIT_ONE_SIDED_1PCT.get(len(d) - 1)
    if crit is None or math.isnan(floor):
        return "INCOMPLETE"
    mean, _, t = paired(d)
    return "EFFECT" if (t > crit and mean > floor) else "NO EFFECT"


def decide_b(d: list[float], floor: float) -> str:
    if len(d) < 3 or math.isnan(floor):
        return "INCOMPLETE"
    return "CONSISTENT" if (all(x > 0 for x in d) and sum(d) / len(d) > floor) else "NOT CONSISTENT"


def summarise(frame: pd.DataFrame) -> dict:
    final = frame[(frame["status"] == "ok") & (frame["final"] == True)].copy()  # noqa: E712
    final["replicate"] = final["replicate"].fillna(0).astype(int)
    out = {}
    a = final[final["d_model"] == RUNG_WIDTH["A"]]
    b = final[final["d_model"] == RUNG_WIDTH["B"]]
    # run-to-run floor from the same-seed reruns of rung A
    spans = []
    for (chunk, seed), g in a.groupby(["chunk", "seed"]):
        if g["replicate"].nunique() > 1:
            values = g.sort_values("replicate")["val_loss"].tolist()
            spans.append({"chunk": int(chunk), "seed": int(seed), "losses": " ".join(f"{v:.4f}" for v in values),
                          "span": max(values) - min(values)})
    out["reruns"] = spans
    out["floor"] = max((s["span"] for s in spans), default=float("nan")) if len(spans) >= 2 else float("nan")
    for name, rung in (("A", a), ("B", b)):
        first = rung[rung["replicate"] == 0].pivot(index="seed", columns="chunk", values="val_loss")
        out[f"table_{name}"] = first
        have = [c for c in CELLS if c in first]
        if not {32, 512} <= set(have):
            out[f"d_{name}"], out[f"anchor_d_{name}"] = [], []
            continue
        both = first.dropna(subset=[32, 512])
        out[f"d_{name}"] = (both[512] - both[32]).tolist()
        out[f"seeds_{name}"] = both.index.tolist()
        anchor = first.dropna(subset=[32, 1024]) if 1024 in first else first.iloc[:0]
        out[f"anchor_d_{name}"] = (anchor[1024] - anchor[32]).tolist() if len(anchor) else []
    out["verdict_A"] = decide_a(out["d_A"], out["floor"]) if out["d_A"] else "INCOMPLETE"
    out["verdict_B"] = decide_b(out["d_B"], out["floor"]) if out["d_B"] else "INCOMPLETE"
    if out["verdict_A"] == "NO EFFECT":
        out["science"] = ("NO-GO for the full paper", "neither the precision branch nor the architectural branch shows an "
                          "effect above noise; fall back to the shorter paper the roadmap describes")
    elif out["verdict_A"] == "INCOMPLETE":
        out["science"] = ("UNDECIDED", "rung A did not complete")
    else:
        out["science"] = SCIENCE[(out["verdict_A"], out["verdict_B"])]
    return out


def figure(result: dict, path_stem: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, ink2, muted, grid, axis = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
    blue, grey = "#2a78d6", "#898781"
    plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "axes.edgecolor": axis, "axes.labelcolor": ink2,
                         "xtick.color": muted, "ytick.color": muted, "text.color": ink,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), facecolor=surface, gridspec_kw={"wspace": 0.28})
    titles = {"A": "Rung A: 5.2M parameters, six new seeds", "B": "Rung B: about 39M parameters, three seeds"}
    for ax, name in zip(axes, ("A", "B")):
        ax.set_facecolor(surface)
        table = result[f"table_{name}"]
        cells = [c for c in CELLS if c in table]
        for seed, row in table.iterrows():
            xs = [i for i, c in enumerate(cells) if not pd.isna(row[c])]
            ax.plot(xs, [row[cells[i]] for i in xs], color=axis, lw=1.0, zorder=1)
            if xs:
                ax.annotate(f"seed {seed}", (xs[-1], row[cells[xs[-1]]]), xytext=(8, 0), textcoords="offset points",
                            va="center", fontsize=7, color=ink2)
        for x, chunk in enumerate(cells):
            values = table[chunk].dropna().tolist()
            ax.scatter([x] * len(values), values, s=38, color=grey if chunk == 1024 else blue, edgecolor=surface,
                       linewidth=1.2, zorder=3)
        ax.set_xticks(range(len(cells)))
        ax.set_xticklabels([f"chunk {c}" + ("\n(no test-time training)" if c == 1024 else "") for c in cells])
        ax.set_xlim(-0.4, len(cells) - 0.3)
        ax.grid(axis="y", color=grid, lw=0.6)
        ax.set_title(titles[name], loc="left", fontsize=9, color=ink)
    axes[0].set_ylabel("final held-out loss (nats per byte)")
    for ext in ("png", "pdf"):
        fig.savefig(f"{path_stem}.{ext}", dpi=200, bbox_inches="tight", facecolor=surface)
    plt.close(fig)


def _table(table: pd.DataFrame) -> str:
    cells = [c for c in CELLS if c in table]
    lines = ["| Seed | " + " | ".join(f"chunk {c}" for c in cells) + " | chunk 512 minus chunk 32 |", "|---|" + "---|" * (len(cells) + 1)]
    for seed, row in table.iterrows():
        d = row[512] - row[32] if {32, 512} <= set(cells) and not (pd.isna(row[32]) or pd.isna(row[512])) else float("nan")
        lines.append(f"| {seed} | " + " | ".join("n/a" if pd.isna(row[c]) else f"{row[c]:.4f}" for c in cells)
                     + f" | {'n/a' if math.isnan(d) else f'{d:+.4f}'} |")
    return "\n".join(lines)


# one-sided critical values of t, to say where a statistic falls without computing a p-value
T_LADDER = {5: [(0.05, 2.015), (0.025, 2.571), (0.01, 3.365), (0.005, 4.032)],
            4: [(0.05, 2.132), (0.025, 2.776), (0.01, 3.747), (0.005, 4.604)],
            3: [(0.05, 2.353), (0.025, 3.182), (0.01, 4.541), (0.005, 5.841)]}


def outside_the_rule(r: dict) -> str:
    """Facts the pre-registered rule does not use. Reported for the reader, never to change the verdict."""
    d_a, d_b, floor = r["d_A"], r["d_B"], r["floor"]
    if len(d_a) < 2:
        return ""
    mean, sd, t = paired(d_a)
    ladder = T_LADDER.get(len(d_a) - 1, [])
    passed = [level for level, crit in ladder if t > crit]
    failed = [level for level, crit in ladder if t <= crit]
    where = (f"It clears the one-sided {max(passed) if not failed else min(passed):.1%} level" if passed else "It clears none of the usual levels")
    if passed and failed:
        where = f"It clears the one-sided {min(passed):.1%} level and not the {max(failed):.1%} level"
    biggest = max(d_a)
    rest = [x for x in d_a if x != biggest]
    rest_mean, rest_sd, rest_t = paired(rest) if len(rest) >= 2 else (float("nan"),) * 3
    lines = [
        f"- **Sign.** The larger chunk is worse in {sum(x > 0 for x in d_a)} of {len(d_a)} new seeds in rung A"
        + (f" and {sum(x > 0 for x in d_b)} of {len(d_b)} in rung B." if d_b else "."),
        f"- **Size against the run-to-run floor.** The floor is {floor:.4f}. The smallest of the {len(d_a)} rung A differences is "
        f"{min(d_a):.4f}, which is {min(d_a) / floor:.1f} floors; the mean is {mean / floor:.0f} floors. Reruns of this model "
        f"are nearly reproducible, unlike the Gated DeltaNet kernels in the precision pilot.",
        f"- **Where t falls.** t = {t:.2f} with {len(d_a) - 1} degrees of freedom. {where}.",
        f"- **Why t is low.** One seed has a difference of {biggest:+.4f}, about {biggest / (sum(rest) / len(rest)):.0f} times the "
        f"others. It is in the predicted direction, but it inflates the standard deviation the t statistic divides by. "
        f"Without it the other {len(rest)} give mean {rest_mean:+.4f}, standard deviation {rest_sd:.4f}, t = {rest_t:.2f}. "
        f"Dropping a seed after the fact is not allowed; this only explains the number.",
    ]
    return ("## Outside the rule: what the verdict does not use\n\n"
            "These are facts about the same data. They do not change the verdict above, which is the one the rule gives.\n\n"
            + "\n".join(lines) + "\n")


def document(frame: pd.DataFrame, r: dict) -> str:
    ok = frame[frame["status"] == "ok"]
    row0 = ok.iloc[0]
    sha_now = hashlib.sha256((ROOT / "docs" / "go_no_go_confirmatory.md").read_bytes()).hexdigest()
    sha_run = str(frame["thresholds_sha256"].dropna().iloc[0])
    failed = frame[frame["status"] != "ok"]
    stats = []
    for name, label in (("A", "Rung A, chunk 512 minus chunk 32"), ("B", "Rung B, chunk 512 minus chunk 32")):
        d = r[f"d_{name}"]
        if len(d) >= 2:
            mean, sd, t = paired(d)
            crit = T_CRIT_ONE_SIDED_1PCT.get(len(d) - 1)
            stats.append(f"| {label} | {len(d)} | {mean:+.4f} | {sd:.4f} | {t:+.2f}"
                         + (f" (needs more than {crit})" if name == "A" and crit else "") + f" | {sum(x > 0 for x in d)} of {len(d)} |")
    for name, label in (("A", "Rung A, no test-time training minus chunk 32"), ("B", "Rung B, no test-time training minus chunk 32")):
        d = r[f"anchor_d_{name}"]
        if len(d) >= 2:
            mean, sd, t = paired(d)
            stats.append(f"| {label} (reported, not part of the decision) | {len(d)} | {mean:+.4f} | {sd:.4f} | {t:+.2f} | {sum(x > 0 for x in d)} of {len(d)} |")
    reruns = "\n".join(f"| {s['chunk']} | {s['seed']} | {s['losses']} | {s['span']:.4f} |" for s in r["reruns"])
    tput = ok[ok["final"] == True].groupby(["d_model", "chunk"])["tokens_per_s"].mean()  # noqa: E712
    speed = "; ".join(f"width {int(w)}, chunk {int(c)}: {v / 1e3:.0f}k" for (w, c), v in tput.items())
    return f"""# Confirmatory pilot on the architectural class

Generated by `uv run python -m src.pilot.analyze_confirmatory` from `results/pilot_confirmatory.csv`
(run {row0['run_utc']}, {row0['gpu']}, torch {row0['torch']}). Criteria: `docs/go_no_go_confirmatory.md`.

Criteria file hash stored in the result rows: `{sha_run[:16]}`. Hash of the file now: `{sha_now[:16]}`
({"unchanged since the pilot was launched" if sha_run == sha_now else "the file has been edited since the pilot was launched"}).

**Hypothesis fixed in advance.** At an equal number of training steps, a LaCT-style layer trained with chunk
length 32 reaches a lower held-out loss than the same model trained with chunk length 512.

## Rung A: verdict {r['verdict_A']}

Six seeds never used before, 5.2M parameters, final held-out loss in nats per byte:

{_table(r['table_A'])}

Run-to-run floor, from the same seed run three times:

| Chunk | Seed | Final loss of each run | Span |
|---|---|---|---|
{reruns}

Floor used by the rule (the largest span): **{r['floor']:.4f}**.

## Rung B: verdict {r['verdict_B']}

About 39M parameters, three seeds:

{_table(r['table_B']) if len(r['table_B']) else "No runs completed."}

## The statistics

| Comparison (same seed) | Seeds | Mean | Std | t | Larger chunk worse in |
|---|---|---|---|---|---|
{chr(10).join(stats)}

Rung A is an EFFECT if its t exceeds the one-sided 1% critical value and its mean exceeds the floor. Rung B is
CONSISTENT if the difference is positive in all three seeds and its mean exceeds the rung A floor.

## Decision on the science: {r['science'][0]}

By the table fixed in advance: {r['science'][1]}.

{outside_the_rule(r)}
![Confirmatory pilot](../results/figures/pilot_confirmatory.png)

*One dot per run; lines join the runs that share a seed.*

{"## Runs that did not complete" + chr(10) + chr(10) + failed[[c for c in ("rung", "d_model", "chunk", "seed", "replicate", "status") if c in failed]].to_string(index=False) if len(failed) else ""}

## What this does and does not say

- The comparison is at equal steps. Measured training throughput in tokens per second: {speed}. Chunk 32 costs
  several times the compute of chunk 512, so a lower loss at equal steps is not a better model at equal
  wall-clock. That is the roadmap's separate wall-clock question, and these numbers say it is a real trade.
- Both rungs are far below the roadmap's pilot scale and far below the chunk lengths LaCT is designed for.
- One layer type, one dataset, one fixed attention window.
"""


def main() -> int:
    frame = pd.read_csv(RESULTS / "pilot_confirmatory.csv")
    result = summarise(frame)
    (RESULTS / "tables").mkdir(exist_ok=True)
    rows = []
    for name in ("A", "B"):
        d = result[f"d_{name}"]
        mean, sd, t = paired(d) if len(d) >= 2 else (float("nan"),) * 3
        rows.append({"rung": name, "seeds": len(d), "d_per_seed": " ".join(f"{x:+.5f}" for x in d), "d_mean": mean, "d_sd": sd,
                     "t": t, "t_crit": T_CRIT_ONE_SIDED_1PCT.get(len(d) - 1) if name == "A" else float("nan"),
                     "d_min": min(d) if d else float("nan"), "positive": sum(x > 0 for x in d),
                     "floor": result["floor"], "verdict": result[f"verdict_{name}"], "science": result["science"][0]})
    pd.DataFrame(rows).to_csv(RESULTS / "tables" / "pilot_confirmatory_summary.csv", index=False)
    figure(result, RESULTS / "figures" / "pilot_confirmatory")
    (ROOT / "docs" / "pilot_confirmatory.md").write_text(document(frame, result))
    print(pd.DataFrame(rows).to_string(index=False))
    print("reruns:", result["reruns"])
    print("science:", result["science"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
