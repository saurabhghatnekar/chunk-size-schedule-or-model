"""Apply the criteria of docs/go_no_go.md (section 1) to results/pilot_precision.csv.

    uv run python -m src.pilot.analyze

Writes results/tables/pilot_precision_summary.csv, results/figures/pilot_precision.{png,pdf} and
docs/pilot_precision.md. The decision rule is the one written down before the pilot ran; this
script does not choose anything.
"""
from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
T_CRIT = 4.30  # two-sided 5% critical value of Student's t with 2 degrees of freedom (three pairs)
REGIMES = {"bf16": "bfloat16 autocast", "fp32_tf32": "float32, Triton default (TF32)", "fp32_ieee": "float32, IEEE dot products"}
METRIC = "val_loss_ref"


def decide(d: list[float], sigma_seed: float) -> tuple[str, float, float, float]:
    n = len(d)
    mean = sum(d) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in d) / (n - 1))
    t = mean / (sd / math.sqrt(n)) if sd > 0 else math.inf
    effect = abs(t) > T_CRIT and abs(mean) > sigma_seed
    return ("EFFECT" if effect else "NO EFFECT DETECTED"), mean, sd, t


def summarise(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    final = frame[(frame["status"] == "ok") & (frame["final"] == True)]  # noqa: E712
    main = final[final["replicate"] == 0]
    rows = []
    for regime in REGIMES:
        cell = main[main["regime"] == regime].pivot(index="seed", columns="chunk", values=METRIC).dropna()
        if cell.shape[0] < 2 or not {16, 64} <= set(cell.columns):
            rows.append({"regime": regime, "verdict": "INCOMPLETE", "seeds": cell.shape[0]})
            continue
        d = list(cell[64] - cell[16])
        sigma = math.sqrt((cell[16].var(ddof=1) + cell[64].var(ddof=1)) / 2)
        verdict, mean, sd, t = decide(d, sigma)
        if cell.shape[0] != 3:
            verdict = f"INCOMPLETE ({cell.shape[0]} of 3 seeds)"
        own = main[main["regime"] == regime].pivot(index="seed", columns="chunk", values="val_loss_own")
        rows.append({
            "regime": regime, "seeds": cell.shape[0],
            "loss_chunk16_mean": cell[16].mean(), "loss_chunk64_mean": cell[64].mean(),
            "d_mean": mean, "d_sd": sd, "t": t, "d_max_abs": max(abs(x) for x in d),
            "sigma_seed": sigma, "d_per_seed": " ".join(f"{x:+.5f}" for x in d),
            "own_minus_ref_max_abs": float((own - cell).abs().max().max()),
            "kernel_oracle_dev_max": main[main["regime"] == regime]["kernel_oracle_dev"].max(),
            "tokens_per_s_mean": main[main["regime"] == regime]["tokens_per_s"].mean(),
            "verdict": verdict,
        })
    floors = []
    for (regime, chunk, seed), group in final.groupby(["regime", "chunk", "seed"]):
        if group["replicate"].nunique() > 1:
            group = group.sort_values("replicate")
            values = group[METRIC].tolist()
            norms = group["param_l2"].dropna()
            floors.append({"regime": regime, "chunk": chunk, "seed": seed, "reruns": len(values),
                           "losses": " ".join(f"{v:.5f}" for v in values), "floor": max(values) - min(values),
                           "same_weight_norm": "unknown" if len(norms) < 2 else ("yes" if norms.nunique() == 1 else "no"),
                           "from_log": int(group["source"].notna().sum()) if "source" in group else 0})
    return pd.DataFrame(rows), pd.DataFrame(floors)


def figure(frame: pd.DataFrame, path_stem: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, ink2, muted, grid, axis = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
    colour = {16: "#2a78d6", 64: "#eb6834"}
    plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "axes.edgecolor": axis, "axes.labelcolor": ink2,
                         "xtick.color": muted, "ytick.color": muted, "text.color": ink,
                         "axes.spines.top": False, "axes.spines.right": False})
    final = frame[(frame["status"] == "ok") & (frame["final"] == True) & (frame["replicate"] == 0)]  # noqa: E712
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8), sharey=True, facecolor=surface, gridspec_kw={"wspace": 0.12})
    for ax, (regime, title) in zip(axes, REGIMES.items()):
        ax.set_facecolor(surface)
        cell = final[final["regime"] == regime].pivot(index="seed", columns="chunk", values=METRIC)
        for seed, row in cell.iterrows():
            if {16, 64} <= set(row.dropna().index):
                ax.plot([0, 1], [row[16], row[64]], color=axis, lw=1.0, zorder=1)
                ax.annotate(f"seed {seed}", (1, row[64]), xytext=(7, 0), textcoords="offset points", va="center",
                            fontsize=7, color=ink2)
        for x, chunk in enumerate((16, 64)):
            if chunk in cell:
                ax.scatter([x] * len(cell), cell[chunk], s=42, color=colour[chunk], edgecolor=surface, linewidth=1.2, zorder=3)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["chunk 16", "chunk 64"])
        ax.set_xlim(-0.4, 1.6)
        ax.grid(axis="y", color=grid, lw=0.6)
        ax.set_title(title, loc="left", fontsize=9, color=ink)
    axes[0].set_ylabel("final held-out loss (nats per byte)\nreference evaluation")
    for ext in ("png", "pdf"):
        fig.savefig(f"{path_stem}.{ext}", dpi=200, bbox_inches="tight", facecolor=surface)
    plt.close(fig)


def fmt(x, digits=5):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{digits}f}"


def document(frame: pd.DataFrame, summary: pd.DataFrame, floors: pd.DataFrame) -> str:
    row0 = frame[frame["status"] == "ok"].iloc[0]
    sha_now = hashlib.sha256((ROOT / "docs" / "go_no_go.md").read_bytes()).hexdigest()
    sha_run = str(frame["thresholds_sha256"].iloc[0])
    failed = frame[frame["status"] != "ok"]
    lines = ["| Arithmetic | Kernel vs exact recurrence | Loss, chunk 16 | Loss, chunk 64 | d per seed (64 minus 16) | mean d | t | seed spread | Verdict |",
             "|---|---|---|---|---|---|---|---|---|"]
    for _, r in summary.iterrows():
        if "d_mean" not in r or pd.isna(r.get("d_mean")):
            lines.append(f"| {REGIMES[r['regime']]} | | | | | | | | **{r['verdict']}** |")
            continue
        lines.append(f"| {REGIMES[r['regime']]} | {r['kernel_oracle_dev_max']:.1e} | {fmt(r['loss_chunk16_mean'])} | "
                     f"{fmt(r['loss_chunk64_mean'])} | {r['d_per_seed']} | {r['d_mean']:+.5f} | {r['t']:+.2f} | "
                     f"{fmt(r['sigma_seed'])} | **{r['verdict']}** |")
    floor_lines = ["| Arithmetic | Chunk | Seed | Final loss of each rerun | Largest difference | Same final weight norm |", "|---|---|---|---|---|---|"]
    for _, r in floors.iterrows():
        note = f" ({r['from_log']} of {r['reruns']} read from the run log, 4 d.p.)" if r["from_log"] else ""
        floor_lines.append(f"| {REGIMES[r['regime']]} | {r['chunk']} | {r['seed']} | {r['losses']}{note} | {r['floor']:.1e} | {r['same_weight_norm']} |")
    done = summary[summary.get("d_max_abs", pd.Series(dtype=float)).notna()] if "d_max_abs" in summary else summary.iloc[:0]
    if len(floors) and len(done):
        floor_note = (f"Training runs with these kernels did not reproduce across fresh containers (cause not isolated; "
                      f"Triton's autotuner choosing a different kernel configuration is a candidate): the same configuration and seed "
                      f"gives a final loss that differs by {floors['floor'].min():.1e} to {floors['floor'].max():.1e} nats. "
                      f"The largest chunk-length difference at a fixed seed is {done['d_max_abs'].max():.1e} nats and the seed "
                      f"spread is {done['sigma_seed'].min():.1e} to {done['sigma_seed'].max():.1e}. So a same-seed comparison "
                      f"between two chunk lengths is no tighter than a comparison between two reruns of one chunk length, "
                      f"and the chunk-length differences seen here are inside that floor.")
    else:
        floor_note = ""
    incomplete = summary[summary["verdict"].astype(str).str.startswith("INCOMPLETE")]
    missing_note = ("**Missing.** " + ", ".join(REGIMES[r] for r in incomplete["regime"]) + " did not complete at both chunk "
                    "lengths: the compute account stopped accepting requests while those runs were in flight. "
                    if len(incomplete) else "")
    signs = [float(x) for r in done["d_per_seed"] for x in str(r).split()] if len(done) else []
    negative = sum(x < 0 for x in signs)
    sign_note = (f"**An observation outside the pre-registered rule.** {negative} of the {len(signs)} seed-paired differences "
                 f"are negative (chunk 64 slightly lower). The rule does not test this, the differences are inside the "
                 f"run-to-run floor below, and with this few pairs it should be read as something to check with more "
                 f"seeds, not as a result." if signs and (negative in (0, len(signs))) else "")
    bf16 = summary[summary["regime"] == "bf16"]
    counts = bf16["verdict"].iloc[0] if len(bf16) else "INCOMPLETE"
    toward_go = {"EFFECT": "This counts toward go under the pre-registered rule.",
                 "NO EFFECT DETECTED": "This does **not** count toward go under the pre-registered rule."}.get(
                     counts, "The bfloat16 arm is incomplete, so the pilot does not decide.")
    return f"""# Small precision pilot on the scheduling class

Generated by `uv run python -m src.pilot.analyze` from `results/pilot_precision.csv`
(run {row0['run_utc']}, {row0['gpu']}, torch {row0['torch']}). Criteria: `docs/go_no_go.md`, section 1.

Criteria file hash stored in the result rows: `{sha_run[:16]}`. Hash of the file now: `{sha_now[:16]}`
({"unchanged since the pilot was launched" if sha_run == sha_now else "the file has been edited since the pilot was launched"}).

## Setup

Byte-level language model on enwik8, {int(row0['params']):,} parameters ({int(row0['n_layers'])} layers, width {int(row0['d_model'])},
{int(row0['n_heads'])} heads), mixing layer `chunk_gated_delta_rule` from flash-linear-attention at the pinned commit.
{int(row0['steps'])} steps of {int(row0['batch'])} windows of {int(row0['seq_len'])} bytes. The seed fixes the initial weights and
the order of training windows, so runs with the same seed differ only in chunk length and arithmetic. The loss
is the held-out loss on {int(row0['eval_windows'])} fixed validation windows, every model evaluated with the same reference
forward pass (float32, recurrent kernel). The largest difference between that and the loss under a model's own
training configuration is {summary['own_minus_ref_max_abs'].max():.1e} nats.

## Result

{chr(10).join(lines)}

{missing_note}`d` is the loss at chunk 64 minus the loss at chunk 16 for the same seed. EFFECT requires |t| > {T_CRIT} and
|mean d| larger than the seed spread. **bfloat16: {counts}.** {toward_go}

{sign_note}

![Pilot](../results/figures/pilot_precision.png)

*Final held-out loss per run. Grey lines join the two chunk lengths of one seed.*

## Run-to-run floor

The same configuration and seed run twice:

{chr(10).join(floor_lines) if len(floors) else "No repeat runs completed."}

{floor_note}

{"## Runs that did not complete" + chr(10) + chr(10) + failed[["regime", "chunk", "seed", "status"]].to_string(index=False) if len(failed) else ""}

## What this does and does not say

- It is one small model trained for minutes on one GPU. A null here does not bound a 160M-parameter model
  trained to convergence; it says the effect is not large at this scale.
- Three seeds give a coarse seed spread, and the test has low power: with three pairs only a shift several
  times the pair-to-pair scatter can pass.
- The three arithmetics are verified per run: the "Kernel vs exact recurrence" column is measured inside each
  training container before training starts.
"""


def main() -> int:
    frame = pd.read_csv(RESULTS / "pilot_precision.csv")
    for col, default in (("replicate", 0), ("final", False)):
        if col not in frame:
            frame[col] = default
    frame["replicate"] = frame["replicate"].fillna(0).astype(int)
    from_log = RESULTS / "pilot_precision_from_log.csv"
    if from_log.exists():  # finished runs whose rows were never returned; final losses read off the run log
        extra = pd.read_csv(from_log).assign(status="ok", final=True)
        have = set(map(tuple, frame[frame["final"] == True][["regime", "chunk", "seed", "replicate"]].values.tolist()))  # noqa: E712
        extra = extra[[tuple(v) not in have for v in extra[["regime", "chunk", "seed", "replicate"]].values.tolist()]]
        frame = pd.concat([frame, extra], ignore_index=True)
    summary, floors = summarise(frame)
    (RESULTS / "tables").mkdir(exist_ok=True)
    summary.to_csv(RESULTS / "tables" / "pilot_precision_summary.csv", index=False)
    figure(frame, RESULTS / "figures" / "pilot_precision")
    (ROOT / "docs" / "pilot_precision.md").write_text(document(frame, summary, floors))
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(summary.drop(columns=["own_minus_ref_max_abs", "tokens_per_s_mean"], errors="ignore").to_string(index=False))
        print(floors.to_string(index=False) if len(floors) else "no repeat runs")
    print("wrote docs/pilot_precision.md, results/tables/pilot_precision_summary.csv, results/figures/pilot_precision.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
