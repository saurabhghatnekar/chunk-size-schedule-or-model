"""Turn results/equivalence.csv into the classification: a table, a document and figure one.

    uv run python -m src.equivalence.classify

Writes results/precision_scaling.csv, results/tables/classification.csv,
docs/classification.md and results/figures/fig1_classification.{png,pdf}.
Every number in the document is read from the CSV; nothing is typed in by hand.
"""
from __future__ import annotations

import math
import sys

import pandas as pd

from .archs import SPECS
from .controls import check
from .kernels.verdict import FLOOR_ULPS32, ROUNDING_X, STRUCTURAL_X, summarise_kernels
from .library.vendor import PINS
from .model import ROOT, ModelConfig
from .precision_scaling import PERSIST_MAX, SHRINK_MIN, TOL64, summarise

RESULTS = ROOT / "results"
DOCS = ROOT / "docs"

# Why each architecture is expected to fall where it does. These are statements of
# mechanism, each backed by a numerical check in tests/theory/.
REASONS = {
    "softmax_tile": "Tile size only re-associates the softmax normaliser (streaming softmax). Pure kernel scheduling.",
    "gla": "The chunkwise form is the recurrence S_t = diag(a_t) S_{t-1} + k_t^T v_t with the sums re-associated; "
           "chunk 1 is the recurrent form and one chunk is the parallel form.",
    "deltanet": "Same recurrence, re-associated through the unit-triangular system for the pseudo-values "
                "(the WY / UT-transform representation of the product of Householder-like factors).",
    "gated_deltanet": "As DeltaNet with a scalar gate folded into the cumulative decay; still a re-association.",
    "mamba2": "The SSD block decomposition of one lower-triangular operator; scalar decay per head makes the "
              "off-diagonal blocks low rank.",
    "ttt_titans": "Inner-loop boundary: every gradient in a chunk is taken at the memory frozen at the chunk start. "
                  "Chunk 1 is online gradient descent (the delta rule when momentum and decay are off); a single "
                  "chunk evaluates every gradient at the initial memory (Hebbian linear attention in the same limit).",
    "lact": "Inner-loop boundary: apply-then-update, one fast-weight step and one weight renormalisation per chunk. "
            "The chunk sets how many updates happen and what each query can see; the last chunk is never used "
            "for an update, so a chunk of at least the sequence length means no test-time training at all.",
    "flash": "Attention-pattern boundary: quadratic relu^2 attention inside a chunk, linear attention to earlier "
             "chunks, and the chunk size also normalises both paths. One chunk is pure quadratic attention.",
    "hla": "Routing boundary: completed chunks are composed into affine maps and each query re-weights them with "
           "content-dependent sigmoid gates. With all gates at one it is Gated DeltaNet at every chunk length; "
           "with one chunk there is nothing to route and it is Gated DeltaNet again.",
    "fla_delta_rule": "Library reference ops for the DeltaNet recurrence; recurrent loop against chunkwise.",
    "fla_gated_delta_rule": "Library reference ops for the Gated DeltaNet recurrence; recurrent loop against chunkwise.",
    "fla_simple_gla": "Library reference ops for scalar-decay linear attention; recurrent loop against chunkwise.",
    "fla_ttt_linear": "Library TTT-Linear op: the mini-batch size is the inner-loop boundary.",
    "fla_titans_sequential": "Library Titans op, token loop: `chunk_size` sets when the frozen memory is refreshed.",
    "fla_titans_chunkwise": "Library Titans op, closed form per chunk: same boundary as the token loop.",
    "fla_titans_loop_vs_closed_form": "Two implementations of the same Titans function at one chunk length. "
                                      "This is a consistency check of the library, not a chunk-length sweep.",
    "lact_block_causal_swiglu": "The official LaCT op: apply-then-update with one fast-weight step per chunk.",
}


def fmt(x: float) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    if x == 0:
        return "0 (identical)"
    if math.isinf(x):
        return "inf"
    return f"{x:.1e}"


def md_table(frame: pd.DataFrame, columns: dict) -> str:
    head = "| " + " | ".join(columns.values()) + " |"
    rule = "|" + "|".join("---" for _ in columns) + "|"
    body = ["| " + " | ".join(str(row[c]) for c in columns) + " |" for _, row in frame.iterrows()]
    return "\n".join([head, rule, *body])


def per_chunk_table(frame: pd.DataFrame, tier: str, dtype: str, column: str) -> str:
    sub = frame[(frame["tier"] == tier) & (frame["dtype"] == dtype) & frame["chunk"].notna()].copy()
    sub["label"] = sub["chunk"].astype(int).astype(str)
    if tier == "library":
        sub["label"] = sub["variant"] + " " + sub["label"]
    lines = []
    for arch, group in sub.groupby("architecture", sort=False):
        cells = ", ".join(f"{lab}: {fmt(val)}" for lab, val in zip(group["label"], group[column]))
        lines.append(f"| `{arch}` | {cells} |")
    return "\n".join(["| Architecture | chunk length: deviation from the reference (first entry) |", "|---|---|", *lines])


def figure(frame: pd.DataFrame, summary: pd.DataFrame, path_stem) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, ink2, muted, grid, axis = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
    colour = {"scheduling": "#2a78d6", "architectural": "#eb6834"}
    control = "#898781"
    floor = 1e-17
    plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "axes.edgecolor": axis,
                         "axes.labelcolor": ink2, "xtick.color": muted, "ytick.color": muted,
                         "text.color": ink, "axes.spines.top": False, "axes.spines.right": False})
    ref = frame[(frame["tier"] == "reference") & frame["chunk"].notna()]
    cls = summary[summary["tier"] == "reference"].set_index("architecture")
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.0), facecolor=surface,
                             gridspec_kw={"width_ratios": [1, 1, 0.8], "wspace": 0.75})

    def style(name):
        role = cls.loc[name, "role"]
        if role != "candidate":
            return {"color": control, "ls": "--" if role == "sensitivity-control" else "-", "lw": 1.2, "zorder": 2}
        return {"color": colour.get(cls.loc[name, "classification"], ink), "ls": "-", "lw": 2.0, "zorder": 3}

    def labels(ax, entries, x):
        """Direct labels right of the last point, spread apart in log space around their
        lines and tied back with a hairline leader; text stays in ink, never the series colour."""
        entries = sorted(entries, key=lambda e: e[0])
        gap = 0.68
        want = [math.log10(y) for y, _ in entries]
        pos = list(want)
        for i in range(1, len(pos)):
            pos[i] = max(pos[i], pos[i - 1] + gap)
        start = 0
        for i in range(1, len(pos) + 1):  # re-centre each run of labels that had to be pushed apart
            if i == len(pos) or pos[i] - pos[i - 1] > gap + 1e-9:
                shift = sum(p - w for p, w in zip(pos[start:i], want[start:i])) / (i - start)
                pos[start:i] = [p - shift for p in pos[start:i]]
                start = i
        for i in range(len(pos) - 2, -1, -1):  # re-centring can close a gap to the run above or below
            pos[i] = min(pos[i], pos[i + 1] - gap)
        for (y, text), ly in zip(entries, pos):
            ax.annotate(text, xy=(x, y), xytext=(22, 10 ** ly), textcoords=("offset points", "data"),
                        va="center", fontsize=7, color=ink2, annotation_clip=False,
                        arrowprops={"arrowstyle": "-", "color": muted, "lw": 0.5, "shrinkA": 1, "shrinkB": 5})

    short = {"softmax_tile": "softmax tile control", "gla": "GLA", "deltanet": "DeltaNet",
             "gated_deltanet": "Gated DeltaNet", "mamba2": "Mamba-2", "ttt_titans": "TTT (Titans)",
             "lact": "TTT (LaCT)", "flash": "FLASH", "hla": "HLA routing",
             "perturbed_leak_1e-3": "defect 1e-3", "perturbed_leak_1e-6": "defect 1e-6",
             "perturbed_leak_1e-9": "defect 1e-9", "perturbed_stale_gate": "off-by-one gate"}

    for ax, column, title in ((axes[0], "fwd_max_rel", "Outputs (logits), float64"),
                              (axes[1], "grad_max_rel", "Gradients, float64")):
        ax.set_facecolor(surface)
        exact_end, ends = [], []
        for name in cls.index:
            rows = ref[(ref["architecture"] == name) & (ref["dtype"] == "float64") & (ref["chunk"] != ref["ref_chunk"])]
            rows = rows.sort_values("chunk")
            y = rows[column].clip(lower=floor)
            ax.plot(rows["chunk"], y, marker="o", ms=4.5, mec=surface, mew=0.8, **style(name))
            (exact_end if y.iloc[-1] < 1e-12 else ends).append((y.iloc[-1], short[name]))
        if exact_end:
            ends.append((max(v for v, _ in exact_end), "exact family and\nsoftmax tile control"))
        labels(ax, ends, ref["chunk"].max())
        ax.axhspan(floor, TOL64, color=grid, alpha=0.45, lw=0, zorder=0)
        ax.text(1.15, TOL64 * 0.25, "rounding level in float64", fontsize=7, color=muted, va="top")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_ylim(floor, 100)
        ticks = [1, 4, 16, 64, 256, 512]
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks], fontsize=7)
        ax.set_xticks([c for c in sorted(ref["chunk"].unique()) if c not in ticks], minor=True)
        ax.set_xticklabels([], minor=True)
        ax.grid(axis="y", color=grid, lw=0.6)
        ax.set_xlabel("chunk length (tokens)")
        ax.set_title(title, loc="left", fontsize=9, color=ink)
    axes[0].set_ylabel("max deviation from smallest chunk length\n(relative to largest reference value)")

    ax = axes[2]
    ax.set_facecolor(surface)
    order = ["bfloat16", "float16", "float32", "float64"]
    ends = []
    exact_end = []
    for name in cls.index:
        ys = [cls.loc[name, f"fwd_dev_{d}"] for d in order]
        ax.plot(range(4), [max(v, floor) for v in ys], marker="o", ms=4.5, mec=surface, mew=0.8, **style(name))
        (exact_end if ys[-1] < 1e-12 else ends).append((max(ys[-1], floor), short[name]))
    if exact_end:
        ends.append((max(v for v, _ in exact_end), "exact family and\nsoftmax tile control"))
    labels(ax, ends, 3)
    ax.set_yscale("log")
    ax.set_ylim(floor, 100)
    ax.set_xticks(range(4))
    ax.set_xticklabels(order, fontsize=7)
    ax.grid(axis="y", color=grid, lw=0.6)
    ax.set_xlabel("arithmetic precision")
    ax.set_title("Outputs, by precision", loc="left", fontsize=9, color=ink)

    handles = [plt.Line2D([], [], color=colour["scheduling"], lw=2, label="scheduling class"),
               plt.Line2D([], [], color=colour["architectural"], lw=2, label="architectural class"),
               plt.Line2D([], [], color=control, lw=1.2, label="negative control"),
               plt.Line2D([], [], color=control, lw=1.2, ls="--", label="injected-defect controls")]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.02))
    for ext in ("png", "pdf"):
        fig.savefig(f"{path_stem}.{ext}", dpi=200, bbox_inches="tight", facecolor=surface)
    plt.close(fig)


def trained_section(frame: pd.DataFrame) -> str:
    """The same test on two small trained models, from results/equivalence_trained.csv."""
    from .precision_scaling import classify as classify_verdict
    from .precision_scaling import verdict as scaling_verdict

    ok = frame[frame["status"] == "ok"]
    ref = ok[ok["tier"] == "trained-reference"]
    lines = ["| Trained model | Chunk lengths | Output dev. f64 | Gradient dev. f64 | Output dev. f32 | Shrink f32 to f64 | Diagnostic | Class |",
             "|---|---|---|---|---|---|---|---|"]
    for arch, g in ref.groupby("architecture", sort=False):
        worst = g.groupby("dtype")[["fwd_max_rel", "grad_max_rel"]].max()
        d32, d64 = worst.loc["float32"].max(), worst.loc["float64"].max()
        v = scaling_verdict(d32, d64)
        chunks = " ".join(str(int(c)) for c in dict.fromkeys(g["chunk"]))
        lines.append(f"| {arch} | {chunks} | {fmt(worst.loc['float64', 'fwd_max_rel'])} | {fmt(worst.loc['float64', 'grad_max_rel'])} | "
                     f"{fmt(worst.loc['float32', 'fwd_max_rel'])} | {fmt(d32 / d64 if d64 > 0 else float('inf'))} | {v} | "
                     f"**{classify_verdict(v, g['prior'].iloc[0])}** |")
    kern = ok[ok["tier"] == "trained-kernel"].copy()
    klines = ["| Trained model | Arithmetic | Chunk | Dev. from first chunk, outputs | Dev. from float64 reference, outputs | Dev. from float64 reference, gradients |",
              "|---|---|---|---|---|---|"]
    for _, r in kern.iterrows():
        grad = r.get("oracle_grad_max_rel")
        klines.append(f"| {r['architecture']} | {r['dtype']} | {int(r['chunk'])} | {fmt(r['fwd_max_rel'])} | "
                      f"{fmt(r['oracle_fwd_max_rel'])} | {fmt(grad) if pd.notna(grad) else 'n/a'} |")
    first = ok.iloc[0]
    return f"""## Result: the same test on trained weights

Everything above uses randomly initialised weights. Here the two pilot models are trained first
({int(first['steps'])} steps on enwik8, about 5M parameters each) and the test is repeated on the trained weights, with a
batch of two validation windows of 1024 bytes. Outputs are the logits; gradients are those of the next-byte
loss with respect to every weight.

Reference ops in float32 and float64, same decision rule as for the untrained models:

{chr(10).join(lines)}

The shipped code on a GPU with the same trained weights. For the Gated DeltaNet model the float64 reference
is the token-by-token recurrence, the same for every chunk length; for the LaCT model it is the reference
definition at the same chunk length, so "from first chunk" shows the chunk dependence and "from float64
reference" shows how closely the shipped op follows its own definition:

{chr(10).join(klines)}

"""


KERNEL_SCOPE_UNTESTED = """- **Implementations, not architectures in general.** The reference tier is code written for this project
  from the papers. The library tier covers only pure-PyTorch reference ops. The Triton kernels people
  train with (`chunk_gla`, `chunk_delta_rule`, `chunk_gated_delta_rule`, the Mamba-2 CUDA path, the fused
  LaCT kernel) have **not** been tested: this machine has no CUDA. Run `uv run modal run modal_app.py`."""

KERNEL_SCOPE_TESTED = """- **Implementations, not architectures in general.** The reference tier is code written for this project
  from the papers. The library and kernel tiers cover the pinned commits of flash-linear-attention and LaCT
  on the GPUs named above, with `torch.compile` disabled. The official Mamba-2 kernels (`mamba_ssm`) are not
  covered; `chunk_simple_gla` is the same scalar-decay recurrence but a different kernel."""


def kernel_section(kframe: pd.DataFrame) -> str:
    """The GPU kernel tier, from results/equivalence_kernels.csv."""
    summary = summarise_kernels(kframe)
    summary.to_csv(RESULTS / "tables" / "classification_kernels.csv", index=False)
    view = summary.copy()
    view["arch"] = "`" + view["architecture"] + "`"
    for col in ("cross_fwd", "cross_grad", "oracle_fwd", "oracle_grad", "definition_cross", "floor", "cross_over_floor", "oracle_over_floor"):
        view[col + "_s"] = view[col].map(lambda x: fmt(x) if pd.notna(x) else "n/a") if col in view else "n/a"
    view["class"] = "**" + view["classification"] + "**"
    view["ran_s"] = view["ran"].replace("", "none")
    cols = {"arch": "Kernel", "dtype": "Input format", "ran_s": "Variants that ran (label:chunk)",
            "cross_fwd_s": "Cross-chunk dev., outputs", "cross_grad_s": "Cross-chunk dev., gradients",
            "oracle_fwd_s": "Oracle dev., outputs", "oracle_grad_s": "Oracle dev., gradients",
            "definition_cross_s": "Definition's own cross-chunk dev. (f64)", "floor_s": "Floor",
            "cross_over_floor_s": "Cross / floor", "oracle_over_floor_s": "Oracle / floor",
            "verdict": "Diagnostic", "class": "Class"}
    parts = []
    for gpu, group in view.groupby("gpu", sort=False):
        env = kframe[kframe["gpu"] == gpu].iloc[0]
        ran = group[group["verdict"] != "NOT RUN"]
        controls = ran[ran["role"] != "candidate"]
        candidates = ran[ran["role"] == "candidate"]
        refused = group[group["refused"].astype(str) != ""].copy()
        refused["why"] = refused["refusal"].str.replace("|", "/", regex=False).str.replace("`", "'", regex=False)
        fwd_only = group[group["forward_only"].astype(str) != ""]
        missing = group[group["inputs_without_gradient"].fillna("").astype(str) != ""] if "inputs_without_gradient" in group else group.iloc[:0]
        sens = ran[ran["role"] == "sensitivity-control"]
        missed = sens[sens["classification"] == "scheduling"]
        caught = sens[sens["classification"] != "scheduling"]
        power = ("Injected defects flagged: " + (", ".join(f"`{a}` in {d}" for a, d in zip(caught["architecture"], caught["dtype"])) or "none")
                 + ". **Not flagged** (the test cannot see them at this precision): "
                 + (", ".join(f"`{a}` in {d}" for a, d in zip(missed["architecture"], missed["dtype"])) or "none") + ".")
        text = f"""### {gpu}

torch {env['torch']}, triton {env['triton']}, CUDA {env['cuda']}, `torch.compile` disabled, run {env['run_utc']}.

Controls and yardstick (the reference chunkwise Gated DeltaNet from this repository, run on the same GPU in the same formats):

{md_table(controls, cols)}

{power}

Kernels:

{md_table(candidates, cols)}
"""
        if len(refused):
            text += f"""
Variants a kernel refused, or that did not finish. These are recorded on purpose: what a kernel permits on a
given GPU bounds any chunk-length sweep. "Input aborted" means the container was stopped from outside while
the variant was still compiling; in an earlier run the same variants ran into the 25-minute cap.

{md_table(refused, {"arch": "Kernel", "dtype": "Input format", "refused": "Refused (label:chunk)", "why": "Error"})}
"""
        if len(fwd_only):
            text += ("\nForward only (no backward pass available, so no gradient comparison): "
                     + "; ".join(f"`{a}` ({d}): {m}" for a, d, m in zip(fwd_only["architecture"], fwd_only["dtype"], fwd_only["forward_only"])) + ".\n")
        if len(missing):
            text += ("\nInputs with an all-zero gradient in every variant that ran (for LaCT this is expected when the only "
                     "variant is a chunk of at least the sequence length, which never updates the fast weights): "
                     + "; ".join(f"`{a}` ({d}): {m}" for a, d, m in zip(missing["architecture"], missing["dtype"], missing["inputs_without_gradient"])) + ".\n")
        parts.append(text)
    pilot_csv = RESULTS / "pilot_precision.csv"
    solve_note = "Whether that is the source of the residual has not been checked."
    if pilot_csv.exists():
        pilot = pd.read_csv(pilot_csv)
        fixed = pilot[(pilot["regime"] == "fp32_ieee") & (pilot["chunk"] == 64) & pilot["kernel_oracle_dev"].notna()]
        if len(fixed):
            solve_note = (f"That is the source: with the solve also set to IEEE (done inside the precision pilot's "
                          f"containers, `results/pilot_precision.csv`), the forward pass at chunk 64 is "
                          f"{fmt(fixed['kernel_oracle_dev'].max())} from the float64 recurrence. Gradients were not "
                          f"compared in that configuration.")
    solve_csv = RESULTS / "equivalence_kernels_ieee_solve.csv"
    if solve_csv.exists():
        solve = pd.read_csv(solve_csv)
        solve = solve[(solve["architecture"] == "fla_chunk_gated_delta_rule") & (solve["chunk"] == 64)
                      & (solve["variant"] == "chunk") & solve["oracle_fwd_max_rel"].notna()]
        if len(solve):
            solve_note = (f"That is the source: with the solve also set to IEEE "
                          f"(`results/equivalence_kernels_ieee_solve.csv`), chunk 64 is "
                          f"{fmt(solve['oracle_fwd_max_rel'].max())} from the float64 recurrence in its outputs and "
                          f"{fmt(solve['oracle_grad_max_rel'].max())} in its gradients.")
    ieee_csv = RESULTS / "equivalence_kernels_ieee.csv"
    if ieee_csv.exists():
        both = []
        for label, frame in (("default", kframe), ("ieee", pd.read_csv(ieee_csv))):
            sub = frame[(frame["dtype"] == "float32") & frame["oracle_fwd_max_rel"].notna()].copy()
            sub["what"] = sub["variant"].str.replace(" (size fixed by the kernel)", "", regex=False) + ":" + sub["chunk"].astype(int).astype(str)
            both.append(sub.groupby(["architecture", "what"], sort=False)["oracle_fwd_max_rel"].max().rename(label))
        table = pd.concat(both, axis=1).dropna().reset_index()
        table["arch"] = "`" + table["architecture"] + "`"
        table["default_s"], table["ieee_s"] = table["default"].map(fmt), table["ieee"].map(fmt)
        ablation_title = "The cause is TF32: switching the dot products to IEEE removes it."
        ablation = ("Rerunning the float32 kernels with `TRITON_F32_DEFAULT=ieee` (worst output deviation from the "
                    "float64 oracle over both sequence lengths):\n\n"
                    + md_table(table, {"arch": "Kernel", "what": "Variant:chunk", "default_s": "Triton default (TF32)", "ieee_s": "IEEE"})
                    + "\n\n   The recurrent kernels are unchanged, as they have no dot products. The chunked kernels fall to "
                    "float32 rounding, with one exception: `chunk_gated_delta_rule` at chunk 64 keeps a residual about "
                    "three orders of magnitude above rounding. At chunk 64 that kernel takes a separate fused path "
                    "that requests TF32 explicitly for its triangular solve, which the environment switch does not "
                    "override. " + solve_note)
    else:
        ablation_title = "The likely cause is TF32, and it has not been isolated."
        ablation = "That rerun has **not** been done yet, so TF32 remains the likely cause and not a demonstrated one."
    return f"""## Result: Triton kernels (GPU, run on Modal)

The same question asked of the kernels people train with. They compute in float32 or lower and have no
float64 path, so the precision ladder is replaced by a **float64 oracle**: every kernel output and input
gradient is compared with the token-by-token recurrence (exact family) or the reference definition at the
same chunk length (test-time training), evaluated in float64 on the same already-rounded inputs. The
oracles are checked against the libraries' own reference ops in `tests/theory/test_oracle_conventions.py`.

Rule (`src/equivalence/kernels/verdict.py`, constants fixed before the first GPU run): with the floor equal
to the larger of the output format's measured quantisation floor and {FLOOR_ULPS32:g} float32 unit roundoffs,
**STRUCTURAL** if the cross-chunk deviation is at least {STRUCTURAL_X:g} floors, **ROUNDING** if the oracle
deviation is at most {ROUNDING_X:g} floors, **INCONCLUSIVE** otherwise. A kernel with a single runnable
variant gets no chunk-length verdict, only whether it matches its reference. Two sequence lengths are used,
1024 (every chunk size divides it) and 1000 (ragged tail); each cell is the worst over both and over all variants.

**Resolving power.** The sensitivity controls say how small a chunk-boundary defect this tier can see. In
float32 it flags a 1e-2 defect and misses 1e-4. In bfloat16 it misses even 1e-2: the reference chunkwise
algorithm itself, run in pure bfloat16, differs across chunk lengths by several percent. So a bfloat16
"scheduling" row means "no chunk effect larger than rounding in that format", which is a weak statement,
and the float32 and float16 rows are the informative ones.

{chr(10).join(parts)}
### What the kernel tier adds

1. **In float32 the chunked kernels are about 1e-3 from the exact recurrence; the recurrent kernels are not.**
   With float32 inputs the `fused_recurrent_*` kernels match the float64 oracle to float32 rounding, while
   `chunk_gated_delta_rule`, `chunk_simple_gla` and `chunk_gla` sit near 1e-3 from it at every chunk size.
   That is four orders of magnitude above float32 rounding and well below a structural difference, which is
   why the rule returns INCONCLUSIVE for them.
   Read with point 2, those INCONCLUSIVE rows are a precision effect of the kernel's default arithmetic and
   not a chunk boundary: the classification of these architectures as scheduling stands.
2. **{ablation_title}** On a GPU with tensor cores Triton's `tl.dot`
   defaults to TF32 for float32 operands (its documentation, Triton {kframe['triton'].iloc[0]}: "For devices that do have
   tensor cores, the default precision is tf32"). The chunked kernels use `tl.dot`; the recurrent ones do
   not. At the pinned commit the delta-rule kernels force IEEE in most dot products but refuse float32 input
   outright, the gated delta-rule forward kernel explicitly selects TF32 for its triangular solve, and the
   scalar-decay and shared chunk kernels leave the default. Triton exposes `TRITON_F32_DEFAULT=ieee`;
   `CHUNK_TRITON_F32=ieee uv run modal run modal_app.py --only ...` reruns the kernels that way.
   {ablation}
3. **The feasible chunk sizes are set by the kernel, the GPU and the Triton version together.** See the
   refusals table: a fixed list of sizes for the delta-rule kernels, powers of two for the scalar-decay
   kernel, no choice at all for `chunk_gla`, block sizes that exceed this GPU's shared memory for the
   scalar-decay and TTT-Linear kernels, and a LaCT fused kernel that no longer compiles because it uses
   block pointers, which this Triton version has removed.
4. **Large chunk sizes are slow before they fail.** A kernel that tiles a whole chunk has to be compiled
   for each new chunk size, and compile time grows quickly with the block. Stack dumps of the units that
   went quiet were all inside Triton's compiler (`make_llir`, `make_ptx`) under the autotuner, not in GPU
   execution. The first runs here made this worse by giving the containers Modal's default fractional CPU.
"""


def document(frame: pd.DataFrame, summary: pd.DataFrame, control_lines: list[str], controls_passed: bool,
             kernels_md: str = "", trained_md: str = "") -> str:
    cfg = ModelConfig()
    env = frame.iloc[0]
    view = summary.copy()
    view["arch"] = "`" + view["architecture"] + "`"
    view["paper"] = [SPECS[a].paper if a in SPECS else i for a, i in zip(view["architecture"], view["implementation"])]
    view["expectation"] = view["prior"].map({"exact": "exact"}).fillna("boundary") + " (" + view["mechanism"] + ")"
    for col in ("fwd_dev_float64", "grad_dev_float64", "fwd_dev_float32", "grad_dev_float32", "shrink_f32_to_f64"):
        view[col + "_s"] = view[col].map(fmt)
    view["class"] = "**" + view["classification"] + "**"
    cols = {"arch": "Architecture", "paper": "Source", "expectation": "Expected from the mathematics",
            "fwd_dev_float64_s": "Output dev. f64", "grad_dev_float64_s": "Gradient dev. f64",
            "fwd_dev_float32_s": "Output dev. f32", "shrink_f32_to_f64_s": "Shrink f32 to f64",
            "verdict": "Diagnostic", "class": "Class"}
    candidates = view[(view["tier"] == "reference") & (view["role"] == "candidate")]
    controls = view[(view["tier"] == "reference") & (view["role"] != "candidate")]
    library = view[view["tier"] == "library"]

    reasons = "\n".join(f"- `{a}`: {REASONS[a]}" for a in summary["architecture"] if a in REASONS)
    sixteen = view[(view["tier"] == "reference") & (view["role"] == "candidate")].copy()
    for col in ("fwd_dev_bfloat16", "fwd_dev_float16"):
        sixteen[col + "_s"] = sixteen[col].map(fmt)
    sixteen_cols = {"arch": "Architecture", "class": "Class", "fwd_dev_bfloat16_s": "Output dev. bfloat16",
                    "fwd_dev_float16_s": "Output dev. float16", "fwd_dev_float32_s": "Output dev. float32"}
    non_finite = frame[frame["status"].isin(["non-finite gradient", "non-finite output"])]
    nf = (non_finite.groupby(["tier", "architecture", "dtype", "status"]).size().reset_index(name="rows")
          if len(non_finite) else pd.DataFrame())
    errors = frame[frame["status"].astype(str).str.startswith(("error", "unsupported"))]

    return f"""# Classification: is chunk length a schedule or part of the model?

Generated by `uv run python -m src.equivalence.classify` from `results/equivalence.csv`
(run {env['run_utc']}, torch {env['torch']}, {env['platform']}). Do not edit the tables by hand; rerun the script.

**Status of this document.** The measurements and the decision rule are mechanical. The one-line reasons
are statements of mechanism that each have a numerical check in `tests/theory/`, but the chunkwise
derivations behind them have not yet been re-derived by hand, which the roadmap asks for before the paper
leans on them. Read the "Scope and limits" section before quoting any row.

## What was run

One checkpoint per architecture and one batch, forward and backward at several chunk lengths, no training.

- Model: {cfg.n_layers}-layer language model, width {cfg.d_model}, {cfg.n_heads} heads, vocabulary {cfg.vocab},
  seeded random initialisation saved once to `checkpoints/` and reused (sha256 prefixes are in the CSV).
- Batch: {cfg.batch} sequences of {cfg.seq_len} tokens, fixed. Loss: next-token cross-entropy.
- "Outputs" are the logits; "gradients" are the gradient of the loss with respect to every weight.
- Deviation: the largest absolute difference from the reference chunk length (the smallest one the layer
  supports), divided by the largest absolute reference value. Gradients are concatenated over all weights.
- Precisions: bfloat16, float16, float32 and float64, the whole model cast to the format.
  float64 is the precision the classification is read at.
- Chunk lengths: 1, 4, 16, 64, 256 and 512 (the full sequence), plus 48 and 100, which do not divide the
  sequence and exercise the ragged last chunk. HLA starts at 16.

## Decision rule

The rule lives in `src/equivalence/precision_scaling.py`. Its thresholds were written down before the full
sweep ran, but after a float64 smoke test of the layers had been seen, so they are not blind to the data.
The measured gaps are wide enough that the classification does not depend on where exactly they sit.

With d32 and d64 the worst deviation over chunk lengths, outputs and gradients together:

- **ROUNDING** if d64 <= {TOL64:g} and d32/d64 >= {SHRINK_MIN:g}. The difference collapses when precision rises, so it
  is floating-point re-association. Class: **scheduling**.
- **STRUCTURAL** if d64 > {TOL64:g} and d32/d64 <= {PERSIST_MAX:g}. The difference does not care about precision, so the
  chunk length changes the function. Class: **architectural** if a named mechanism makes the boundary
  load-bearing, **implementation bug** if the chunkwise form is provably exact.
- **INCONCLUSIVE** otherwise. Class: **unresolved**, to be investigated and never passed silently.

## Controls

{"All controls pass." if controls_passed else "**CONTROLS FAILED. Nothing below may be read until the harness is fixed.**"}

```
{chr(10).join(control_lines)}
```

{md_table(controls, cols)}

The injected-defect rows are a provably exact kernel (Gated DeltaNet, same checkpoint) with a deliberate
error: the carried state scaled by (1 + delta) at each chunk boundary, or the carried state missing the last
token's gate. "implementation bug" and "unresolved" both count as flagged. The delta = 1e-9 row marks the
sensitivity floor: in float64 the test still sees it, in float32 it is buried in rounding, and the rule
refuses to call it either way.

## Result: reference implementations

{md_table(candidates, cols)}

Why each one falls where it does:

{reasons}

![Figure 1](../results/figures/fig1_classification.png)

*Figure 1. Deviation from the smallest chunk length for outputs (left) and gradients (middle) in float64,
and the worst output deviation at each precision (right). Scheduling-class layers sit at rounding level and
fall with precision; architectural-class layers and the larger injected defects do not move. Unlabelled
minor ticks are the chunk lengths that do not divide the sequence (48, 100) and HLA's 32 and 128.*

### Deviation at each chunk length, outputs, float64

{per_chunk_table(frame, "reference", "float64", "fwd_max_rel")}

### Deviation at each chunk length, gradients, float64

{per_chunk_table(frame, "reference", "float64", "grad_max_rel")}

## Result: library code (pinned, pure-PyTorch paths only)

The same procedure applied at op level to code other people ship: fixed random inputs, the op's output, and
the gradient of a fixed linear probe with respect to every input. Sources:
flash-linear-attention at `{PINS['fla'][1][:12]}` and LaCT at `{PINS['lact'][1][:12]}`, fetched to `third_party/`.
float32 rows are the code as shipped. Several of these ops cast to float32 internally, so the float64 rows
run a copy of the same source with those casts lifted to float64, used only for the diagnostic.

{md_table(library, cols)}

### Deviation at each variant, outputs, float32 as shipped

{per_chunk_table(frame, "library", "float32", "fwd_max_rel")}

{kernels_md}
{trained_md}## Low precision (descriptive only; this is what the precision study in Milestones 2 and 4 has to measure)

{md_table(sixteen, sixteen_cols)}

For the scheduling class these are pure rounding effects, and they are not small in 16-bit arithmetic: the
function is the same, the computed numbers are not. This is the whole model cast to the format, including
recurrent states, which is harsher than mixed-precision training where states and accumulators are often
kept in float32. It says the precision study is worth running; it does not say what it will find.

## Things the test turned up that need a decision

1. **Kernel constraint on chunk length.** In flash-linear-attention at the pinned commit, `chunk_delta_rule`
   and `chunk_gated_delta_rule` accept only chunk sizes 16, 32 and 64, `chunk_simple_gla` requires a power
   of two, and `chunk_gla` sets its chunk size internally. For the scheduling class the "largest chunk the
   kernel permits" is therefore 64 in the delta-rule kernels, which bounds the precision study's sweep.
2. **Non-finite gradients in the library Titans closed form with a ragged tail.** `chunk_titans_linear`
   (`fla/ops/titans/naive.py`) returns finite outputs but NaN gradients with respect to `eta` and `alpha`
   whenever the sequence length is not a multiple of the chunk size (checked: T=256 with chunk 48, T=250 with
   chunk 64; T=240 with chunk 48 is fine). The token-loop version is finite in the same cases. A likely cause,
   from reading the code and not isolated further: the padded gates are zero and the closed form works in
   log space. This looks like a defect in the reference op and not a design choice; confirm before
   reporting it upstream.
3. **Titans token loop and closed form agree** at a fixed chunk length to rounding level when the length
   divides the sequence (`fla_titans_loop_vs_closed_form`), so the chunk dependence of that op is the
   intended inner-loop boundary and not an artefact of one of its two implementations.
4. **LaCT never uses the final chunk for an update**, and a chunk length of at least the sequence length
   disables test-time training entirely (verified in `tests/theory/test_boundary_family.py`). Any sweep that
   includes chunk >= context compares against a layer with a static MLP in place of the fast weights.

{"### Non-finite rows" + chr(10) + chr(10) + md_table(nf, {c: c for c in nf.columns}) + chr(10) + chr(10) + "The float16 backward pass is non-finite at every chunk length, the reference one included, in exactly the layers that L2-normalise their keys; this is consistent with float16's narrow exponent range and has not been isolated further. It is unrelated to chunking. Outputs in float16 are finite and are reported; bfloat16, the format mixed-precision training uses, is finite throughout." if len(nf) else ""}

{"### Rows that did not run" + chr(10) + chr(10) + md_table(errors, {"tier": "tier", "architecture": "architecture", "dtype": "dtype", "status": "status"}) if len(errors) else ""}

## Scope and limits

{KERNEL_SCOPE_TESTED if kernels_md else KERNEL_SCOPE_UNTESTED}
{"- **Mostly random initialisation.** The main tables use a seeded initialisation. The test was repeated on two small trained models (section above) with the same outcome; it has not been run on a large or publicly released checkpoint." if trained_md else "- **Random initialisation.** The checkpoint is a seeded initialisation, not a trained model. Whether chunk length changes the function does not depend on the weights for the mechanisms above, but the *size* of the rounding deviation does (gates near one, long effective memory). Repeat on a trained checkpoint before quoting magnitudes."}
- **HLA is written from the method section of the paper**, not from released code, and is labelled
  HLA-style. The inference-time gate threshold and the training regulariser are not part of the test.
- **FLASH** follows the paper's pseudo-code with one substitution: a fixed-size table of relative-position
  biases so the parameter count does not depend on the chunk length.
- **Small.** Width {cfg.d_model}, {cfg.n_layers} layers, {cfg.seq_len} tokens. Enough to decide the class; not evidence about scale.
- **The class label for a structural difference rests on the mathematical prior.** The test decides whether
  chunk length changes the function. Calling that architectural instead of a bug requires knowing that the
  mechanism is intended, which is the reason column and the theory tests, and ultimately the hand derivation.
"""


def main() -> int:
    frame = pd.read_csv(RESULTS / "equivalence.csv")
    summary = summarise(frame)
    summary.to_csv(RESULTS / "precision_scaling.csv", index=False)
    (RESULTS / "tables").mkdir(exist_ok=True)
    keep = ["tier", "architecture", "role", "implementation", "prior", "mechanism", "variants", "fwd_dev_float64",
            "grad_dev_float64", "fwd_dev_float32", "grad_dev_float32", "shrink_f32_to_f64", "verdict", "classification"]
    summary[keep].to_csv(RESULTS / "tables" / "classification.csv", index=False)
    passed, lines = check(frame)
    (RESULTS / "figures").mkdir(exist_ok=True)
    figure(frame, summary, RESULTS / "figures" / "fig1_classification")
    DOCS.mkdir(exist_ok=True)
    kernel_csv = RESULTS / "equivalence_kernels.csv"
    kernels_md = kernel_section(pd.read_csv(kernel_csv)) if kernel_csv.exists() else ""
    trained_csv = RESULTS / "equivalence_trained.csv"
    trained_md = trained_section(pd.read_csv(trained_csv)) if trained_csv.exists() else ""
    (DOCS / "classification.md").write_text(document(frame, summary, lines, passed, kernels_md, trained_md))
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary[["tier", "architecture", "fwd_dev_float32", "fwd_dev_float64", "grad_dev_float64",
                       "verdict", "classification"]].to_string(index=False, float_format=lambda v: f"{v:.2e}"))
    print("\n".join(lines))
    print("wrote docs/classification.md, results/tables/classification.csv, results/figures/fig1_classification.png")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
