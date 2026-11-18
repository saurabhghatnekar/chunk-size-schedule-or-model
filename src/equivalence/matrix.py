"""Build architecture_matrix.xlsx: the architecture sheet from measurements, the literature sheet from the reading list.

    uv run python -m src.equivalence.matrix

The architecture sheet is regenerated from results/equivalence.csv. The literature
rows are maintained in this file; `verified` records that the identifier, title and
first author were checked against the arXiv API on the stated date, and `basis`
records what the scheduling-or-architectural entry actually rests on.
"""
from __future__ import annotations

import sys

import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from .archs import SPECS
from .classify import REASONS
from .kernels.cases import CASES as KERNEL_CASES
from .kernels.verdict import summarise_kernels
from .model import ROOT
from .precision_scaling import summarise

CHECKED = "2026-11-18"
TESTED = "equivalence test in this repository"
ROADMAP = "roadmap description; paper not re-read for this sheet"

LITERATURE = [
    # key, title, authors, year, arXiv, venue, role, chunk claim concerns, basis
    ("DeltaNet", "Parallelizing Linear Transformers with the Delta Rule over Sequence Length", "Yang et al.", 2024,
     "2406.06484", "NeurIPS 2024", "Premise of the scheduling class; derive the chunkwise form by hand",
     "Scheduling: the chunkwise form is a re-association of the recurrence", TESTED),
    ("Gated DeltaNet", "Gated Delta Networks: Improving Mamba2 with Delta Rule", "Yang et al.", 2024,
     "2412.06464", "ICLR 2025", "Premise of the scheduling class", "Scheduling: re-association", TESTED),
    ("GLA", "Gated Linear Attention Transformers with Hardware-Efficient Training", "Yang et al.", 2023,
     "2312.06635", "ICML 2024", "Candidate architecture", "Scheduling: re-association", TESTED),
    ("Mamba-2", "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality",
     "Dao and Gu", 2024, "2405.21060", "ICML 2024", "Premise of the scheduling class (SSD)",
     "Scheduling: block decomposition of one operator", TESTED),
    ("Titans", "Titans: Learning to Memorize at Test Time", "Behrouz et al.", 2024, "2501.00663", "preprint",
     "Premise of the architectural class; the citation for gradients frozen at the chunk start",
     "Architectural: inner-loop boundary", TESTED),
    ("LaCT", "Test-Time Training Done Right", "Zhang et al.", 2025, "2505.23884", "ICLR 2026 (per roadmap)",
     "Architectural class; large-chunk baseline B2", "Architectural: inner-loop boundary (apply then update per chunk)",
     TESTED + "; official op tested at op level"),
    ("TTT layers", "Learning to (Learn at Test Time): RNNs with Expressive Hidden States", "Sun et al.", 2024,
     "2407.04620", "preprint", "Architectural class", "Architectural: mini-batch inner loop",
     "library TTT-Linear reference op tested at op level"),
    ("Modular TTT", "Modular TTT: Rethinking Test-Time Training as Composable Modules", "Tang et al.", 2026,
     "2608.07110", "preprint", "Mechanism: update norm over a chunk grows with chunk size",
     "Architectural: inner-loop boundary", ROADMAP),
    ("E2E TTT", "End-to-End Test-Time Training for Long Context", "Tandon et al.", 2025, "2512.23675", "preprint",
     "Notes larger mini-batches improve parallelism and stability", "Architectural: inner-loop boundary", ROADMAP),
    ("Rethinking TTT", "Rethinking Expressivity and Efficiency in Test-Time Training", "Zhong et al.", 2026,
     "2608.21308", "preprint", "Test-time training family", "Architectural: inner-loop boundary", ROADMAP),
    ("HLA", "HLA: Expressive Hybrid Linear Attention via Chunk-Wise Dynamic Mixing", "Chen et al.", 2026,
     "2610.05842", "preprint", "Architectural-class evidence; LongBench-V2 25.25 / 29.62 / 26.64 / 26.24 at chunk 128 / 256 / 512 / 1024",
     "Architectural: chunk-level routing. Not evidence about an exact chunkwise kernel",
     TESTED + " (HLA-style layer written from the method section); ablation figures read from the arXiv HTML"),
    ("FLASH", "Transformer Quality in Linear Time", "Hua et al.", 2022, "2202.10447", "ICML 2022",
     "Architectural-class evidence; chunk 512 beats 256 beyond 1024 context (Appendix C.3, per roadmap)",
     "Architectural: quadratic attention within a chunk, linear across", TESTED + " (from the paper's pseudo-code)"),
    ("StateFlow", "StateFlow: Sequence Pipeline Parallelism for Long-Context Modeling with Linear Recurrence",
     "Zhao et al.", 2026, "2608.06838", "preprint", "Prior art that treats chunking as scheduling, correctly",
     "Scheduling: linear recurrence", ROADMAP),
    ("Placement Semantics", "Placement Semantics for Distributed Deep Learning: A Systematic Framework for Analyzing Parallelism Strategies",
     "Mehta", 2026, "2601.02311", "preprint (single author, unrefereed)", "Formalisation of when a configuration reproduces single-device training",
     "Neither: no treatment of chunked recurrence", ROADMAP),
    ("Alpa", "Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning", "Zheng et al.", 2022,
     "2201.12023", "OSDI 2022", "Automatic parallelism; the gap is the absence of a joint optimisation",
     "Neither: searches parallelism plans for a given model and does not choose chunk length", ROADMAP),
    ("Galvatron", "Galvatron: Efficient Transformer Training over Multiple GPUs Using Automatic Parallelism", "Miao et al.", 2022,
     "2211.13878", "VLDB 2023", "Automatic parallelism; same position as Alpa",
     "Neither: searches parallelism plans for a given model and does not choose chunk length", ROADMAP),
    ("xLSTM scaling", "xLSTM Scaling Laws: Competitive Performance with Linear Time-Complexity", "Beck et al.", 2025,
     "2510.02228", "preprint", "Milestone 2: FLOP accounting for linear-time models", "Not a chunk claim", ROADMAP),
    ("Chinchilla", "Training Compute-Optimal Large Language Models", "Hoffmann et al.", 2022, "2203.15556",
     "NeurIPS 2022", "Milestone 2: structure of a joint optimisation argument", "Not a chunk claim", ROADMAP),
    ("Test-time scaling", "Test-Time Scaling Makes Overtraining Compute-Optimal", "Roberts et al.", 2026, "2604.01411",
     "preprint", "Milestone 2: how much validation error a compute-optimal claim reports", "Not a chunk claim", ROADMAP),
]


def architecture_sheet() -> pd.DataFrame:
    frame = pd.read_csv(ROOT / "results" / "equivalence.csv")
    summary = summarise(frame)
    rows = []
    for _, r in summary.iterrows():
        spec = SPECS.get(r["architecture"])
        if r["verdict"] == "INCONCLUSIVE":
            exact = "unclear"
        elif r["role"] == "sensitivity-control":
            exact = "yes for the clean kernel; this row carries a deliberate defect"
        else:
            exact = "yes" if r["prior"] == "exact" else "no"
        rows.append({
            "Architecture": spec.title if spec else r["architecture"],
            "Key": r["architecture"],
            "Paper": spec.paper if spec else "",
            "Implementation and version": r["implementation"] if r["tier"] == "library" else "this repository, src/equivalence/archs",
            "Tier": r["tier"],
            "Chunkwise Form Exact in Exact Arithmetic": exact,
            "Reason": r["mechanism"],
            "Forward Agreement Across Chunk Lengths (max relative deviation)": r["fwd_dev_float64"],
            "Gradient Agreement (max relative deviation)": r["grad_dev_float64"],
            "Forward deviation, float32": r["fwd_dev_float32"],
            "Precision Used for the Test": "float64" + (" (library float32 casts lifted)" if r["tier"] == "library" else ""),
            "Classification": r["classification"] if r["role"] == "candidate" else f"{r['classification']} ({r['role']})",
            "Notes": REASONS.get(r["architecture"], ""),
        })
    kernel_csv = ROOT / "results" / "equivalence_kernels.csv"
    if kernel_csv.exists():
        titles = {c.name: c for c in KERNEL_CASES}
        for _, r in summarise_kernels(pd.read_csv(kernel_csv)).iterrows():
            case = titles.get(r["architecture"])
            if r["verdict"] == "NOT RUN":
                continue
            rows.append({
                "Architecture": case.title if case else r["architecture"],
                "Key": r["architecture"],
                "Paper": "",
                "Implementation and version": f"{r['implementation']} on {r['gpu']}",
                "Tier": "kernel",
                "Chunkwise Form Exact in Exact Arithmetic": "unclear" if r["verdict"] in ("INCONCLUSIVE", "ONE VARIANT") else ("yes" if r["prior"] == "exact" else "no"),
                "Reason": r["mechanism"],
                "Forward Agreement Across Chunk Lengths (max relative deviation)": r["cross_fwd"],
                "Gradient Agreement (max relative deviation)": r["cross_grad"],
                "Forward deviation, float32": r["cross_fwd"] if r["dtype"] == "float32" else None,
                "Precision Used for the Test": f"{r['dtype']} inputs, {r['output_dtype']} outputs; float64 oracle deviation {r['oracle_fwd']:.1e}, floor {r['floor']:.1e}",
                "Classification": r["classification"] if r["role"] == "candidate" else f"{r['classification']} ({r['role']})",
                "Notes": (case.notes if case else "") + (f" Refused or unfinished: {r['refused']}." if r["refused"] else ""),
            })
    return pd.DataFrame(rows)


def literature_sheet() -> pd.DataFrame:
    cols = ["Key", "Title", "Authors", "Year", "arXiv", "Venue", "Role in this project",
            "Does This Paper's Chunk Claim Concern a Scheduling or an Architectural Mechanism", "Basis for that entry"]
    frame = pd.DataFrame(LITERATURE, columns=cols)
    frame.insert(5, "Identifier verified against arXiv", f"yes, title and first author via arXiv API on {CHECKED}")
    return frame


def main() -> int:
    out = ROOT / "architecture_matrix.xlsx"
    sheets = {"Architectures": architecture_sheet(), "Literature": literature_sheet()}
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False)
            ws = writer.sheets[name]
            ws.freeze_panes = "B2"
            for cell in ws[1]:
                cell.font = Font(bold=True, color="FFFFFF")
                cell.fill = PatternFill("solid", fgColor="1F3864")
                cell.alignment = Alignment(wrap_text=True, vertical="top")
            ws.row_dimensions[1].height = 48
            for i, column in enumerate(frame.columns, start=1):
                longest = max([len(str(column))] + [len(str(v)) for v in frame[column]])
                ws.column_dimensions[get_column_letter(i)].width = min(max(12, longest * 0.9), 60)
                for cell in ws[get_column_letter(i)][1:]:
                    cell.alignment = Alignment(wrap_text=True, vertical="top")
                    if isinstance(cell.value, float):
                        cell.number_format = "0.0E+00"
    print(f"wrote {out.relative_to(ROOT)}: {len(sheets['Architectures'])} architecture rows, {len(sheets['Literature'])} literature rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
