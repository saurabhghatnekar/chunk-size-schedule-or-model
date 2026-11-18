# Claims audit

One row per claim the paper may make. Status is one of **verified** (checked against the source),
**measured** (a number in `results/`), **derived** (follows from mathematics that has been worked through),
**open**, or **withdrawn**. Started on 2026-11-18 from Appendix B of the roadmap; keep it current.

A claim is not allowed into the abstract or introduction unless its row names a file and a column.

## Premises carried over from the roadmap (Appendix B)

| Claim | Status | Support | What is still owed |
|---|---|---|---|
| The chunkwise form of gated linear attention, DeltaNet / Gated DeltaNet and Mamba-2 SSD is an exact re-association of the recurrence | verified in the literature (roadmap); **measured** here | `results/tables/classification.csv`, rows `gla`, `deltanet`, `gated_deltanet`, `mamba2`, columns `fwd_dev_float64`, `grad_dev_float64`, `verdict`; `tests/theory/test_exact_family.py` | Hand derivation by the author (roadmap, Milestone 1 reading item 1). The code's docstrings state the algebra but that is not a substitute |
| In test-time training, gradients inside a chunk are evaluated at the weights frozen at the chunk start, so chunk length changes the function | verified from the Titans formulation (roadmap); **measured** here | rows `ttt_titans`, `lact`, `fla_ttt_linear`, `fla_titans_sequential`, `lact_block_causal_swiglu`; `tests/theory/test_boundary_family.py` | Cite Titans for the formulation |
| HLA reports LongBench-V2 25.25 / 29.62 / 26.64 / 26.24 at chunk 128 / 256 / 512 / 1024 | verified (roadmap, Figure 3(a)); figures re-read from the arXiv HTML on 2026-11-18 | arXiv 2610.05842 | Check against the PDF figure itself before quoting |
| HLA's chunk dependence comes from chunk-level routing, so it is not evidence about an exact kernel | **measured** on an HLA-style layer | row `hla`; unit-routing and single-chunk limits in `tests/theory/test_boundary_family.py` | The layer is written from the paper's method section, not from released code |
| FLASH states chunk size affects quality and cost, 512 beating 256 beyond 1024 context | verified (roadmap, Appendix C.3) | arXiv 2202.10447 | Not re-read here |
| FLASH's chunk dependence comes from quadratic attention within a chunk and linear across | **measured** | row `flash`; single-chunk limit in `tests/theory/test_boundary_family.py` | |
| Earlier test-time training updates fast weights every 16 or 64 tokens and sits below five percent of peak FLOPS; LaCT uses chunks of 2K to 1M | verified (roadmap) | arXiv 2505.23884 | The roughly seventy percent utilisation figure needs table-level confirmation |
| Placement Semantics gives two necessary and sufficient conditions and does not treat chunked recurrence | verified (roadmap) | arXiv 2601.02311 | Single-author unrefereed preprint; cite as a formalisation, not a standard |
| StateFlow reports 2.22x throughput and 2.45x memory reduction and treats chunking as scheduling | verified (roadmap) | arXiv 2608.06838 | Not re-read here |
| Alpa and Galvatron choose architectural chunk length | **withdrawn** | | State the gap as the absence of a joint optimisation |
| Chunk length changes the computed function across all chunked recurrent architectures | **withdrawn** | contradicted by the scheduling-class rows above | |
| Hardware-dependent throughput implies a hardware-dependent optimum | **open**, does not follow automatically | | `docs/hardware_condition.md` before any test (Milestone 5) |
| Quality at fixed tokens determines the wall-clock optimum | **withdrawn** as an inference | | Joint model, then measure at equal wall-clock (Milestones 2 and 4) |
| Algebraic equivalence survives bfloat16 over a full training run | **open**; this project measures it | first indication only: `results/tables/classification.csv` columns via `results/precision_scaling.csv` `fwd_dev_bfloat16` | The precision study. A single forward pass in pure bfloat16 is not a training run |
| ICML 2027 abstract 16 January, full paper 22 January | **provisional** | | Confirm on icml.cc in week eleven |

## New claims from Milestone 1

| Claim | Status | Support | Caveat |
|---|---|---|---|
| A one-checkpoint, one-batch test with a precision-scaling diagnostic separates scheduling from architectural chunk length | **measured** | `results/equivalence.csv`, `results/precision_scaling.csv`, `docs/classification.md` | Reference implementations and pure-PyTorch library ops. For Triton kernels see the kernel-tier rows below |
| The negative control (softmax attention at varying tile size) agrees to machine precision | **measured** | row `softmax_tile` | |
| Injected defects down to a relative 1e-6 per chunk boundary are flagged as structural, and 1e-9 is flagged as unresolved | **measured** | rows `perturbed_*` | Sensitivity floor depends on model size and depth |
| A single chunk spanning the sequence turns each architectural-class layer into a chunk-free one (Titans-style memory: Hebbian linear attention; LaCT: no update; FLASH: quadratic attention; HLA: Gated DeltaNet) | **measured** numerically; **derived** only informally | `tests/theory/test_boundary_family.py` | Hand derivation owed before it is stated as a proposition |
| With momentum and decay off, Titans-style memory at chunk length 1 is the delta rule with beta = 2 theta | **measured** numerically | `tests/theory/test_boundary_family.py::test_ttt_chunk_one_without_momentum_is_the_delta_rule` | Same |
| flash-linear-attention's delta-rule kernels accept only chunk sizes 16, 32 and 64 | verified by reading the source at the pinned commit | `third_party/fla/fla/ops/delta_rule/chunk.py`, `gated_delta_rule/chunk.py` | Pinned commit only |
| The library Titans closed-form op yields NaN gradients when the chunk does not divide the sequence | **measured** | `results/equivalence.csv`, `fla_titans_chunkwise`, status `non-finite gradient` | Reference op, pinned commit; confirm before reporting upstream |

## New claims from the kernel tier (one GPU: NVIDIA A10, Triton 3.9.0)

| Claim | Status | Support | Caveat |
|---|---|---|---|
| In bfloat16 and float16 the DeltaNet, Gated DeltaNet, scalar-decay and GLA chunk kernels agree across chunk sizes and with a float64 oracle to within a few quantisation floors | **measured** | `results/tables/classification_kernels.csv`, columns `cross_over_floor`, `oracle_over_floor` | In bfloat16 the test misses a 1e-2 per-boundary defect, so this is a weak bound. One GPU |
| With float32 inputs the chunked Gated DeltaNet, scalar-decay and GLA kernels are about 1e-3 from the float64 recurrence at every chunk size, while the recurrent kernels match it to float32 rounding | **measured** | `results/equivalence_kernels.csv`, column `oracle_fwd_max_rel`, rows with `dtype == float32` | One GPU, one Triton version |
| That float32 gap is caused by Triton's TF32 default for float32 dot products | **open** (likely) | Triton documentation of `tl.dot`; the chunked kernels use `tl.dot` and the recurrent ones do not (source at the pinned commit) | Not isolated: the `TRITON_F32_DEFAULT=ieee` rerun has not been done |
| The TTT-Linear kernels and the LaCT op are chunk-dependent on the GPU and match their per-chunk float64 reference | **measured** | rows `fla_chunk_ttt_linear`, `fla_fused_chunk_ttt_linear`, `lact_block_causal_swiglu_autocast` | TTT-Linear: only chunk 16 and 64 run on this GPU, and 64 is forward-only |
| Which chunk sizes a kernel accepts depends on the kernel, the GPU and the Triton version | **measured** | `results/equivalence_kernels.csv`, rows whose `status` starts with `error` | Pinned commits; an H100 has a larger shared-memory limit and may accept more |
| LaCT's fused Triton kernel does not compile on Triton 3.9.0 (it uses block pointers) | **measured** | row `lact_fused_triton_kernel`, `status` | Pinned commit |
