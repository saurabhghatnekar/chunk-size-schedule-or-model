# Go or no-go

The roadmap's decision at the end of Milestone 2: **go** if a pilot shows a chunk effect above noise on the
architectural class, or a precision effect above noise on the scheduling class, and the compute budget covers
a reduced grid. This file fixes the criteria before the results exist. The sha256 of this file as it stood
when a pilot was launched is stored in every row that pilot produces.

## 1. Small precision pilot on the scheduling class

Written on 2026-11-18, before any training run.

**Question.** For a layer whose chunkwise form is exact, does the chunk length change the *trained* model by
more than seed noise, once arithmetic is not exact?

**Why it can be asked at all.** Milestone 1 measured that the shipped Gated DeltaNet kernel differs from the
exact recurrence by about 5e-3 in bfloat16, about 2e-3 with float32 inputs under Triton's TF32 default, and
about 3e-7 under IEEE float32. Training is chaotic, so two runs that differ only by rounding still end up
with different weights. A different end point is therefore not the question. The question is whether the
*distribution* of final quality shifts with chunk length.

**Design.** A small byte-level language model on enwik8 whose mixing layer calls the library kernel
`chunk_gated_delta_rule` directly. Everything is identical across cells except the two factors:

- chunk length: 16 and 64 (the smallest and largest the kernel accepts; 64 is the field default);
- arithmetic: `bf16` (bfloat16 autocast, what people train in), `fp32_tf32` (float32, Triton default),
  `fp32_ieee` (float32 with IEEE dot products);
- seeds: 3, each fixing initial weights and the order of training windows, shared across all cells.

That is 18 runs. Two more repeat one cell with the same seed, to measure the run-to-run floor.

**Primary metric.** Held-out loss (nats per byte) on a fixed set of validation windows at the end of
training, each model evaluated with the same reference forward pass (float32, recurrent kernel) so the
metric reflects the trained weights and not the arithmetic used to evaluate them. The loss under each
model's own training configuration is reported beside it.

**Quantities.**

- `d_s` = loss(chunk 64, seed s) - loss(chunk 16, seed s), within one arithmetic. Three values.
- `sigma_seed` = standard deviation of the final loss across seeds, pooled over the two chunk lengths of
  that arithmetic.
- `floor` = absolute loss difference between the two same-seed repeats.

**Decision, per arithmetic.**

- **EFFECT** if `|mean(d)| / (sd(d) / sqrt(3)) > 4.30` (two-sided 5% for a paired comparison with three
  pairs) **and** `|mean(d)| > sigma_seed`.
- Otherwise **NO EFFECT DETECTED**, reported as a bound: chunk length moved the final loss by at most
  `max |d_s|`, against a seed spread of `sigma_seed`.

**What counts toward go.** Only an EFFECT in `bf16`, because that is the arithmetic people train in. An
EFFECT in `fp32_tf32` that is absent in `fp32_ieee` would be evidence for the mechanism and would justify the
full-length precision study, but it is not by itself a go.

**What this pilot cannot show.** It is a model of about five million parameters trained for minutes. A null
here does not bound a 160M-parameter model trained to convergence, where rounding has far longer to
accumulate; it only says the effect is not large at small scale. Three seeds give a coarse estimate of the
seed spread, and the test above has low power. A positive result would need replication at the roadmap's
pilot scale before it is believed.

## 2. Pilot on the architectural class

Criteria to be written here before that pilot is run.

## 3. Compute budget for the reduced grid

Arithmetic to be written in `docs/compute_budget.md` before any grid is committed.

## Decision

Not taken. Nothing below this line may be filled in before the pilots above have reported.
