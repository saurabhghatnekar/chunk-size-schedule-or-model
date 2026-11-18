# Go or no-go, section 3: confirmatory pilot on the architectural class

Written on 2026-11-18, after the first architectural pilot and before any run of this one. The sha256 of this
file as it stood at launch is stored in every row the confirmatory pilot produces.

## Why a second pilot

The first architectural pilot ([go_no_go_architectural.md](go_no_go_architectural.md)) came out
UNINFORMATIVE under its rule, which treated runs as independent draws. Looked at by seed, the same data showed
the loss rising with chunk length in every seed. That reading was chosen after seeing the data, so it is a
hypothesis and not a result. This pilot tests that hypothesis on data that did not suggest it.

**Hypothesis, fixed now.** At an equal number of training steps, a LaCT-style layer trained with chunk length
32 reaches a lower held-out loss than the same model trained with chunk length 512.

## Rung A: the test

Same model, data, optimiser and evaluation as the first architectural pilot (5.2M parameters, 1024-byte
context, attention window 64, 2,000 steps of 32 windows).

- **New seeds 5 to 10**, never used before in this project: six seeds.
- Cells: chunk 32, chunk 512, and chunk 1024 (the anchor with test-time training off). 18 runs.
- **Run-to-run floor:** seed 5 is run two more times at chunk 32 and two more times at chunk 512. The floor
  is the largest absolute difference in final loss between any two runs of one configuration and seed.

Primary quantity: `d_s` = loss(chunk 512, seed s) minus loss(chunk 32, seed s), for the six new seeds, using
the first run of each configuration.

**Decision.**

- **EFFECT** if the one-sided paired t statistic `mean(d) / (sd(d) / sqrt(n))` exceeds the 1% critical value
  (3.365 for six seeds; 3.747 for five; 4.541 for four) **and** `mean(d)` exceeds the run-to-run floor.
- Otherwise **NO EFFECT**: the hypothesis is not confirmed.

The same statistic for the anchor against chunk 32 is reported beside it (whether test-time training helps at
all) and is not part of the decision.

## Rung B: does the direction survive a larger model

The same design at about 39M parameters (width 512, 8 heads, 12 layers), batch 16 because 32 does not fit
a 24 GB GPU with test-time training on, 2,000 steps, seeds 5, 6 and 7, cells chunk 32, 512 and 1024. Nine runs.

- **CONSISTENT** if `d_s` is positive in all three seeds and its mean exceeds the rung A floor.
- **NOT CONSISTENT** otherwise.

Three seeds cannot carry a test of their own (all three positive happens one time in eight by chance). Rung B
is evidence about direction at a larger size, nothing more.

## How the two rungs and the budget combine into a decision

Fixed now, before the data:

| Rung A | Rung B | Decision on the science |
|---|---|---|
| EFFECT | CONSISTENT | **GO**: chunk length moves trained quality on the architectural class, and it does not vanish at the first step up in size |
| EFFECT | NOT CONSISTENT | **WEAK GO**: the effect exists at small scale and may not survive scale. Proceed only as far as the roadmap-scale pilot, not the grid |
| NO EFFECT | either | **NO-GO for the full paper**: neither the precision branch nor the architectural branch shows an effect above noise. Fall back to the shorter paper the roadmap describes (the test, the classification, and the negative results) |

The roadmap's third condition is that the compute budget covers a reduced grid. That part is arithmetic from
measured throughput, written in `docs/compute_budget.md`. Whether the resulting figure is affordable is the
author's decision; a GO above means the science supports spending it.

## What this cannot show

- Both rungs are far below the roadmap's pilot scale (160M parameters) and LaCT's intended chunk lengths.
- The comparison is at equal steps. Chunk 32 costs several times more compute per token than chunk 512, so a
  lower loss at equal steps does not mean a better model at equal wall-clock. That is the roadmap's separate
  wall-clock question.
- One layer type, one dataset, one attention window.
