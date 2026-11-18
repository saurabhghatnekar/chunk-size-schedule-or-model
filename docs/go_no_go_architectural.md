# Go or no-go, section 2: small pilot on the architectural class

This is section 2 of [go_no_go.md](go_no_go.md). It lives in its own file so that the earlier file stays
byte-identical: the precision pilot's result rows store that file's hash. The sha256 of *this* file as it stood
when the architectural pilot was launched is stored in every row that pilot produces.

Written on 2026-11-18, before any training run of this pilot.

**Question.** For a layer where the chunk boundary is part of the function, does the chunk length change the
*trained* model's quality by more than run-to-run noise?

**Why it is not obvious.** Milestone 1 showed that a LaCT-style layer computes a different function at each
chunk length (4% to 100% of the output scale on an untrained model). Training can absorb such differences:
the slow weights adapt to whatever chunk length they are trained with. A different function is therefore not
the same as a different quality.

**Design.** A small byte-level language model on enwik8. Each block is a LaCT-style layer: a causal
sliding-window softmax attention plus the official LaCT fast-weight op (`block_causal_lact_swiglu`, pinned
commit, run under its own bfloat16 autocast), summed. Everything is identical across cells except the
test-time-training chunk length:

- sequence length 1024 bytes; attention window fixed at 64 for every cell, so the chunk length is the only
  thing that changes;
- chunk length 32, 128 and 512: the three cells the decision is made on;
- chunk length 1024, the full sequence, as an **anchor**: with one chunk the official op never updates its
  fast weights, so this cell is the same model with test-time training switched off;
- five runs per cell, seeds 0 to 4, each seed fixing the initial weights and the order of training windows.

That is 20 runs. Parameter count, optimiser, schedule, data order per seed and evaluation are identical.

**Primary metric.** Held-out loss (nats per byte) on a fixed set of validation windows at the end of
training, each model evaluated at the chunk length it was trained with.

**Noise.** The precision pilot found that rerunning one configuration with the same seed moves the final loss
about as much as changing the seed. Runs are therefore treated as independent draws, not paired by seed. The
noise scale is the pooled within-cell standard deviation, `s_within`.

**Decision.**

- **EFFECT** if a one-way analysis of variance across the three decision cells gives `F` above the 1%
  critical value (6.93 for 15 runs; 7.21, 7.56, 8.02 if one, two or three runs are lost) **and** the range of
  the three cell means exceeds `2 * s_within`.
- Otherwise **NO EFFECT DETECTED**, reported as a bound: the cell means span at most the observed range,
  against a within-cell spread of `s_within`.

**Check that the pilot is informative.** If none of the three decision cells differs from the anchor by more
than `2 * s_within`, the layer is not using test-time training at this scale, and the pilot says nothing about
chunk length either way. That outcome is reported as **UNINFORMATIVE**, not as a null.

**What counts toward go.** An EFFECT. The direction (which chunk length is better) is reported but is not
part of the criterion.

**Also recorded, not part of the decision.** Each trained model's loss when evaluated at the other chunk
lengths, which is the deployment question of inference-time chunk length differing from training-time.

**What this pilot cannot show.** It is a model of a few million parameters trained for minutes at a context of
1024 bytes, far below the chunk lengths LaCT is designed for (thousands to a million tokens). An effect here
shows that chunk length can matter for trained quality; it does not say how large the effect is at scale or
which chunk length is best there. The fixed attention window is a design choice: LaCT usually ties the window
to the chunk, which would change two things at once.
