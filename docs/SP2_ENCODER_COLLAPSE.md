# SP2 — why the richer encoders collapsed (root-cause investigation, 2026-07-05)

**Symptom:** at matched 15k-step budget, `masked` and `transformer` encoders collapsed to
degenerate generation (validity ~0.09, uniqueness ~0.08, route_len ~0 = single building
block, no reactions), while the study's `baseline` linear encoder was healthy
(validity 0.37, uniqueness 0.99, route_len 1.8).

## Confirmed (robust, from TensorBoard logs)
The richer encoders **underfit — they never fit even the training set**:

| variant | train loss_token (final) | val loss_token | val/loss |
|---|---|---|---|
| baseline | 0.0018 | 0.042 | 28 |
| masked | 0.27 (150x worse) | 0.57 | 1041 |
| transformer | 0.27 | 0.57 | 1039 |

Val metrics barely moved over 15k steps (masked val_token 1.29 -> 0.57; val_fp_bce
2169 -> 2074). So this is an **optimization failure**, not overfitting: the larger,
freshly-initialized MLP/transformer encoders + reinitialized cross-attention do not
converge under the matched budget (15k steps, lr 3e-4, full fine-tune with cross-attn
reinit), while the trivially-small linear (~0.9M params) does. Degenerate token head ->
premature END -> route_len 0.

## Hypothesis tested and REJECTED
"Encoder output magnitude at init blows up the pretrained decoder (scale mismatch)."
Measured fresh-encoder output std (baseline 0.031, masked 0.57, transformer 0.67, latent
1.11) — richer encoders are 18-36x larger. BUT directly scaling the masked encoder's
output down to baseline magnitude did NOT reduce the loss (one-batch token loss 2.35 ->
4.48; also batch-noisy: 8.81 vs 2.35 across batches). The step-0 loss=43201 in the log was
batch noise, not the mechanism. So init-scale is not the root cause; convergence is.

## Verdict
The collapse is an **optimization/undertraining failure of the richer fresh encoders at the
matched budget — NOT evidence that richer protein conditioning is worse.** A fair
architecture comparison requires making the richer encoders trainable.

## Recommended follow-up experiments (each needs a retrain; run in a persistent terminal)
Isolate the optimization lever, one at a time, on the `transformer` arm:
1. **Lower LR + warmup** (e.g., lr 1e-4 or 5e-5, linear warmup) — most likely fix for a
   large fresh module destabilized at lr 3e-4.
2. **Do NOT reinit cross-attention** (`decoder.reinit: false`) — keep the pretrained
   cross-attn so the fresh encoder attaches to a working decoder.
3. **Small-init the encoder's final projection** (near-zero) so training starts close to
   the pretrained-good regime and grows conditioning gradually (adapter-style).
4. **More steps** (30-50k) — if it's purely budget.
Success = train loss_token approaching baseline (~1e-2 or better) and non-degenerate
generation (route_len > 1, uniqueness > 0.5).
