# Experimental protocols, version 2

Numbering continues the existing e-series. E0 to E4 are the gates and pilots
already run; the main collections are `collect_decay` and `collect_splice_v3`.
Everything below is new work.

Standing conventions inherited from version 1, unchanged unless stated:
eight-way forced choice so chance is exactly 12.5%; trials built in blocks of
eight sharing one candidate list with each candidate correct exactly once;
letterboxed stimuli on a common canvas; 512-dimensional Johnson-Lindenstrauss
projection of the recurrent state; dual-form ridge with block-grouped k-fold
cross validation; exact binomial tails; a permutation null on every headline
number.

---

## E5. The cliff, predicted from the weights

**Why.** The paper currently reports that decodable image content collapses
within 32 tokens. It does not say why. A reviewer reads that as an observation
about one checkpoint. If the collapse can be predicted from parameters that were
never fitted to it, the observation becomes a property of the architecture.

**The quantity.** In Mamba2 the state update is

    S_t = S_{t-1} exp(Δ_t A_h) + Δ_t x_t B_t^T

with `A_h` a scalar per head, stored as `A_log` so that `|A_h| = exp(A_log_h)`,
and `Δ_t,h = softplus(dt_raw + dt_bias)` clamped to the configured time-step
limits. Retention of anything written before a run of k tokens is therefore

    R_h(k) = exp(-|A_h| Σ_{t=1..k} Δ_t,h)

and the per-head half-life in tokens is `ln 2 / (|A_h| Δ̄_h)`.

Both ingredients are already within reach. `A_log` is a stored weight. `Δ̄_h` is
the mean of the dt slice, which is the final 112 channels of the `in_proj`
output that `capture_projections` already hooks. The only code change needed is
a capture variant that accumulates dt across every filler position instead of
storing the last position only.

**Protocol.**

1. Extract `A_log` from all 81 mixers. Report the distribution of `|A_h|` by
   layer depth.
2. Run one decay collection with dt accumulated over the filler region. Compute
   `Δ̄_h` per (layer, head).
3. Compute the predicted half-life for all 81 × 112 = 9,072 heads. Report the
   median, the interquartile range, and the fraction of heads falling below 32
   tokens, between 32 and 512, and above 512.
4. Plot mean predicted retention against distance on the same axes as the
   measured probe accuracy. One curve is derived from weights and gate
   statistics alone and never sees the probe; the other is the measurement.

**The sharp test, and the reason this experiment cannot fail.** Steps 1 to 4
give a curve comparison, which is suggestive but not decisive. The decisive
version is a within-model manipulation. Split the heads into terciles by
predicted half-life and train separate probes on the state slices of the fastest
and slowest terciles only. Decay theory makes a quantitative prediction: the
slow-head probe must retain decodable identity to materially larger distances
than the fast-head probe, and the crossing points should sit near the respective
predicted half-lives. If the two curves are indistinguishable, decay is not the
mechanism, and we take the branch below.

**Branch E5b, interference.** If measured survival is far shorter than predicted
retention, the state is not forgetting the image, it is being overwritten by the
`Δ_t x_t B_t^T` write term. That is the more surprising result and we would lead
with it. Two measurements settle it:

- *Write-to-state magnitude ratio.* Per token, compute the Frobenius norm of the
  added write term against the norm of the running state. If writes are
  comparable in magnitude to the state, the image is being swamped rather than
  decaying.
- *Filler comparison.* Run the same distances with four fillers: natural prose,
  a repeated single token, whitespace padding, and the visual tokens of a second
  image. Decay predicts survival ordered exactly by each filler's own predicted
  retention, since Δ is input dependent. Interference predicts survival ordered
  by how much the filler writes, which is a different ordering. Whichever
  ordering the data follow identifies the mechanism.

**Outputs.** Figure: predicted retention against measured decodability. Figure:
head-stratified decay curves. Table: half-life distribution by depth.

**Cost.** About one GPU hour plus analysis. Everything else is reuse.

---

## E6. Scale replication inside the family

**Why.** The cheapest available answer to "n = 1".

**Protocol.** `Zyphra/Zamba2-VL-2.7B` and `Zamba2-VL-1.2B`, both confirmed to
exist. Identical materials, distances, probe width and seeds. For each model
report the accuracy at distance zero per family, the distance at which identity
first falls inside the chance confidence interval, and the behaviour curve. Run
E5 on each as well, so that if the cliff moves with scale we can check whether
the predicted half-life moves with it in the same direction.

**Claim discipline.** Three scales support "present at every scale tested". They
do not support a scaling law, and the paper must not imply one.

**Risk.** If the VL variants do not exist at both small scales, substitute the
text-memory version of the protocol (E8) on the plain Zamba2 language models,
which still supports a scale claim about the recurrent channel.

**Cost.** Two model downloads, two collections. Mechanical once the telemetry
migration is done.

---

## E7. The architecture contrast, and the positive control we are missing

**Why, first reason.** A pure-SSM VLM has no attention layers, so it has nowhere
else to keep the image. If its recurrent state also emptied within 32 tokens it
would be blind, and it is not. Measuring how long it does hold the image tells
us whether the 32-token lifetime is a property of Mamba2 or a property of having
an attention channel to offload onto.

**Why, second reason, and this one matters more.** The paper's central causal
result is a null: swapping all 81 recurrent states changes nothing. Nulls
require a positive control showing that the manipulation is capable of producing
an effect at all. Today we have only a weaker check, that the swapped tensors
genuinely differ, verified 90 out of 90. A sceptical reviewer will note that
"the tensors changed" is not "the intervention can move behaviour". On a
pure-SSM VLM the recurrent swap **must** flip the answer, because there is no
other channel. If it does, our instrument is validated. If it does not, our
instrument is broken and we need to know that before publishing, not after.

This is the strongest argument for running E7 and it should be stated in the
paper in these terms.

**The comparable statistic.** Models differ in absolute accuracy, so compare a
normalised **decoupling gap**: behaviour accuracy at distance d minus probe
accuracy at distance d, over the same materials. In Zamba2-VL-7B this gap opens
to roughly 78 points by d = 32. In a pure-SSM model it should stay near zero,
because the state is the only memory. One number per model, one figure across
all models.

**Candidates, and an important distinction.**

| Model | Family | Role |
|---|---|---|
| Cobra | Mamba1 backbone, no attention | Pure-SSM contrast and instrument positive control |
| VL-Mamba | Mamba1 backbone, no attention | Backup for the above |
| MaTVLM (`hustvl/MaTVLM_0_25_Mamba2`) | Mamba2 layers distilled into TinyLLaVA, still hybrid | Different family and different training recipe, **not** a pure-SSM control |

These are two different jobs and one model cannot do both. MaTVLM answers "is
this specific to Zyphra's recipe"; Cobra or VL-Mamba answers "is this about
Mamba, or about having attention available". We need one of each.

**Instrument work.** Mamba1 does not share Mamba2's geometry: `A` is per channel
rather than per head, and B and C are not group shared. The read operator
`y = S C` still applies but the shapes and the grouping differ. Budget one day
to port `read_state`, `project_state` and the splice, and to re-run the E0-style
gate on the new model before trusting any number from it.

**Cost.** Three model downloads, three collections, one instrument port.

---

## E8. Is this about vision, or about Mamba2?

**Why.** The paper says something about images. A reviewer will ask what the
image has to do with it, and if the same collapse happens to a one-token text
fact then the finding is about Mamba2 memory in general and the framing must
broaden.

**Protocol.** Same block construction and same eight-way format, but the item to
be remembered is injected as text: a code word drawn from eight candidates,
stated once, then k filler tokens, then the forced-choice question. Probe the
recurrent state at the same distances.

**Stated in advance.** If the text fact survives materially longer than the
image gist, the compression is vision specific. If they decay together, the
claim generalises and the paper should say so plainly rather than keeping a
vision framing it has not earned.

**E8b, self-interference within the image.** There is a natural mechanistic
hypothesis here that no one has tested to our knowledge: an image writes to the
state 394 times, once per visual token, so most of the image's own trace may be
destroyed by the rest of the image before a single word of text arrives. Test it
by probing for a property of an early patch after 50, 100, 200 and all 394
visual tokens have been written, with no text filler at all. If decodability
falls across the image block itself, the picture is burying itself, and that is
both a memorable figure and a direct explanation for why gist survives while
particulars do not.

**Cost.** Two collections. E8b reuses the existing capture path entirely.

---

## E9. The consequence: visual KV pruning removes the only copy

**Why.** This is the section that answers "what do I do with it", and its
absence is one of the two reasons the current draft scores a 5.

A substantial literature prunes visual tokens or evicts visual KV entries on the
premise that they are largely redundant after the early layers. Our splice says
that in this hybrid the 13 attention layers hold the only copy of the visual
particulars, because the recurrent channel demonstrably does not carry them. If
that is right, the redundancy premise fails on hybrid architectures, and it
fails harder the fewer attention layers the model has.

**Protocol.**

1. Host: Zamba2-VL-7B. Locate the 394-position visual span inside the KV cache
   at each of the 13 attention layers.
2. Two eviction schedules. Uniform random retention at fraction ρ in
   {1.0, 0.75, 0.5, 0.25, 0.1, 0.05, 0}. Attention-ranked retention keeping the
   top ρ of visual positions by accumulated attention from the question tokens,
   which is the standard method family.
3. Measure accuracy across all three question families at each ρ.
4. **Matched comparison.** Run the identical schedules on Qwen2.5-VL-7B. This is
   the right control rather than an arbitrary dense model, because Zamba2-VL uses
   the Qwen2.5-VL vision encoder, so the vision front end and the visual token
   count are matched and only the language backbone differs.
5. Report accuracy against ρ for both, and the retention fraction at which each
   crosses a five-point degradation threshold.

**Prediction.** The hybrid degrades at a much higher retention fraction than the
dense model, because a dense transformer has the image distributed across all
its layers whereas the hybrid has concentrated it into thirteen.

**If the prediction fails.** Report it. A null here would mean the 13 attention
layers are substantially redundant with one another, which contradicts nothing
we have published and is worth knowing. It would cost us the payoff section, and
in that case the honest move is ICML in January with a different payoff rather
than a weak ICLR submission.

**Every named pruning method must be verified against its actual publication
before it enters the paper.** No citation goes in unchecked.

**Cost.** Two model loads, roughly fourteen sweeps. Cheap.

---

## E10. The splice, at scale and with a dose-response curve

**Why.** Two problems with the current splice. The host baseline of 17 out of 30
means the control tops out at 57%, which invites the question of what the other
43% were doing. And the result is binary, so it shows that attention matters
without showing how much of it matters.

**Protocol.**

1. n = 120 pairs, restricted to pairs where **both** images are answered
   correctly at baseline. The ceiling is then 100% by construction and the
   17-out-of-30 question disappears.
2. Full 2 × 2: {no swap, recurrent, attention, both} against outcome {host
   image, donor image, neither}. Wilson intervals throughout, and a McNemar test
   between the recurrent and attention conditions since the pairs are matched.
3. **Dose response.** Swap only k of the 13 attention layers, for k in
   {1, 3, 7, 13}, and separately compare swapping the early attention layers
   (6, 11, 17) against the late ones (65, 71, 77). This converts a binary result
   into a curve and answers a question we cannot currently answer: whether the
   image is held redundantly across all thirteen or concentrated in a few.

**Cost.** One collection, larger than before. The instrument already exists.

---

## E11. Depth profile

Free. We already store per-layer projected features for every trial, so
layerwise probe accuracy against layer index at each distance requires no new
compute. Likely one of the more interpretable figures in the paper: it shows
where in the stack the gist lives and whether it migrates with depth.

---

## E12. Probe capacity ladder

**Why.** Our negative results for count and position are the claim most
vulnerable to "your probe was underpowered", and there is a history here worth
disclosing on our own terms: count read at chance with 128 features and reached
19.4% at 512.

**Protocol.** Collect once at 4,096 projection features. Because the projection
columns are drawn independently, the 128, 512 and 2,048 conditions are simply
column prefixes of the same collection, so the entire ladder costs one run. Report
accuracy against probe width at distance zero for all three families, and at
distance 32 for identity, as an appendix table.

---

## E13. Second dataset

GQA balanced validation or VQAv2, recast into the same eight-way forced-choice
block structure so the analysis machinery is unchanged. Lowest return of
anything here, and the first item to cut if the calendar tightens. Its only job
is to show the cliff is not an artefact of our COCO construction.

---

## Dependency order

```
telemetry migration  ──┬──> E5 (mechanism)      ──> E6 (scale)
                       ├──> E11, E12 (free / cheap)
                       └──> instrument port ────> E7 (architecture + positive control)

independent of migration: E8 (modality), E9 (consequence), E10 (splice at scale)
```

E9 and E10 do not depend on the migration and can run in parallel with it. E5 is
first because it is cheap, it cannot fail, and its outcome decides whether the
paper's mechanism section says "decay" or says "interference", which changes the
framing of everything downstream.
