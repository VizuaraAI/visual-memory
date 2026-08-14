# Taking "What Is Left of a Picture" to NeurIPS / ICML / ICLR

Status of record: the current draft self-reviews at **5/10, confidence 4/5** under
the a-star-reviewer rubric. That is the standard landing spot for a clean result
on one model with no consequence. This document is the plan to move it to 7 or 8.

---

## 1. The three objections that decide the outcome

Every reviewer of an analysis paper runs the same four attacks: confound,
generality, direction of inference, consequence. We already pass two.

| Attack | Current standing | Verdict |
|---|---|---|
| Confound | Pre-image control at exactly 12.5%, permutation at chance, capacity control at 71.0% vs 50%, letterbox fix documented | **Passing.** Controls are unusually clean. |
| Direction of inference | The splice is causal, not correlational: swap the channel, the answer follows | **Passing.** This is the paper's strongest asset. |
| Generality | n = 1 model, 1 family, 1 dataset, 1 scale | **Fatal.** "Case study" is the word that ends it. |
| Consequence | None stated. The paper describes and stops | **Fatal.** "Interesting, but what do I do with it?" |

Two secondary weaknesses: the splice ceiling of 17/30 on the host baseline, and
a probe-capacity history (count read at chance with 128 features, 19.4% at 512)
that a careful reviewer will find and ask about before we volunteer it.

So the plan has exactly two jobs: **kill generality, and manufacture a
consequence.** Everything else is polish.

---

## 2. The claim we should be making instead

Current claim, one sentence: *in Zamba2-VL-7B, the recurrent state holds gist for
about 32 tokens and the attention cache does the real work.*

That is a measurement. Here is the claim that gets championed in a discussion
period:

> **In hybrid linear-attention VLMs, the recurrent state is not a memory of the
> image at all. It is a short-lived summary whose lifetime is set by the model's
> own decay parameters, and the presence of even a few attention layers is what
> allows it to be short-lived. Visual particulars live entirely in a thin
> attention channel, which means the compression methods that treat visual KV as
> redundant are removing the only copy.**

That version has a mechanism (decay parameters), a scope (hybrid VLMs, plural),
a contrast that could falsify it (a pure-SSM VLM), and a casualty (KV and visual
token pruning). Each clause maps to one experiment below.

---

## 3. The experiments, ranked by score gained per day spent

### Tier 1: without these there is no A* submission

**A. Predict the cliff from the weights.** *2 days, near-zero compute.*

Mamba2 decays as `S_t = S_{t-1} exp(Δ_t A_h) + ...` with a scalar `A_h < 0` per
head. Over k tokens the retention factor is `exp(A_h Σ Δ_t)`, so the per-head
half-life in tokens is `ln 2 / (|A_h| Δ̄_h)`. `A_h` is a stored weight. `Δ̄_h` we
already capture, because `capture_projections` hooks `in_proj` and the dt slice
is the last 112 channels. So we can compute a per-head, per-layer predicted
half-life distribution from the model itself and lay it against the measured
32-token collapse.

This is the single highest-value item in the plan. It converts phenomenology
into prediction, and it is honest under either outcome:

- If the median predicted half-life lands near 32 tokens, the paper now explains
  its own headline result from first principles.
- If it lands at hundreds of tokens, the collapse is **not decay**, it is
  interference from subsequent writes. That is a stronger and more surprising
  result: the state is not forgetting, it is being overwritten. We would test it
  by feeding k tokens of padding that carry no information (repeated whitespace,
  or a fixed filler) and checking whether the readout survives longer than under
  real text. If it does, interference is confirmed.

Either branch is publishable. There is no failure mode where we learn nothing.

**B. A pure-SSM VLM with no attention at all.** *4 days.*

This is the control that answers the question a reviewer will ask in the first
paragraph: is this about Mamba2, or about the hybrid? If a VLM with zero
attention layers also dumps the image from its recurrent state within 32 tokens,
it would be blind, and it is not, so it must keep the image longer. That
contrast is the whole architectural claim.

Candidates to evaluate, in order of preference:
1. **Cobra** (Mamba backbone + vision encoder). Pure SSM, no attention layers.
2. **VL-Mamba.** Same category.
3. **MaTVLM** (`hustvl/MaTVLM_0_25_Mamba2`), a distilled TinyLLaVA with Mamba2
   layers, already confirmed to exist. Different family and different training
   recipe from Zamba2, which is what the n=1 objection actually asks for.

Availability and exact recurrence form must be verified before committing;
Mamba1 models need a different read operator than Mamba2 (`y = S C` still holds
but B and C are not group-shared the same way). Budget one day for that check.

**C. Scale replication inside the family.** *2 days, mostly mechanical.*

`Zyphra/Zamba2-VL-2.7B` and `Zamba2-VL-1.2B`, both confirmed to exist. Same
instruments, same materials, same analysis. If the cliff sits near 32 tokens at
all three scales the result reads as a property of the architecture; if it moves
with scale that is itself a finding worth a figure.

**Blocker:** `lab/vtelemetry.py` is half-migrated. The `Geometry` dataclass and
`geometry(model)` are in place, but `decoder_layers` still asserts 81 layers,
`split_projection` asserts 81 captures, and `read_state`, `project_state`,
`snapshot_recurrent`, `restore_recurrent` and `channel_swapped` all loop over the
module constant. That migration is about an hour of work and it gates A, B and C.
It is the first thing to do.

### Tier 2: this is where the consequence comes from

**D. The casualty: visual KV pruning on a hybrid.** *3 days.*

A substantial literature prunes visual tokens or evicts visual KV entries on the
premise that they are highly redundant after the early layers. Our splice says
that in a hybrid, the 13 attention layers hold the *only* copy of the visual
particulars, because the recurrent channel demonstrably does not carry them.

The experiment: apply an eviction schedule to the visual KV entries at the 13
attention layers, sweeping retention from 100% down to 0%, and measure VQA
accuracy. Then run the identical schedule on a comparable dense transformer VLM
of similar size. The prediction is a much steeper collapse on the hybrid, because
the dense model has visual information distributed across all its layers whereas
the hybrid has concentrated it into thirteen.

This is the payoff section. It turns "here is what we observed" into "here is a
method class that silently breaks on this architecture, and here is the
measurement." It also gives the paper a number that a practitioner can act on:
the retention fraction below which a hybrid VLM stops seeing.

Before any of this reaches the paper, every pruning method we name must be
verified against its actual publication. No citation goes in unchecked.

**E. Is this about vision, or about Mamba2?** *2 days.*

Same probe, same distances, but the item to be remembered is a text fact rather
than an image ("the code word is CRIMSON", asked back after k tokens). If text
decays on the same 32-token schedule, our claim is about Mamba2 memory in
general and the paper should say so. If the image decays faster, that is a
vision-specific claim and it needs the comparison to stand.

This is cheap, and it is the control a reviewer names when they ask what the
image has to do with it.

### Tier 3: rigor patches, each one closing a review question

**F. Splice at n = 120 with a proper 2 by 2.** The current 17/30 host baseline is
a ceiling that invites the question "why is your control only 57%?" Answer it by
scaling to 120 pairs, reporting host and donor rates with confidence intervals,
and restricting to pairs where both images are answered correctly at baseline so
the ceiling is 100% by construction.

**G. Depth profile.** Which of the 81 layers carry the gist, and does the
decodable content move with depth? We already store per-layer projected features,
so this is analysis on data we have. One figure, possibly the most interpretable
in the paper.

**H. Second dataset.** GQA or VQAv2 alongside COCO, to show the cliff is not an
artefact of our eight-way COCO construction.

**I. Probe-capacity ladder.** 128, 512, 2048, 4096 projection dimensions, reported
as an appendix table. This pre-empts "your negative results for count and position
might just be an underpowered probe," and it puts the 128-dimension history on the
record on our terms rather than a reviewer's.

---

## 4. Schedule and venue

Verified deadline: **ICLR 2027, abstract 18 September 2026, full paper 25
September 2026, decisions 16 December 2026.** That is six weeks from today
(11 August 2026).

ICML 2027 and NeurIPS 2027 dates are not yet published. Based on the pattern of
recent years, expect ICML around late January 2027 and NeurIPS around May 2027.
Both should be treated as expected, not verified, until the calls appear.

| Window | Work |
|---|---|
| Week 1 (11 to 17 Aug) | Telemetry migration. Experiment A, both branches. Verify pure-SSM model availability and recurrence form. |
| Week 2 (18 to 24 Aug) | Experiment C at two scales. Start B. Experiment G from existing data. |
| Week 3 (25 to 31 Aug) | Finish B. Experiment E. |
| Week 4 (1 to 7 Sep) | Experiment D, both architectures. Experiment F. |
| **8 Sep: go / no-go** | Submit to ICLR only if A, B and D have all landed. |
| Week 5 (8 to 14 Sep) | Experiment H and I. Rewrite around the new claim. New figures. |
| Week 6 (15 to 25 Sep) | Two full adversarial review passes with the a-star-reviewer skill, one at the start and one 48 hours before the deadline. Submit. |

**If the gate fails on 8 September**, the correct move is not to submit a thin
version. It is to take ICML 2027 in January, which gives four extra months, and
to put the current version into a NeurIPS 2026 workshop in December for feedback.
Most relevant workshops are non-archival, so this does not burn either venue.
Workshop deadlines land around September and October and must be checked when
the NeurIPS 2026 workshop list is announced.

Recommendation: **aim at ICLR, but plan for ICML.** Six weeks is enough for A, C,
E, F, G and I with confidence. B and D are the two that could slip, and they are
also the two that matter most, which is precisely why the gate exists.

---

## 5. Cost

Everything so far has cost under $30. Adding four models across three experiment
families, at the observed per-model cost, lands somewhere in the $250 to $450
range on Modal H100 time. Experiment A is essentially free because it reads
weights and reuses captures we already have. Compute is not the constraint here;
the calendar is.

---

## 6. What the paper looks like when this is done

Retitled around the mechanism rather than the observation. Five claims, each with
its own figure:

1. The recurrent state records gist and not particulars. *(have)*
2. Its lifetime is set by the model's decay parameters, and we predict the
   observed cliff from the weights. *(A)*
3. This holds across three scales and two model families, and it does **not**
   hold for a pure-SSM VLM, which must keep the image longer because it has
   nowhere else to put it. *(B, C)*
4. Causally, the answer follows the attention channel. *(have, strengthened by F)*
5. Therefore visual KV pruning, which assumes redundancy, removes the only copy
   on hybrid architectures, and we measure where it breaks. *(D)*

That paper is not a case study. Claims 2 and 5 are the difference between a 5 and
an 8.

---

## 7. First action

Finish the `vtelemetry.py` migration, because A, B and C all sit behind it, and
then run experiment A the same day since it needs no new GPU time.
