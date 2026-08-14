# Paper, version 2

The current draft is seven pages built around one measurement on one model. The
target is a nine-page ICLR or ICML paper built around five claims, of which
three are new, with an appendix carrying the controls.

---

## 1. What changes at the top

**Title.** The current title, "What Is Left of a Picture", is memorable but
promises a description. The next version needs the mechanism and the scope in
the title. Candidates, to be chosen once E5 has decided between decay and
interference:

- *A Picture Lasts Thirty Tokens: Where Visual Information Lives in Hybrid
  Linear-Attention VLMs*
- *The Recurrent State Is Not Where the Picture Is*
- *Buried by Its Own Patches: Visual Memory in Hybrid Linear-Attention VLMs*
  (only if E5b confirms interference)

**Abstract, rewritten around the payoff.** The current abstract ends on the
splice, which is the middle of the story. The new one ends on the consequence:
that methods which prune visual KV on a redundancy premise are removing the only
copy on hybrid architectures, and here is the retention fraction at which they
break. A reviewer who reads only the abstract should be able to say what the
paper is for.

**One-sentence claim, the version a reviewer repeats to a colleague.** In hybrid
linear-attention VLMs the recurrent state is not a memory of the image; it is a
short-lived summary whose lifetime is set by the model's own parameters, and the
visual particulars live entirely in a thin attention channel.

---

## 2. Section plan

| # | Section | Status | Backed by |
|---|---|---|---|
| 1 | Introduction, with the five claims stated as a numbered list | rewrite | |
| 2 | What was known before, and what is new here | expand | add the KV and visual-token pruning literature, all verified |
| 3 | Significance | expand | now has a concrete casualty to point at |
| 4 | Model, instruments, materials | keep, extend | add the second-family and pure-SSM instrument ports |
| 5 | Claim 1: the state records gist, not particulars | keep | existing results |
| 6 | Claim 2: the lifetime is predicted by the model's own parameters | **new** | E5, E5b |
| 7 | Claim 3: it generalises across scale and family, and inverts without attention | **new** | E6, E7 |
| 8 | Claim 4: causally, the answer follows the attention channel | strengthen | E10 dose-response |
| 9 | Claim 5: therefore visual KV pruning removes the only copy | **new** | E9 |
| 10 | Limitations | expand | probe capacity history, splice ceiling, model count |
| 11 | Conclusion | rewrite | |
| A | Controls: pre-image, permutation, capacity, swap efficacy | keep | existing |
| B | Probe capacity ladder | **new** | E12 |
| C | Depth profile | **new** | E11 |
| D | Second dataset | **new, optional** | E13 |

Sections 2 and 3 stay as explicitly named sections. That is a standing
requirement on this project and it also happens to be good practice: reviewers
who cannot find the novelty statement invent one, usually less generous than
ours.

---

## 3. Figure plan

Eight figures for nine pages. Schematics go through PaperBanana, with briefs
that state the exact outcome verbatim, because the failure mode we have already
hit once is a beautiful figure asserting the opposite of our result.

| Fig | Content | Source | Status |
|---|---|---|---|
| 1 | The idea: two memory channels, one question | PaperBanana | regenerate for the new claim |
| 2 | Architecture: 81 recurrent layers, 13 with attention, where each channel lives | PaperBanana | have, re-inspect at full size |
| 3 | Gist against particulars: 83.1 / 19.4 / 17.5 against 12.5 chance | plot | have, overlap defect already fixed |
| 4 | The cliff, with predicted retention from the weights overlaid | plot | **new, E5** |
| 5 | Head-stratified decay: slow heads against fast heads | plot | **new, E5** |
| 6 | Decoupling gap across four models, including the pure-SSM inversion | plot | **new, E6 + E7** |
| 7 | Splice dose response: answer source against number of attention layers swapped | plot | **new, E10** |
| 8 | KV eviction: hybrid against dense with matched vision encoder | plot | **new, E9** |

Figure 6 is the one that kills the generality objection and figure 8 is the one
that supplies the consequence. If only two new figures land, those are the two.

Every figure must survive the check we adopted after the last round: render at
full size, read it, and confirm no text collides with other text. That check
runs on all eight, not only the new ones.

---

## 4. Prior work, to be assembled and verified

The section needs three groups, and not one citation enters without being
checked against the actual publication.

1. **Hybrid and linear-attention architectures.** What the design claims about
   memory, and what has actually been measured about it.
2. **Probing and causal intervention in interpretability.** Where our
   probe-plus-splice design sits, and specifically the standard that
   interventions must meet.
3. **Visual token and KV compression in VLMs.** This group is new and it is the
   one that gives claim 5 its target. We need the methods that explicitly rest
   on a redundancy premise for visual tokens, stated in their own words, so that
   the paper contrasts our measurement against what those methods assume rather
   than against a straw version of them.

The tone on group 3 matters. We are not saying those methods are wrong; they
were developed and validated on dense transformers, where they work. We are
saying the premise they rest on does not transfer to hybrids, and we measure
where it stops holding. Written that way it is a contribution. Written as a
takedown it invites a hostile reviewer from exactly that community.

---

## 5. Limitations, written before a reviewer writes them for us

Four items, each stated plainly rather than buried:

1. **Model count.** Four models after this work, not forty. The claim is about
   the architectures tested and the paper says so.
2. **Probe capacity.** Count read at chance with 128 features and reached 19.4%
   at 512. The ladder in appendix B shows where the negative results are stable
   and where they are not. Disclosing this ourselves is worth more than the
   appearance of never having had a weaker result.
3. **Splice ceiling.** The version-one splice had a host baseline of 17 out of
   30. E10 removes the ceiling by construction, and the paper reports both.
4. **Forced choice.** Eight-way multiple choice is not open-ended VQA. It buys
   an exact chance rate and a clean pre-image control, and it costs generality
   to free-form answering.

---

## 6. Writing standard for this draft

Carried over and non-negotiable on this project:

- No em dashes and no en dashes anywhere. Use commas, colons, parentheses, or
  the words "and" and "to".
- No hallucinated references. Every citation verified against the publication.
- Named sections for novelty and for significance.
- Do not compress. The previous draft was criticised for too few words. Results
  and controls in particular get full paragraphs of prose, not telegraphed
  sentences beside a figure.
- Do not quote the fitted 469-token half-life. Fitting an exponential to a cliff
  produces a meaningless number. Report the measured step. Note that E5 changes
  the situation: a half-life computed from the model's parameters is a different
  object from one fitted to the decay curve, and that one we do quote.

---

## 7. Review discipline

Two full adversarial passes with the `a-star-reviewer` skill: one when the first
complete draft exists, one 48 hours before submission. The rubric row that
applies is Analysis and interpretability, where the killers are "interesting,
but what do I do with it", n = 1, and correlational evidence for a causal claim.
After this plan, we have an answer to all three, and the second pass exists to
check that the answers actually landed in the text rather than only in our
intentions.

Target after the second pass is a 7. If it reads as a 6 we submit anyway. If it
reads as a 5 the work did not land and ICML in January is the correct venue.
