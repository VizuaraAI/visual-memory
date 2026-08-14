# The six-week push to ICLR 2027

Deadline: **abstract 18 September 2026, full paper 25 September 2026**, decisions
16 December 2026. That is the only top venue whose decision lands this calendar
year, so it is the target. Six weeks from 12 August.

Current self-review: **6/10, confidence 4/5.** Up from 5/10 for version 1. The
job is to reach a defensible 7.

---

## The limitations, ranked by how much they cost us

### 1. Eviction changes more than the image's presence (FATAL if unaddressed)

Removing the visual key-value entries shortens every attention cache by about
400 positions and changes the softmax normalisation for everything downstream.
Our central result is a 43.7 point drop under that intervention. A reviewer will
ask, correctly, how much of that is the missing image and how much is the
perturbation.

**Fix: the matched non-visual eviction control.** Evict an equal number of
non-visual positions from the same caches and show fast-head decodability is
unaffected. Same machinery as E9, a different index set.

Cost: one day. This is the highest value per hour in the entire plan, because it
converts the single sharpest objection into a control table row.

### 2. The relay result could be a sensitivity asymmetry (SERIOUS)

Freshly re-encoded content plausibly sits at higher amplitude than content
written many tokens ago and decayed. A probe that recovers high-amplitude
content well and low-amplitude content poorly would produce everything we
observe without the state having stopped storing anything.

**Fix A, cheap: calibrate the probe against amplitude.** Scale the state by a
known factor before projecting and measure how decodability falls with
amplitude. If the probe is roughly amplitude-invariant over the relevant range,
the alternative account loses its footing. Half a day.

**Fix B, decisive but expensive: a model with no attention to relay from.** In a
pure state-space VLM, probing the recurrent state must be measuring retention
because there is nothing else it could measure. Cobra is the candidate. It is
Mamba1, so `read_state`, `project_state` and the splice need porting, and the
E0-style gate must be re-run before any number is trusted. Four days with real
risk of not landing.

### 3. One model family (SERIOUS, and the reason version 1 scored 5)

Three checkpoints, all Zamba2-VL. One vendor, one recipe.

**Fix, ranked by risk:**

- **A second hybrid Mamba2 family, text task.** Falcon-H1, Nemotron-H, Bamba and
  Granite 4.0 are Mamba2 hybrids available through transformers, so the existing
  instruments apply with no port. Run the relay experiment on a text memory task
  (a code word stated once, asked back after k tokens). This buys a second
  family \emph{and} a second modality in one experiment, and it tests whether
  "relays rather than stores" is architectural or vision-specific. Three days.
  This is the best generality purchase available.
- **MaTVLM** (`hustvl/MaTVLM_0_25_Mamba2`), a distilled TinyLLaVA with Mamba2
  layers. Different family, different recipe, still vision. Needs their repo
  rather than transformers. Three days, moderate risk.
- **Cobra**, as above, doubles as the fix for limitation 2.

### 4. The consequence is thin where it matters (SERIOUS)

At 25% retention there is no measurable cost at either distance tested. The
34-versus-59 point contrast is at **zero** retention, which no deployed method
uses. As written, claim 5 will be read as a warning about a regime nobody is in.

**Fix: find where the interaction actually bites.** Sweep retention
$\{1.0, 0.25, 0.1, 0.05\}$ against distance $\{0, 32, 128, 512, 1024\}$. Either
a realistic retention level degrades at long distance, which gives a genuine
practical claim, or it does not, in which case claim 5 is honestly reframed as a
scientific boundary on the redundancy premise and the abstract stops implying
practical danger. One day of compute, and either outcome is publishable.

### 5. The stratum dose-response rests on three points (FIXABLE, cheap)

Eviction sensitivity is ordered exactly by decay rate and correlates at
$r = 0.991$, but across three strata that is a suggestion, not a correlation.

**Fix: re-run E5c with ten strata instead of three.** Same forward passes, more
projections per pass. Ten points would make this a real quantitative link
between the weights and the mechanism, and it is currently the most elegant
result in the paper. One day.

### 6. One projection seed (FIXABLE, cheap)

Every probe number uses a single Gaussian basis. Repeat the headline analyses
across three seeds and report the spread. Half a day, closes a standard
objection before it is raised.

### 7. The route is not traced (FIXABLE, cheap, high value)

We show information arrives from attention, not how.

**Fix: layerwise eviction.** Remove the visual entries from only the early
attention layers, then only the late ones, and see which kills fast-head
decodability at distance. The machinery exists in E10's attention subsets and
E9's eviction. One day, and it upgrades "arrives from attention" to "arrives via
these layers".

### 8. One task format, one dataset (LOWER PRIORITY)

Eight-way forced choice on COCO. A second dataset would help and is the first
thing to cut if the calendar tightens.

---

## What to add that is not a fix

**A positive methodological prescription.** The paper currently warns that
probing a recurrent state at distance measures the attention channel. It should
also say what to do instead: report probe results under matched channel
ablation, so that retained and relayed content are separated by construction.
That turns a negative result into a protocol other people can adopt, and
protocols get cited. Half a day of writing, no compute.

This matters more than it sounds. Analysis papers that end with a warning score
worse than ones that end with something the reader can use.

---

## Schedule

| Window | Work |
|---|---|
| Week 1 (12 to 18 Aug) | Non-visual eviction control. Layerwise eviction route. Ten-strata re-run. Seed robustness. |
| Week 2 (19 to 25 Aug) | Retention by distance sweep to settle claim 5. Amplitude calibration of the probe. |
| Week 3 to 4 (26 Aug to 8 Sep) | Second hybrid family on the text task. This is the long pole. |
| **8 September: go / no-go** | Submit only if the non-visual control and the second family have both landed. |
| Week 5 (9 to 15 Sep) | Rewrite around whatever the second family showed. New figures. Methodological protocol section. |
| Week 6 (16 to 25 Sep) | Two adversarial review passes, one at the start and one 48 hours out. Submit. |

If the gate fails on 8 September, the correct move is ICML 2027 in January
rather than a thin ICLR submission, with the current version going to a NeurIPS
2026 workshop in December for feedback. Most such workshops are non-archival and
burn nothing.

## Honest expectation

Fixes 1, 5, 6 and 7 are cheap and near-certain, and together they take the paper
from 6 to a solid 6, maybe 7. The two that decide whether it clears the bar are
the second model family and the consequence sweep. Neither is guaranteed. I
would put the odds of a genuine 7 by 25 September at roughly even, and the odds
of acceptance conditional on submitting a 7 at somewhere near half again, since
ICLR acceptance runs near a third overall and analysis papers with one family
sit below that.

The honest summary is that this is a real shot rather than a likely one, and the
single largest determinant is whether a second architecture family reproduces
the relay finding.
