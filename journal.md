# Journal

This file records **decisions that could have gone the other way** and **things
that surprised me**. Nothing goes here that can be read off the code.

Per-step format:
- *Surprises* — what I expected, what actually happened
- *Decisions* — what I chose, which counterargument I rejected and why
- *Open* — what remains to be measured, and in which step

---

## 0.2 — Loading the soybean data
*2026-09-05*

### Surprises

**The class is the first field, not the last.** I expected the target at the
end, the way most tabular datasets do it. `.names` numbers only the 35
attributes, because the class is not an attribute — so the offset between the
numbering and the position in a row is +1.

**`hail: yes,no` means 0 = yes.** I read `hail = 0` as "no hail". It's the
opposite. Same for `lodging`. There is no default ordering — for every attribute
the order has to be read off `.names`.

**`?` is not a category, `dna` is.** In `.names`, `?` sits at the end of the
value list as if it were one more level. It isn't — it's the missing-value
marker. Counting it as a category would have shifted every code by one.
`dna` is a different thing entirely: it is a *recorded* answer meaning "this
question doesn't apply here". `?` = we didn't ask. `dna` = we asked, and it
doesn't apply.

**Everything turned into `float64` the moment NaNs appeared.** NumPy can't hold
NaN in an integer column, so pandas promoted the whole frame to float. Hence
`6.0` instead of `6`. There is a fix (`Int64`, capital I); left alone for now.

### The check that catches shifted column names

`leaves` is the only one of the 35 attributes with no `?` in its value list in
`.names` — so it is never missing, and must have **exactly 0 NaNs**.

This is the only one of the four checks that catches shifted column names.
`shape = (307, 36)`, 19 classes and 712 total NaNs would all match even if every
column name were off by one position. `leaves` would not.

Target numbers: `(307, 36)` / `19` / `712` / `0`.

---

## 0.3 — Attribute types
*2026-09-05*

Split into `ORDINAL` (9) and `NOMINAL` (26), plus the independent
`ENVIRONMENTAL` (10) and `BINARY` (16). The first two sum to 35, no overlap.

### Decisions

**`stem_cankers` → ORDINAL.**
Reason: `absent → below-soil → above-soil → above-sec-nde` describes how high
the canker sits on the stem, and height has an order.
Counterargument I rejected: the step from below-soil to above-soil is not
biologically the same size as the step from above-soil to above-second-node, so
equal spacing lies here too. I accepted that it lies **less** than throwing the
ordering away entirely would.

**`germination` → ORDINAL, but inverted.**
A higher code means *worse* germination: `90-100% → 80-89% → lt-80%`.
Remember this when reading the sign of a coefficient — a positive coefficient
here means "worse germination raises the odds", not the other way around.

**Four columns containing `dna` → NOMINAL for now.**
`leafspots_marg`, `leafspot_size`, `fruit_pods`, `fruit_spots`.

Reason: `dna` is not a level on the scale but a state **off** it. In
`leafspot_size` the ordering `lt-1/8 < gt-1/8` is perfectly sound, but `dna` is
not "even larger than gt-1/8". Treated as ordinal, the distance
`lt-1/8 → dna` comes out twice the distance `lt-1/8 → gt-1/8`. Translated:
"a plant with no spots is twice as far from a plant with small spots as a plant
with large spots is." Nonsense.

Going nominal throws away an ordering that genuinely exists among the remaining
levels, but it doesn't invent a false one. **Discarded information is
recoverable; invented information is not.**

**`seed_size` and `plant_stand` stay in ORDINAL** even though they are binary
and the ordering buys nothing. These lists are documentation of intent, not just
algorithm input — if they sat in `NOMINAL`, in six months I wouldn't know whether
that was because they have no ordering or because I dropped them as redundant.

### Observation

**16 of the 35 attributes are binary.** For a two-valued attribute the
ordinal/nominal distinction doesn't exist — every ordering of two elements is
the same ordering. There are **19 real ordering decisions, not 35**, and four of
them are contaminated by `dna`.

The practical consequence lands in 4.4 — the number of candidate splits for an
attribute with *k* levels:
- ordinal → *k−1* ("everything below the threshold goes left")
- nominal → *2^(k−1) − 1* (every subset)

For `fruit_spots` with 5 levels: 4 versus 15. This isn't only about speed — more
candidates means a higher chance of finding, by luck, a split that works
beautifully on 307 cases. Ordinality is a constraint, and a constraint is
protection against overfitting.

### Open

Three options for the four `dna` columns:
- **(a)** leave them nominal — the ordering is discarded
- **(b)** keep the ordering and accept `dna` sitting at the end of the scale —
  false spacing
- **(c)** split into two columns: "are there any at all" + "if so, how large" —
  `dna` becomes the answer to the first question, and the second column is empty
  where it doesn't apply

**Measured in 1.2** (Gower distance, KNN with all three variants).
**Decided in 4.5.**

Deciding without measuring is guessing — which is why this stays open.

---

## 0.4 — Where the holes sit
*2026-09-05*

The question was not how many holes there are, but **where they sit**. Answered
by reading the raw file, before writing any code.

### What the rows show

**`cyst-nematode` — the pattern is a constant.** All six rows have their holes
in exactly the same 24 columns. The missingness does not vary by case; it varies
by *disease*.

**`phytophthora-rot` — the pattern is bimodal.** 24 rows are sparse, 16 are
complete. So within a single disease there are two groups of cases, and whatever
separates them is not the disease.

**Between the two diseases the patterns differ.** Some columns are missing in
both, others in only one.

### `cyst-nematode`, column by column

Present (11):
```
date, crop_hist, area_damaged, plant_growth, leaves,
stem, fruit_pods, seed, mold_growth, seed_size, roots
```

Missing (24): everything else, including `leafspots_halo`, `leafspots_marg`,
`leafspot_size`, `canker_lesion`, `int_discolor`, `external_decay`, `mycelium`,
`fruiting_bodies`.

Sorted by what a person in the field had to do to record it:

| what was recorded | what it took |
|---|---|
| `date`, `crop_hist`, `area_damaged` | known before approaching the plant |
| `plant_growth`, `leaves`, `stem` | a look from a distance |
| `roots` | pull the plant up |
| `fruit_pods`, `seed`, `mold_growth`, `seed_size` | open a pod |

Everything missing is a **fine-grained observation**: the colour of a spot's
margin, whether a lesion is under or over one-eighth of an inch, the colour of
internal stem discolouration — which requires cutting the stem open.

### Interpretation

**It wasn't knowledge that was missing, it was protocol.**

Somebody pulled up a `cyst-nematode` plant, saw cysts on the root, wrote it down
and moved on. Why measure leaf spot size when the diagnosis was already settled
underground? For `phytophthora-rot`, 16 cases went through a thorough
examination and 24 through a quick one — probably different people, different
seasons, or a different recording form.

**The missingness pattern records how carefully someone looked, and how
carefully they looked depended on what they already suspected.** The diagnosis
influenced which data got recorded — not the other way around.

The statistical name is **MNAR**: missing not at random, and dependent on the
value that would have been measured. The worst of the three kinds, because
imputation cannot fix it. Imputation assumes the hole carries no information.
Here it carries a great deal.

### Consequence — this is leakage

A classifier that looks **only at which cells are empty**, at no values at all,
would score well on this dataset. It would know nothing about soybeans.

This is where the two projects collide. A farmer in the fitomedicina system will
answer three of eight questions — not because the disease is like that, but
because they can't be bothered to type. **There, the missingness pattern is
noise. Here, it is signal.** A model trained here would look for a pattern that
does not exist in the field.

### The recurring bug, and the lesson

My first attempt at listing the missing columns came out **shifted by exactly one
position** — it reported `leaves` as missing, which is impossible. The cause: the
mask was computed on a frame without `class`, then applied to a name list that
included it. 35 booleans, 36 names.

This is the same off-by-one the `leaves` check was introduced to catch in 0.2 —
and it came back, because that check ran once at load time and nothing after it
was guarded.

**A check that runs once does not protect the code written after it.** Whenever
columns get manipulated, `leaves` must still have zero holes. It costs one line
and belongs at the end of every such operation.

### Open

What to do about the leakage, resolved in **4.5**:

- **(a)** ignore it — the model uses the pattern, the paper result looks good,
  the field result does not
- **(b)** drop the 24 columns that are empty for whole classes — loses genuine
  symptoms because of one disease
- **(c)** keep everything, but measure twice: once as-is, once with the
  missingness pattern randomly shuffled across rows

**(c) is not a third opinion, it is a measurement.** The gap between the two runs
is how much of the score comes from leakage. A number, not a judgement — which
is why it waits until the tools exist.

---

## 0.5 — Linear algebra in NumPy
*2026-09-06*

**Cut, deliberately.** The step exists so that `A @ x` stops being magic. I had
already worked through NumPy and pandas, so hand-rolling dot products and matrix
multiplication was transcription, not learning. Written and then deleted — I
don't want scaffolding in the repository that no later step imports.

Counterargument I rejected: shape discipline is easier to acquire by writing the
loops than by debugging a `(307, 35)` @ `(34,)` later. Accepted the risk.

### What moved, rather than disappeared

**`solve` vs `inv` → 2.3.** The one part of 0.5 that NumPy fluency does not give
you; it is numerical analysis, not syntax. It belongs with the normal equations.

**0.6 synthetic generator → 2.1, 0.7 oracle → 6.1.** Same rule as `uv add`:
build the tool when a step needs it, so the history shows why it arrived.

### The one thing worth keeping from it

**Never compare floats with `==`.** Three ways of summing the same 1000 products
give three different results:

+= in a loop 247.75317023267948
math.fsum 247.7531702326793
np.dot 247.75317023267928

`math.fsum` tracks partial sums and returns the correctly rounded value, so a
plain Python loop can be *more* accurate than `np.dot`, which uses pairwise
summation. Neither is "the" answer — they are different summation algorithms.
Every comparison against sklearn from here on uses `isclose`, never `==`.

Related, same cause: `cos([1,1,1], [1,1,1])` comes out as `1.0000000000000002`
because `sqrt(3) * sqrt(3) != 3.0`, and `acos` raises on that. It does *not*
raise for `[1,2,3]` vs `[2,4,6]`, which returns exactly `1.0` — a failure that
appears on some inputs and not others. Clamp to `[-1, 1]` before `acos`.


## 1.2 — Distances
*2026-09-09*

Euclidean, Manhattan, Hamming, Gower in `phase1_evaluation/distances.py`.
Reference pair for every number below: rows 1 and 175, the only pair used
throughout that has no NaN at all.

hamming 17
manhattan 28
euclidean 7.3484692283495345 (squared: 54)
gower 0.45238095238095233


### The three numbers describe the same pair three ways

17 columns differ. Manhattan is 28, which is 11 above its floor — so some
columns jump more than one level. Euclidean squared is 54, well above the 17
it would be if every difference were exactly 1. Neither fact is visible from
Hamming, which only counts *whether* columns disagree.

**Manhattan can never fall below Hamming.** Every differing column contributes
1 to Hamming and at least 1 to Manhattan, because all values in S are integers.
Equality holds exactly when no difference exceeds 1.

**On a single differing column Euclidean and Manhattan are identical.** The
square and the root cancel when there is only one term. They separate only
once there are several columns to sum. Useful as a sanity check that a test
case is built the way it was intended.

### Why Euclidean and Manhattan are meaningless on nominal columns

Shown with numbers, not argued. Three plants identical except `canker_lesion`,
which has four unordered levels (`dna, brown, dk-brown-blk, tan`).

codes 1,2,3 (brown, dk-brown-blk, tan) manhattan 1, 2, 1
codes 2,3,1 (levels reordered) manhattan 2, 1, 1
hamming, both schemes 1, 1, 1


Nothing about the plants changed — only the order the level names were typed
into `.names`. **The most-distant pair moved.** Under the first scheme it is
`brown` vs `tan`, under the second `dk-brown-blk` vs `tan`. Any claim about
which two lesions are most unlike comes from the file, not from the soybean.

Hamming is unchanged and would be unchanged under every permutation, because
"same or not" is the only question codes of a nominal column can answer
honestly.

*Predicted before running: the most-distant pair would change. Correct.*

### What Gower fixes

Two problems, both of which the other three share.

**Column length decides influence.** `date` has 7 levels so differences there
reach 6; `severity` has 3 so they reach 2. Manhattan therefore gives `date`
three times the vote — not because months matter more than severity, but
because someone chose to record 7 buckets rather than 3. Had `date` been
logged weekly, its vote would have grown fivefold and nothing about the
plants would be new.

Gower scores each column on its own range first, so both a `date` pair at the
extremes (0 vs 6) and a `severity` pair at the extremes (0 vs 2) score 1.0.
Equal because both are as far apart as their column can express.

**Units are added that are not comparable.** One Manhattan step in `date` is a
month; one in `severity` is minor → pot-severe. Manhattan sums them as if both
were metres.

Range is a property of the column, computed over all 307 rows — never over the
pair being compared, which would divide by zero whenever two plants agree.
**In 1.7 the range must be computed on the training split only**, or test
information leaks into the distance.

**Upper bound, no reference implementation needed:** Gower ≤ Hamming / n, with
equality exactly when every differing column is nominal. 0.4524 against a
bound of 17/35 = 0.4857 says most of the 17 differing columns are nominal.

### Where Gower stops being comparable — and the bound that was wrong

Rows 1 and 301 (`cyst-nematode`, 24 holes in the same 24 columns as its five
siblings):

n (columns actually compared) 11
differing 6
gower 0.5454545454545454 = 6/11 exactly

Predicted: above 0.4524. Correct. Also predicted, by me: strictly below the
6/11 bound, since ordinal columns contribute a fraction rather than a whole.
**Wrong.** It landed exactly on the bound.

Checked rather than assumed: the 11 compared columns are 8 nominal
(`area_damaged, plant_growth, leaves, stem, fruit_pods, seed, mold_growth,
roots`) and 3 ordinal (`date, crop_hist, seed_size`), and all three ordinal
ones hold identical values in the two rows. So all 6 disagreements fell in
nominal columns and each contributed a full 1.

The lesson is not about Gower. **A bound that relies on how columns "usually"
distribute is a guess wearing a formula.** The `else: continue` branch never
fired and `ORDINAL` + `NOMINAL` covers all 35 — that was verified, not
assumed, and it is what ruled out a bug as the explanation.

### The open problem this creates

0.4524 was computed over 35 columns, 0.5455 over 11. Both are numbers between
0 and 1 and they look comparable. They are not.

Dividing by 11 is not an admission of ignorance about the other 24 columns —
it is the **assumption that those 24 would have behaved like these 11**. The
number reads as "these plants differ in 54.5% of what is known about them" and
then gets used as "54.5% of what they are".

And the error has a direction. On 11 columns a single disagreement is worth
1/11 of the total; on 35 it is worth 1/35. Distances measured on few columns
swing to both extremes, while those measured on all 35 cluster near the middle.

For KNN in 1.3: sparse rows will look suspiciously close to some neighbours
and suspiciously far from others, for reasons that have nothing to do with the
plant. And from 0.4 the sparsity is not random — it tracks the diagnosis. So
the instability lands squarely on `cyst-nematode` and on 24 of the 40
`phytophthora-rot` rows.

This is the MNAR finding from 0.4, now as a number instead of an observation.
**Measured in 4.5**, not decided here.

### Decisions

**NaN handling differs between Hamming and Gower, deliberately.**
Hamming with a mask divides by a constant; the pair 1–301 gave distance 1 out
of 5 compared columns on a first attempt, which flatters the similarity. Gower
drops a column from numerator *and* denominator — the original coefficient's
behaviour, taken from the definition rather than invented here.

Without a mask, Hamming counts two holes in the same column as a *difference*:
`NaN != NaN` is `True`, since NaN equals nothing including itself. Rows 1 and
301 score 30 unmasked against 6 masked, and 24 of that 30 is pure absence of
data. **Absence of measurement must never register as disagreement.**

**The `mask.sum()` check caught a wrong row index.** A first run used row 302
and gave `mask.sum() = 5`; 0.4 recorded that `cyst-nematode` has 11 columns
present, so the mismatch surfaced immediately. The bug was not spotted by
reading the code — it was spotted because a target number existed. Same
mechanism as the `leaves` check in 0.2. **Write the expected number down
before running, or the run cannot contradict you.**

### Open

**The four `dna` columns.** Still nominal. The `canker_lesion` experiment
above is the general argument for why permuting nominal codes changes
Euclidean and Manhattan but not Hamming or Gower — it does not settle whether
splitting those four into "any at all?" + "if so, how large?" beats leaving
them nominal. **Decided in 4.5.**

**Performance, deferred not forgotten.** `gower` recomputes
`data_frame[col].max()` inside the loop, scanning all 307 rows per column per
pair. Fine for single pairs. In 1.3 an all-pairs matrix is ~94k calls, and
that is where ranges get hoisted out and the function gets vectorised.