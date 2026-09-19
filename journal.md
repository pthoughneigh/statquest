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


## 1.3 — KNN
*2026-09-13*

### `make_gower` — a closure, and a test that cannot fail

`gower` took three arguments where the other three distances take two, so it
could not be passed to `knn` as a `Callable[[Vector, Vector], float]`.
Loosening the annotation was rejected: it hides the mismatch until runtime and
forces `knn` to know there are two kinds of distance function. Instead the
frame is captured and only two arguments remain visible from outside.

The same move clears the performance item left open in 1.2. Ordinal ranges are
computed once, for the nine `ORDINAL` columns only — the other 26 never need a
range, because a nominal column contributes 0 or 1 regardless of how many
levels it has. Previously `max` and `min` rescanned all 307 rows per ordinal
column per call; the all-pairs matrix 1.3 needs is 307×306/2 ≈ 47,000 pairs.

The inner function still needs one thing from the frame: `columns`. Values are
read positionally as `a[i]`, `b[i]` while the column type is looked up by name,
so something has to bridge the two — bare arrays carry no names. Capturing
`columns` into a local rather than holding the frame freezes it in the same
instant as the ranges. The frame is captured by reference, the ranges by value;
mutate the frame afterwards and the names shift while the ranges do not, and
the function starts lying quietly.

**Prediction, before running: bit-identical, not merely close.** Correct. The
same values in the same operations in the same order — only *when* `max` and
`min` are evaluated changed, not *what* they return. This is one of the few
places `==` is defensible; had the last digits been allowed to move, this would
have needed `isclose`, which would have been an admission that the refactor was
not clean. The one new line that is not purely a change of timing is the
`float()` cast around the range. It moves nothing on this data — pandas already
returns `np.float64` after `na_values="?"` — but it is the only candidate to
shift something later, on an integer column with no NaN.

### The test that cannot fail

Asked which of the two reference pairs could not detect a broken
`ordinal_ranges`, the answer came back as 1–301 — pair right, reason wrong.
NaN was the first explanation, but NaN only explains why 11 of 35 columns are
compared. Three ordinal columns cleared the NaN filter, reached
`ordinal_ranges[col]`, and executed the division.

They contributed nothing because `date`, `crop_hist` and `seed_size` hold
identical values in rows 1 and 301, recorded in 1.2. `abs(x - y)` is zero three
times, and zero divided by anything is zero. `rng` could be 6, 1 or 1000 and
`score` would not move; `n` still reaches 11 and the result is still 6/11.

**A test whose correct value does not depend on the thing it measures is not a
test.** Third instance of the same mechanism, and the first where it fails:
`leaves` in 0.2 worked because it is the one column that can never be NaN;
`mask.sum()` in 1.2 worked because 0.4 had already fixed the number at 11.
`6/11` passes identically with correct and with corrupted ranges. Pair 1–175 is
the only real check — it disagrees in ordinal columns too, so the ranges enter
the sum. Both pairs belong in `__main__`: one covers 11 columns of 35.

### The model itself

Four steps, of which only the last is a decision: distances to every row,
`argsort`, first `k`, vote. `argsort` returns **positions**, and a position in
`distances` is a position in `classes` because both are ordered as the frame's
rows — so the nearest classes come from indexing `classes`, never from pulling
symptom rows out of the frame. Symptoms are not needed once the distances
exist.

`k=1` was considered and rejected in one sentence: taking the first neighbour
measures the other four and throws them away. And a model whose every answer is
100% is a model that cannot say "not sure" — which is the probability output 1.6
needs, and which he had already volunteered as 60/40 rather than just a winner.

`np.asarray(frame, dtype=float)` turned out to be a free check. If `class` is
still in the frame it raises `ValueError` on the first string, in the first
line — rather than the silent shifted number that the earlier three-argument
`gower` would have returned. The conversion that was there for the distance
function is also the guard `knn` could not otherwise perform.

### Where `class` gets dropped, and why one place

Four candidates were worked through before settling. The deciding fact is that
`make_gower` **freezes** `columns` and `ordinal_ranges` at the moment it is
called. Dropping `class` inside `knn` afterwards cannot reach a closure built
over 36 columns: rows arrive with 35 values, the loop counts to 36, stops at 35
— and `IndexError` never fires. This is the 0.5 finding, in a second place and
in the direction that does not protect you. The prediction in 0.5 was that a
length mismatch would raise both ways; it does not.

Nor can `knn` detect the mismatch. It can read `shape[1]` and get 35, but the
number of columns a closure expects is invisible from outside, and `hamming`
has no notion of columns at all. So `class` is dropped **above** both
`make_gower` and `knn`, in the caller — there is only one frame, so there is
nothing to disagree. That adds a fifth parameter, `classes`, placed next to
`data_frame` because `classes[i]` only means anything if it is ordered as row
`i`.

A fourth option — `knn` computing the ordinal ranges itself — was rejected on a
different ground. `knn`'s four jobs are distance, sort, select, vote, and not
one of them knows what an ordinal column is. Asked who should know, the answer
came back "the distance", which is exactly what `make_gower` exists for.

### Self-exclusion, and two regimes that are not a flag

The plant being predicted must not vote for itself — that vote carries the
answer the model is supposed to derive. At `k=5` it is 20% of the vote handed
over free; at `k=1` the model is perfect and has learned nothing. First
appearance of the train/test distinction, named in 1.7.

But the zero must not be excluded by **value**. Duplicate rows genuinely exist
— 35 integer-coded columns, many binary — and a distance of zero to *another*
plant is the strongest evidence available, not an artifact. Excluding zeros
throws the best neighbour away. "Drop the zero" and "drop yourself" are not the
same rule; one is a measurement, the other is leakage.

So exclusion goes **outside** `knn`, by position, and `knn` takes no regime
argument. Prediction on a new plant passes the whole frame and nothing is
dropped; evaluation over the 307 rows passes the frame without row `i`. The
position is never searched for — either the row was taken out and its position
is known, or it came from outside and is not in there. Two callers, one
function.

His question — how to exclude by position a plant that has no position — was
the right question and dissolves the same way: the case where there is no
position is the case where nothing is excluded.

### Two silent bugs, both label-versus-position

**`drop(index=<row>)` instead of `drop(index=<label>)`.** A whole `pd.Series`
of 35 values was passed where a label belongs; `drop` read the symptom *values*
as row labels and deleted rows 0, 1, 2, 4, 5… The frame and the class series
came out different lengths, the position↔position link broke, `knn` still
returned a class, the class happened to be right, and nothing raised.
`len(frame) == len(classes) == 306` is the assertion that catches it.

**`drop(index=175)` with `iloc[175]`.** Correct here only because the index is
the default 0–306 and untouched, so label and position coincide. Written the
coinciding way it is a guess that happens to hold. Positional and consistent:
`plant` from `iloc[i]`, and the drop label from `index[i]`. The same
label/position confusion has now entered the code twice, and in 1.7 the train
split's index no longer starts at 0 — at which point it stops being stylistic.

### The prediction, and what it explains

Row 1 is `diaporthe-stem-canker`, 10 in the set, 9 after removing itself. Asked
how many of 5 neighbours could at most come from its own class: 9 and 5 — both
right, and the second is the point. The limit is not the class size.

Those 9 must beat all 297 other rows. The four classes at 40 field 160
candidates against 9, so a rare class can lose every vote it should have won.
Accuracy barely notices — that class was 10 of 307 — while its macro recall goes
to zero and drags the average over 19. That is why `0.1303` and `0.0526` travel
with every result, and it is the argument 1.5 formalises.

`pd.set_option` at module level was moved: it changes pandas' display for every
module that imports this one, so it belongs in `z.py` or under the `__main__`
guard.

### Four distances, one model

Leave-one-out over every row, `k=5`. `make_gower` is built **once**, outside the
loop: dropping a single row cannot move `max` or `min` in any ordinal column —
measured, not assumed. The smallest extreme anywhere is `date = 0` at 12 rows;
everywhere else it is tens. The caveat is worth stating as a decision rather
than an oversight: a range computed over the whole frame has seen row `i`, the
row the model is pretending not to know. Harmless here, named in 1.7, the same
shape as target leakage in 4.8.

**Gower over all 307 rows:**
accuracy 271/307 = 0.8827 baseline 0.1303
macro recall 0.8263 baseline 0.0526


Seven times baseline on accuracy, fifteen times on macro recall.

**Prediction, before running: 0.52, below 0.80. Both wrong.** The reasoning was
that the top four classes are 52% of the data, so the model would mostly hit
those. But 52% is the ceiling for *guessing* one of the big four blindly — KNN
does not guess, it measures, and symptoms carry real information. A class's
share of the data bounds the **baseline**, not the model. `0.1303` was that
ceiling, and it had already been computed in 1.1.

### `k` is a threshold a rare class has to clear

`herbicide-injury` scored 0.000 with 3 neighbours still available, while
`cyst-nematode` scored 1.000 with 5. Class size is not what separates them.

With `k=5`, a class needs roughly 3 of the 5 votes to win. `herbicide-injury`
has 3 left after removing itself, so **all three** must land in the top five and
beat 160 candidates from the four classes at 40. One intruder at position three
and it loses — no possible set of neighbours saves it. `cyst-nematode` has
exactly 5, and survives only because it is *isolated*: all six rows share the
identical 11-column hole from 0.4, so Gower divides by 11 for pairs inside the
group and the group sits far from everything else.

Which makes that 1.000 the opposite of a success. It is not knowledge about
soybeans, it is recognition of who filled in the form — the MNAR leakage from
0.4, appearing for the first time as a number. In fitomedicina this is inverted:
a farmer's partial answers are noise, not signal. **Measured in 4.5.**

`2-4-d-injury`, n=1, scores 0.000 by construction: remove it and zero examples
remain, so no neighbour can carry its class. Predicted correctly, both halves.
That is also the 1.7 question about stratifying a class of one, answered early —
leave-one-out on a singleton class guarantees an error regardless of model.

### Only Gower can run on all 307

`euclidean` and `manhattan` return NaN on rows with holes — no mask, no column
dropping. `np.argsort` puts NaN **last**, so nothing raises: those rows are
simply never selected as neighbours. The model silently runs over a subset
nobody chose, and the subset is skewed by diagnosis because sparsity tracks
disease (0.4). A number comes out and it looks like an accuracy.

Three options. Masking gives 307 predictions but compares 35 columns for one
pair and 11 for another — the 1.2 finding about Gower across different `n`, and
worse here since Euclidean does not even divide. Imputation is a method with
three variants measured in 4.5, not a preparation step. **Chosen: filter to
complete rows**, because it is the only option where all four distances see
identical rows and identical columns, so the difference between four numbers is
the difference between definitions of "close" and nothing else.

The price was measured, not assumed: **266 rows, 15 classes.** Four classes
vanish entirely — `cyst-nematode`, `diaporthe-pod-&-stem-blight`,
`herbicide-injury`, `2-4-d-injury` — four of the five smallest.
`phytophthora-rot` drops 40 → 16, the bimodal split from 0.4. The filter
removed exactly the classes the 307-row run had found the model failing on. So
the two runs do not compare: 307 is the result, 266 is the experiment.

          accuracy   macro recall      (266 rows, 15 classes)
gower       0.9023         0.9250
hamming     0.8722         0.8717
manhattan   0.8609         0.8667
euclidean   0.8346         0.8325

**Prediction: Gower first, margin above 0.02. Right on both, on both measures.**
0.053 on macro recall, 0.030 on accuracy.

Gower above Hamming means the invented ordinal spacing **carries information** —
five one-step differences really are closer than four changes of state. Hamming
above Manhattan and Euclidean is the 1.2 `canker_lesion` permutation as a score:
those two read nominal codes as quantities, so they measure the labels rather
than the plants. Hamming, which never looks at the codes at all, beats both.
Euclidean is last because squaring amplifies large differences, and a large
difference between nominal codes is an accident of which integer got assigned.

But no distance dominates. Gower is **third** on `alternarialeaf-spot` (0.900 vs
0.925 for Hamming and Euclidean), second on `brown-spot` (0.925 vs Manhattan's
0.950), and Euclidean — the worst overall — is the only one perfect on
`anthracnose`. Gower wins by being perfect on 8 of 15 classes against Hamming's
6, mostly the classes at 10. Revisit in 8.1.

### The same difference, two sizes

Gower − Hamming is 0.030 by accuracy and 0.053 by macro recall. Class by class,
Gower gains on `bacterial-blight` 9→10, `bacterial-pustule` 7→8, `downy-mildew`
9→10, `phyllosticta-leaf-spot` 2→6, `purple-seed-stain` 9→10, and loses on
`alternarialeaf-spot` 37→36. Every gain is in a class of 10; the loss is in a
class of 40.

Accuracy counts those four `phyllosticta` plants as 4/266. Macro recall counts
them as 4/10 inside the class, lifting its recall 0.200 → 0.600, and the class
is 1/15 of the average — 0.027 from one class, nearly the whole accuracy margin.

This is 1.1's report card producing two different verdicts on one model for the
first time: accuracy is one vote per case, macro recall one vote per class.

**Not an artifact.** The word matters. Gower wins on accuracy too — 240 vs 232,
eight plants, same direction under both measures. Macro recall did not *create*
the advantage, it weighted it differently. It would be an artifact in the case
where the two measures disagree: 85+3 against 80+8 out of 90 and 10 both give
accuracy 0.88, while macro recall reads 0.622 against 0.844 — five cases traded
from the big class to the small one, no case gained, and a difference that
exists only inside that way of counting.

Which weight is right is not a statistical question. If every plant costs the
same, accuracy is the measure and the margin is 0.030. If every disease matters
equally — a farmer with a rare disease does not deserve a worse diagnosis for
being rare — macro recall is the measure and the margin is 0.053. That is a
product decision, the same shape as the cost asymmetry in 1.6.

### Two bugs the checks caught

`support_per_class` and `correct_per_class` both incremented inside the `if`, so
support was a second count of correct answers, summing to 271 instead of 307. A
`defaultdict` key exists only once something is written to it, so iterating over
`correct_per_class` printed 17 classes: `herbicide-injury` and `2-4-d-injury`
had never been hit, so they were not absent-with-zero, they were absent. Iterate
over **support**, which is filled unconditionally, and read the numerator as
`correct_per_class[c]` — a missing key returns 0 on a `defaultdict(int)`.

Denominator 306 with 307 predictions counted. Same slip as 1.1, where a macro
average was divided by the number of cases instead of the number of classes.
Checks that paid: support totals 307 across 19 keys and matches 1.1's
distribution row for row; `sum(correct_per_class.values()) == correct_predictions`.

### Ties, and a second kind nobody was looking for

**Ties in the vote: 11 of 266, 7 resolved correctly.** Predicted under 20 —
right; predicted 8–12 correct — right, at the edge. That is 0.636 against 0.899
overall, so ties are genuinely the harder cases, but better than the ~5.5 a coin
flip between two tied classes would give. On 11 cases that gap is inside the
noise: the rule is not worse than random, and no stronger claim survives.

`k=5` is odd, which prevents ties only for **two** classes. With 15 in play the
vote splits three ways and 2–2–1 is a tie regardless.

The rule `idxmax` was applying, unasked, is "nearest representative wins" — one
of the three options from §6 — because `value_counts()` orders equal counts by
first appearance and `nearest_classes` is sorted by distance. It is a reasonable
rule and it is now **written**: take the classes sharing the maximum, then walk
`nearest_classes` from the front and take the first one in that set. Not
`iloc[0]` of `nearest_classes` — on A, B, B, C, C the nearest neighbour is A,
which lost the vote. Numbers unchanged at 240 / 0.9250 / 12 / 8, which is the
proof the rule is the same one.

The stake was measured before deciding: 4 predictions of 266, at most 0.015 on
accuracy. No option from §6 would be distinguishable from another. The decision
worth making was not which rule but that the rule is stated, since it was a
consequence of how pandas sorts and would have changed silently underneath.

**The second kind: ties in the distance itself.** `np.argsort` defaults to
quicksort, which is not stable — with equal distances the order is arbitrary and
not reproducible. When the 5th and 6th neighbours are equidistant, one enters
the vote and the other does not, and quicksort picks. `kind="stable"` moved the
result: 239/0.9233/11 against 240/0.9250/12. Verified by reverting it alone.

**`kind="stable"` stays, and it is a decision, not a style.** It is worse by one
prediction and identical every run. A model that cannot reproduce its own number
cannot be measured, and 1.7 onward compares numbers across runs.

These ties are frequent here, not incidental. Gower is a sum over columns where
nominal contributes 0 or 1 and ordinal a fraction with a small denominator —
nothing continuous anywhere. Few distinct values are reachable across 35
columns, and 265 pairs per plant fall into them. With real measurements two
identical distances would be a coincidence; on integer codes they are the rule.
Third appearance of 1.1's principle: every tie is broken by someone, and here
the someone was neither him nor the data order but an unspecified sort.

**Final numbers for 1.3, both rules written:**

                accuracy                macro recall
gower, 307 rows 0.8827 0.8263 baseline 0.1303 / 0.0526
gower, 266 complete 0.8985 0.9233 11 ties, 7 correct

`complete = data.notna().all(axis=1)` replaced a list of labels indexed with
`.iloc`. Same 266 rows; the old form worked only because the index was still
0–306. Fourth time label-versus-position entered this file.


## 1.4 — Confusion matrix
*2026-09-14*

Seven lines of function, and the whole point of the step is that it measures
nothing new. `support_per_class` is a row sum, `correct_per_class` is the
diagonal, `correct_predictions` is the trace. Three counters from 1.3 were
summaries of an object that was never built. Now the object exists and the
summaries are readings off it.

### What survives the two arrays, and what does not

Of the five counters the 1.3 loop carried, three fall out of `y_true` and
`y_pred` alone. Two do not: `shared_maximum` and `accurate_shared_maximum`,
both of which need `vote_counts`.

The rule that separates them: **the two arrays record the decision, not the
evidence.** From a pair "actual, predicted" there is no way back to whether the
vote was 3–2 or 2–2–1. Anything asking *how* the model arrived at an answer has
to be measured while the answer is produced; anything asking only *what* the
answer was can wait.

That settles where the tie counters live — inside the loop, permanently. They
are not scaffolding the matrix will eventually replace.

### The class list is an argument, not a derivation

Three candidate sources for the third parameter: read it off `y_true`, off
`y_pred`, or take it from the caller. Chosen: the caller.

A matrix that reads its own class list from what happened cannot know about a
class that did not happen. It builds a smaller table and the table looks
complete. That is 1.3's `defaultdict` bug moved one level up — there, a class
the model never predicted had no key, so the printout showed 17 of 19 and
nothing was missing-with-zero, it was missing. The caller holds the whole
dataset and knows what was *possible*, which is a different question from what
*occurred*.

### The shape is declared, not discovered

Predicted 15 rows and 19 columns if 19 classes were passed. Wrong:
`np.zeros((len(classes), len(classes)))` uses the same number twice, so the
matrix is square by construction.

The expectation came from treating the matrix as something the data produces —
15 diseases occurred, 19 were possible, so the axes should differ. It is
something the caller *declares*, which is the same fact as the third argument
being explicit, seen from the other side.

### 15 classes, not 19

Passing 19 gives a 19×19 with four rows **and** four columns of zeros:
`cyst-nematode`, `diaporthe-pod-&-stem-blight`, `herbicide-injury`,
`2-4-d-injury`, all removed by the complete-case filter. In 1.5 the recall of
those four is 0/0 — a division by zero, or a silent `nan`, or a class reading
0.000 that looks like model failure when the model never met the disease. An
artifact of the filter, not a result.

The matrix describes **one measurement**, and that measurement saw 266 rows and
15 diseases.

Counterargument rejected: a fixed 19×19 makes every run the same shape and
comparable cell by cell. But 307-Gower and 266-Gower do not compare anyway
(1.3), so there is nothing to align.

### `make_gower` goes after the filter

First draft built the closure on all 307 rows and filtered to 266 afterwards.
The closure freezes `ordinal_ranges` at call time, so the filter cannot reach
them: every distance in the run would divide by a range that had seen 41 rows
the run claims not to know.

1.3 established the opposite for one row — `make_gower` is built once, outside
the leave-one-out loop, because dropping a single row cannot move `max` or
`min` in any ordinal column, and that was *measured*. 41 rows is a different
claim and was not measured. **A measurement for n=1 is not a measurement for
n=41**, particularly when the 41 are selected by a rule — the MNAR pattern from
0.4 — rather than at random.

### Two places chosen to be loud

`zip` stops at the shorter list and says nothing. That is 0.5's finding by
another route: a length mismatch does not reliably raise. `strict=True` turns
it into `ValueError`. It matters because `y_true` and `y_pred` are appended in
the same loop, and the moment one `append` ends up inside an `if`, the lists
drift and the matrix fills anyway — just wrongly.

`class_to_index[actual]` raising `KeyError` on an unknown class is the same
choice, against 1.3's `defaultdict`, which was silent. Neither is defensive
coding. Both are: when two things disagree, fail rather than produce a
plausible table.

### `int64`, and where `isclose` does not belong

`np.zeros` returns float by default. Cells are counts — 0, 1, 28, never 3.5 —
and counts are compared with `==`.

0.5's rule is about *measurements*: three summation algorithms give three
different sums, so comparisons against a reference use `isclose`. Counting has
no such spread. The row-sum check was first written with `np.isclose`, which is
both the wrong tool and returns an array of 15 booleans instead of one verdict.
`np.array_equal`.

### pandas 3.0: `Series[key]` no longer falls back to position

Checked, because I was told the opposite and ran it.

s = pd.Series([10, 20, 30], index=["date", "precip", "temp"])
s[1] -> KeyError: 1 (pandas 3.0; earlier versions returned 20)

s = pd.Series([10, 20, 30], index=[5, 6, 7])
s[5] -> 10 (label wins over position)


The positional fallback existed for years, was deprecated through 2.x, and is
gone in 3.0. What survives is the part that bites: **`s[key]` means label, and
when labels happen to be integers that coincide with positions, label and
position agree until they don't.** That is the label-versus-position bug that
entered this file four times in 1.3, stated as a rule instead of a habit.

Consequence in the loop: `gower` reads `a[0]`, `a[1]` positionally, so the test
plant is wrapped in `np.array(...)`. Not because a `Series` would work silently
— on this version it raises — but because an array has positions and nothing
else, so `a[0]` has exactly one meaning.

### The matrix

15×15, rows actual, columns predicted, both alphabetical because `np.unique`
sorts.

Three checks, of which only one can fail on a transpose:

all cells 266 every pair landed in exactly one cell
trace 239 accuracy 0.8985, unchanged from 1.3
row sums match value_counts, class by class


266 and 239 come out identical on a transposed matrix. Row sums do not —
transposed they count how often each class was *predicted*, a different number.
**Write the check that can fail.**

Nothing here is a new measurement: 239/266 = 0.8985, and the mean of the 15
per-class recalls is 0.9233. Both were already in 1.3.

One number did move. 1.3 recorded `alternarialeaf-spot` at 36 of 40 in the
pre-`kind="stable"` table; it is 35 here. That is the single prediction
`kind="stable"` cost — 240 → 239 — and it now has an address.

### The finding: the distance is symmetric, the matrix is not

frog-eye-leaf-spot -> alternarialeaf-spot 11
alternarialeaf-spot -> frog-eye-leaf-spot 2


Two classes of 40 each, and `gower(a, b) == gower(b, a)`.

*Predicted before looking: 9–12 of the 12 lost `frog-eye` cases land in
`alternarialeaf-spot`. Actual 11, with 1 in `brown-spot`. Correct, on a narrow
band. Reasoning given: both are leaf spots, both sit at 40.*

The resolution is that a cell is not a statement about a pair of classes.
**Cell (A, B) counts how many A-points had a B-majority neighbourhood** — a
property of where A's points sit relative to everything else, not of the
relation between A and B. Symmetry of the distance constrains pairs. It says
nothing about neighbourhoods.

A compact class overlapped by a diffuse one: every compact point's five nearest
are compact points, while many diffuse points' five nearest are also compact
points. The leak runs one way. Here `alternaria` holds its centre — 35 of 40,
and its 5 misses scatter across three classes — while `frog-eye` leans into it:
28 correct, 11 into `alternaria`, 1 into `brown-spot`. Twelve misses, eleven at
one address.

A second and separate asymmetry, worth not confusing with the first:
leave-one-out always removes a member from the true class and never from the
rival, so every point is judged against a class one short. Decisive for a
singleton — `2-4-d-injury` at 0.000 in 1.3 — and negligible at 39 against 40.
Density explains 11 against 2; the leave-one-out tax does not.

**What the matrix gives that 1.3 could not.** Accuracy says how much was
missed. Per-class recall says which class missed. Only the matrix says *where
the miss went* — and the misses turn out not to be smeared, they have a
destination.

### Agronomic sense, which the step asks for

Both diseases produce round leaf lesions with a dark margin and a lighter
centre, and they are confused in the field by people. The visible difference
photographs point to is the yellow halo around the lesion — which is
`leafspots_halo`, a column that exists in `.names` because somebody decided
that was the thing separating similar spots.

The model never sees a photograph. It sees 35 recorded answers, and the 11
leaked cases are the cases where those answers were not enough. **4.2 measures
exactly this**: how many bits of uncertainty `leafspots_halo` removes. The
value of a question is not that it marks a real biological difference; it is
how much of the remaining confusion it resolves — and this matrix is where the
remaining confusion sits.

### Open

**The three distances are still pre-`kind="stable"`.** `hamming`, `manhattan`
and `euclidean` were scored before the stable sort, so 1.3's four-number table
is mixed. Now one edit per distance, since the distance is a variable rather
than hard-coded into the `knn` call.

Needed before **`phyllosticta-leaf-spot`, 0.600 under Gower against 0.200 under
Hamming** — the widest gap between two distances in the table — can be read off
a matrix. Its row here is 6 of 10, with 3 into `alternaria` and 1 into
`brown-spot`. The Hamming matrix is what says whether the same class absorbs
the other four.

**The four-way tie at 40**, still unresolved from 1.1. And `baseline.py` still
runs at module level with no `__main__` guard.


### The debt from 1.3, paid — and it cost a sentence

`hamming`, `manhattan` and `euclidean` were scored before `kind="stable"`.
Re-run now that the distance is a variable at the top of `__main__` rather than
hard-coded into the `knn` call. One edit per distance.

    before stable    now    delta
gower         240    239       -1
hamming       232    229       -3
manhattan     229    231       +2
euclidean     222    223       +1


*Predicted: 1–3 predictions move across all three distances. Actual: at least
6.* And that is a floor, not a count — the trace shows the **net**, so three
falling and none rising is indistinguishable from five falling and two rising.

The reasoning behind the prediction was extrapolation from Gower's single
moved prediction. Gower turned out to be the least affected of the four, so it
was the wrong one to extrapolate from. **Generalising from the best-behaved
case is not generalising.**

### The ordering flipped, and a 1.3 finding does not survive it

before: gower 240, hamming 232, manhattan 229, euclidean 222
now: gower 239, manhattan 231, hamming 229, euclidean 223


1.3 recorded: *"Hamming above Manhattan and Euclidean is the 1.2
`canker_lesion` permutation as a score"* — Hamming wins because it refuses to
read nominal codes as quantities. That claim rested on a 3-prediction gap.
The gap is now 2 predictions the other way, 0.0075 of accuracy. **Neither
ordering supports the story.** The sentence goes.

What survives: Gower first with a clear margin, Euclidean last. The middle is
undecided, and the 1.2 argument about permuting nominal codes is still correct
as an argument — it just has no support from this table.

### Why three and not one: resolution, and a blind cut

`k=5` takes the first five of a sorted list. If the 5th and 6th neighbours are
equidistant the distance does not choose between them — the sort does. Call that
a **blind cut**: the boundary falls inside a block of equal values rather than
between two distinct ones.

Distances from position 0 to the other 265, distinct values:
gower 110
hamming 20

Hamming over 35 columns can only return 0 through 35 — **36 reachable values
for any pair, on any dataset of this width**, and only 20 of them occur here.
Gower divides by the column count and nine ordinal columns contribute
fractions, so far more values are reachable.

**The average group size argument, and why it is wrong.** 265 neighbours over
20 values gives ~13 per value, which predicts the cut at position 5 lands
inside a block almost always. Position 0's actual Hamming groups:
distance 0 4 5 6 7 8 9 10 11 12
count 1 2 2 3 2 1 3 20 18 22


Cumulative 1, 3, 5 — the 5th neighbour is the last of its group and the 6th
starts a new one. **Clean cut, for this plant.** The groups near zero are 1, 2,
2, 3; the mass sits at distance 10 and beyond, in groups of ~20. The
distribution is nowhere near uniform, so the mean describes none of it, and the
cut at position 5 lands in the sparse tail rather than in an average block.

Same shape of error as 1.2's disproved Gower bound: **an argument resting on how
values "usually" distribute is an assumption wearing a formula.** Measure the
thing instead.

### Blind cuts, measured

Test per plant: take `np.unique(distances, return_counts=True)`, cumulative-sum
the counts, and ask whether **5 appears among the cumulative sums**. If it does,
some group ends exactly at the 5th neighbour and the cut is clean. If it does
not, the 5th and 6th are in the same group and the cut is blind.

Over all 266 plants:

     blind cuts   of 266    moved by kind="stable"
hamming     176      66%                        3
manhattan   158      59%                        2
euclidean   137      52%                        1
gower        62      23%                        1


*No prediction recorded — the measurement arrived first. Third time this
session.*

Three orderings agree: blind-cut frequency, sort sensitivity, and the results
table upside down. Gower is the least blind and the best; Hamming the most
blind and the most sensitive.

**But 176 blind cuts produce 3 moved predictions.** A blind cut almost never
changes an answer, because three conditions have to chain: the tied group must
straddle the boundary, it must contain more than one class, *and* the vote must
be close enough for one swapped voter to flip it. Being unable to order the
neighbours is not the same as getting them wrong.

This corrects the earlier claim that a 2-prediction gap "sits inside the noise
the sort generates". The noise is now bounded: 3 predictions for Hamming, and
that is an upper bound, because 3 is everything the switch to a stable sort
moved. Two predictions is still inside it — but because the gap is tiny, not
because the noise is large.

**The ratio that is not a finding.** 176/62 ≈ 2.8 blind cuts, 3/1 moved
predictions. It is tempting to read linearity out of that. It rests on the
numbers 3 and 1: had Gower moved 2 instead of 1, the ratio would be 1.5, and
1.3's analysis already showed how fragile a single prediction is here. Same
error as the average-group-size argument two sections up, and as the 1.2 bound.
**Recorded as a coincidence, not a rule.**

### What this leaves standing

The 1.3 explanation for Gower's win was that the invented ordinal spacing
carries information. This measurement adds a second, independent mechanism:
Gower can *order* its neighbours at all, and Hamming largely cannot — 20
distinct values against 110.

Two mechanisms, same direction, not separated here. 8.1: a Gower variant with
every ordinal column forced to nominal would isolate the first from the second.

### Still open

**The four-way tie at 40**, unresolved since 1.1. And `baseline.py` still runs
at module level with no `__main__` guard.

### The four-way tie at 40, resolved by looking

Open since 1.1: four classes hold exactly 40 cases — `phytophthora-rot`,
`brown-spot`, `alternarialeaf-spot`, `frog-eye-leaf-spot` — so "the most
frequent class" is not unique. `idxmax` returns the first maximum, and
`value_counts()` orders equal counts by first appearance, so the winner is
decided by row order in `soybean-large.data`.

Accuracy does not care — 40 over the total either way — which is why 1.1 was
unaffected and the item could wait. The matrix does care: a model that always
answers the same disease fills **exactly one column**, and which column is the
whole shape of the matrix.

**Measured, and it is worse than "arbitrary":**
idxmax on 307 rows -> phytophthora-rot
idxmax on 266 rows -> brown-spot


Same code, same function, one filter in between, and the baseline predicts a
different disease. `phytophthora-rot` falls 40 → 16 under the complete-case
filter (0.4's bimodal split), so it stops being a candidate; the other three
stay at 40 and the tie is still three-way, still broken by file order. **No rule
was applied and no rule changed. The set of candidates changed.**

### The baseline moves with the set
                307 rows, 19 classes    266 rows, 15 classes
accuracy                      0.1303                  0.1504
macro recall                  0.0526                  0.0667


*Computed by hand before running, both correct:* 40/266 and 1/15.

The filter removed four rare classes, so the largest class holds a larger share
and the per-class mean divides by fewer classes. **The baseline went up.**

Which is the reason it has to travel with its own set. Gower on 266 is 0.8985,
6.0× its baseline. Gower on 307 is 0.8827, 6.8× its baseline. The raw number is
higher on 266 and the ratio is lower — an easier task and a better score are
not the same thing.

### Baseline against Gower, both on 266
                     baseline    gower
non-zero columns            1       15
classes with recall>0       1       15
diagonal                   40      239
accuracy               0.1504   0.8985
macro recall           0.0667   0.9233


One column against fifteen. The baseline's matrix is what "knows nothing" looks
like when you stop summarising it into a number: every row points at the same
answer, and 14 of 15 diseases are never named.

**A wrong `classes` argument produced a matrix that was internally correct and
not comparable.** First attempt passed the per-row class Series instead of
`unique_classes`, so the axes came out in order of first appearance rather than
alphabetically. All three checks passed — 266, diagonal, row sums — because each
is invariant to relabelling the axes. Only lining the row sums up against the
Gower matrix, class by class, showed it. Same trap as `value_counts()` against
`np.unique` in the row-sum check: **two aligned-looking vectors of 15 numbers,
not aligned.**

Third instance of 1.1's rule this step. Every tie is broken by someone: here the
class tie by file order, the neighbour tie by an unspecified sort, and the
distance tie at the `k` boundary by the sort again.

### Open

The baseline matrix was computed in `z.py`. It belongs in `baseline.py`, which
still runs at module level with no `__main__` guard — one edit covers both.

### `baseline.py`, rewritten

Module-level body moved under a `__main__` guard — open since session 3, and it
started to matter once anything imported across `phase1_evaluation`.

Both baselines now live in the file rather than one in the file and one in the
journal. 307 is the result, 266 is the comparison, and a number that exists only
in prose gets quoted wrongly within a month.

**One method, not two.** The old code computed macro recall as `1 / nunique`.
That formula is correct only because the majority-class baseline gets recall 1
on exactly one class and 0 on every other — it is not a definition of macro
recall, it is an arithmetic accident of this one model. Left in the file it
would keep printing a plausible number for the first model that partially hits
two classes. Both rows are now read off the matrix: `np.diag(cm) / cm.sum(axis=1)`,
then the mean.

Verification that the rewrite changed nothing: the all-rows row reproduces 1.1's
`0.1303` and `0.0526` exactly. Same check as `make_gower`'s bit-identical ranges
— a refactor is proved by the number it does not move.

              rows  classes  majority_class     accuracy  macro_recall
all rows       307       19  phytophthora-rot     0.1303        0.0526
complete rows  266       15  brown-spot           0.1504        0.0667


`assert cm.sum() == len(classes)` sits inside the function, not at the call
site. A matrix that had lost rows would still return a plausible accuracy,
since numerator and denominator both come from it. Putting the check outside
would mean returning the matrix to a caller that has no other use for it —
the 1.3 rule about a counter escaping `knn`, in a second place.


## 1.5 — Precision, and two ways to average
*2026-09-17*

Nothing new is computed here either. Precision was already in the 1.4 matrix;
what was missing was the reading. Recall divides the diagonal by its row,
precision divides the same diagonal by its column, and
`matrix[actual, predicted]` is what makes the row the disease and the column the
answer.

### One mistake, counted twice

Every off-diagonal cell belongs to one row and one column, so it is two errors
at once: a miss for the disease the plant has, a false alarm for the disease the
model named. The 11 frog-eye plants called alternaria in 1.4 are the same 11 in
both readings — frog-eye's row, alternaria's column.

That gives a mirror pair under Gower:

    frog-eye     recall 28/40 = 0.700    precision 28/30 = 0.933
    alternaria   recall 35/40 = 0.875    precision 35/51 = 0.686

Frog-eye is missed one time in three and is almost never wrong when named.
Alternaria is caught well and is the least trustworthy name in the model: of the
51 plants given that name, 16 do not have it — 11 frog-eye, 3 phyllosticta, 2
brown-spot. Its column is where the other leaf spots' errors collect. 1.4's
asymmetry, measured on the metric it damages.

The same shape in miniature, and cheap to check by hand: bacterial-pustule has
recall 0.800 and precision 1.000, bacterial-blight recall 1.000 and precision
0.833. Two plants explain all four numbers. Blight misses nothing and still does
not reach precision 1.0 — a class can be perfect in its row and imperfect in its
column.

### Precision is a reading, not a cause

Predicted that Hamming's three extra phyllosticta losses would go to brown-spot
rather than alternaria — right (brown-spot 1 → 4, alternaria stays at 3), but
the reason given was alternaria's precision. Wrong number (0.686, the lowest in
the model) and wrong kind of number.

Precision is computed after every answer is in. The model, at the moment it
decides, sees five distances and counts votes; it has no access to any metric
about itself. An explanation of where an error goes must come from something
visible at decision time. Here that is Hamming: it scores each column 0 or 1 and
cannot tell one step from three, so the ordinal degree separating the three leaf
spots disappears, and brown-spot — 40 plants, tied with alternaria and frog-eye
for the largest class on 266 — wins the vote more often.

**A right answer can arrive with the wrong mechanism**, which is why the reason
is asked for and not only the band.

### The empty column

Recall cannot be 0/0 here: the class list comes from the cases actually present,
decided in 1.4, so every row holds at least one plant. That guarantee lives at
the call site, not in the function, and stays written there.

Precision can be. An empty column means the model never once said that name —
not that it was wrong, and not that the class is missing from the data. Three
possible returns: 0, `nan`, or drop the class.

**`nan`, for the same reason `make_gower` returns `nan` on an empty mask in
1.2.** Zero asserts that the model was wrong every time it used a name it never
used. Sklearn does return 0 there, with a warning.

Dropping is worse, and `np.nanmean` does exactly that. A model naming 2 of 15
classes at precision 0.90 then beats a model naming all 15 at 0.75: silence
about thirteen diseases is rewarded, because the average is taken only over the
names the model chose to risk. `np.mean` instead carries one `nan` up to the
model, where it says that macro precision for this model does not exist.

Predicted that a `nan` in the summary gives no information — wrong, and
backwards. Accuracy exists for the baseline and is 0.1504; the `nan` beside it
is the fact that 14 of 15 diseases were never named at all, which no other
number in that row states.

### Fixed before the division, not suppressed after

`per_class_precision` replaces the zero in the denominator before dividing, so
there is no 0/0 anywhere and `np.errstate` is not needed — verified under
`np.errstate(all="raise")`, which raises nothing. Dividing first and silencing
the warning afterwards would have been the worse half of the same fix.
`np.where(..., np.nan)` also promotes int64 to float64, which is what lets the
`nan` survive into the array instead of becoming 0 in an integer division.

Both functions take the matrix and nothing else: no data, no `y_true`, no class
list.

### The baseline's precision is its accuracy

`majority_class_baseline` now reads both: 14 `nan` of 15 on the 266 complete
rows, 18 of 19 on all 307. The one real number is brown-spot's 0.1504, which
equals that row's accuracy exactly, and phytophthora-rot's 0.1303 equals the
other's. Not a coincidence: every plant is predicted brown-spot, so its column
is the whole data set, and TP over the column sum is the trace over the total.
In the degenerate model the two readings collapse into one.

### Accuracy is macro recall, weighted by support

Both come out of the same 15 numbers. Macro recall gives every class 1/15.
Accuracy gives every class its share of plants, because support × recall is that
class's diagonal cell, so the support-weighted sum of the 15 recalls is the
trace. Checked on Gower: 0.8984962406015038, identical to
`np.trace(cm) / cm.sum()` to the last digit.

So macro recall sits above accuracy exactly when the small classes have the
better recall, and below when they have the worse. Under Gower it is strongly
the first case. Nine of 15 classes have recall 1.000 and all nine are small —
106 of 266 plants. They hold 9/15 = 60% of macro recall and contribute 0.600 of
its 0.9233, against 106/266 = 40% of accuracy. Every error is inside the three
large leaf spots.

    Gower     accuracy 0.8985   macro recall 0.9233   gap 0.025
    Hamming   accuracy 0.8609   macro recall 0.8658   gap 0.005

Predicted Hamming's macro recall above its accuracy — right, no reason offered,
and the size is the interesting part: the gap is five times smaller. Hamming is
better than Gower in the two largest classes (alternaria 0.900 against 0.875,
brown-spot 0.950 against 0.925) and pays in the small ones: phyllosticta falls
to 0.300, bacterial-pustule to 0.600, and only five classes stay at 1.000. A bad
small class costs a full fifteenth of macro recall and 10 plants of 266 in
accuracy. That is what closes the gap.

1.1's report card and 1.3's two verdicts on one model, now with the arithmetic:
one set of readings, two weightings, and the question each answers is how many
plants against how many diseases.

### The denominator, three times wrong

Every hand-computation slip in this step was the denominator, and each time it
came from a measure other than the one being computed.

- Frog-eye's precision, before the definition landed: 28 over 266. That is
  accuracy's denominator.
- Then 2/2, once the column was found: the false alarms alone, with the diagonal
  left out of its own denominator.
- Then 12/40 = 0.3 and 4/10 = 0.4 offered as recalls. Those are the miss
  fractions. Recall counts what was found: 0.700 and 0.600.

The third changed an answer and not only a number: it inverted which class is
weaker, and phyllosticta at 0.600 is the lowest recall in the model.

Same root, twice more. The gap between accuracy and macro recall was first
explained by true negatives; TN appears in neither measure, and specificity is
the only metric in this step that uses it. And frog-eye was predicted to hurt
macro recall more than phyllosticta because it misses 12 plants against 4 —
accuracy's rule applied to a macro average, where 12 of 40 is a smaller hole
than 4 of 10.

### Open

Sensitivity and specificity, F1, and micro averaging are still ahead in 1.5.
Specificity is the first metric here that uses TN, and with 15 classes TN is
most of every one-vs-rest table.