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
*pending*