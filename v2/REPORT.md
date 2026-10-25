# Predictor v2: research report

As of 2026-10-25 (data snapshot of 2026-10-23). Everything here was computed from a read-only snapshot of the production
database (96,566 pro results, 7,777 divisions, 1947 to 2026). Nothing was written to the database.

## Headline

On every pro division judged from January 2024 to October 2026 (2,610 divisions), predicted
strictly before each show and refitted every January as production would:

| | v2 | uniform guess |
|---|---|---|
| predicted favourite wins | **54.3%** | 15.0% |
| overlap of predicted and actual top 3 | **72.0%** | 41.9% |
| Spearman, predicted vs actual order | **0.60** | 0 |
| winner log-loss (lower is better) | **1.280** | 2.185 |

Against the v1 production predictor on the 259 divisions of 2026 that v1 scored before the show
(both judged on the same athletes):

| | v1 (live) | v2 |
|---|---|---|
| predicted favourite wins | 37.8% | **54.4%** |
| top-3 overlap | 62.5% | **73.2%** |
| Spearman | 0.40 | **0.69** |
| winner log-loss, 79 divisions where v1 stored a win % | 1.503 | **1.465** |

Win percentages are close to honest for underdogs and outsiders: athletes given 10 to 20% won
14.3% of the time (predicted 14.3%). **Favourites run a few points high:** 70 to 80% won 67.7%
(predicted 74.8%) and 90%+ won 92.2% (predicted 94.3%); 30 to 40% won a bit more often than
predicted (36.1% vs 34.7%). Rounds 2 and 3 tried five fixes for this and none held up (see below).

## How it was tested

* **Walk-forward, no leaks.** Every division is predicted from results dated strictly before it.
  All divisions sharing a date are predicted before any of them is learned from, so an athlete doing
  the Open and the Masters class of one weekend never informs himself. `tests/test_core.py` guards
  this and fails when the guard is sabotaged.
* **The whole lineup counts.** Win % is scored over everyone listed in the division, including
  athletes without a placing (no-shows, unranked; 720 such rows in 2024 to 2026). Field size is the
  lineup size, which is known before the show. A test guards this too.
* **Tuning 2018 to 2023, test 2024 to 2026.** Rating hyperparameters, the stage-2 inputs, the
  boosted model's iteration count, the separate Masters layer, the 50/50 blend and every round-2
  choice were made on 2018 to 2023 (round-2 screens used 2022 and 2023, each predicted by a stage 2
  fitted on the years before it). The stage-2 layer is refitted every January on everything before
  that year, and its temperature is fitted on the last year of its training rows by a model that did
  not train on them.
* **Caveats.**
  * I saw test numbers while developing round 1.
  * In round 2 the 2024 to 2026 years were run four times:
    1. the first run exposed a numerical bug;
    2. the second decided against a probability shrink;
    3. the third was after dropping the shrink;
    4. the fourth was after the code-review fixes.
  * In round 3 they were run three more times: twice with inputs that turned out to leak (see the
    round-3 section), then once with the clean model. The clean model's choices were all made on
    2021 to 2023.

  So 2024 to 2026 is no longer an untouched holdout, and the figures may be slightly optimistic.
  The v1 comparison does not share this issue.
* **Metrics.** Headline: winner log-loss (a proper scoring rule for "win %"). Also top-1, top-3
  overlap, Spearman, top-5 Plackett-Luce likelihood, Brier and calibration tables.

## The model ladder (same 2,610 test divisions)

| Step | winner log-loss | favourite wins | top-3 overlap | Spearman |
|---|---|---|---|---|
| Uniform (everyone equal) | 2.185 | 15.0% | 41.9% | 0.00 |
| Recent placings heuristic (v1 style) | 1.594 | 43.5% | 66.3% | 0.45 |
| Weng-Lin Bayesian Plackett-Luce (OpenSkill style), tuned | 1.681 | 42.3% | 64.5% | 0.48 |
| Multiplayer Elo, tuned | 1.436 | 48.2% | 68.7% | 0.51 |
| Time-weighted reverse Plackett-Luce (MAP refit per show date) | 1.376 | 49.8% | 69.8% | 0.54 |
| + Elo and five more rating views, blended | 1.349 | 50.6% | 70.6% | 0.56 |
| + status, layoff and division-switch features | 1.332 | 52.3% | 71.2% | 0.58 |
| + boosted ranker, separate Open and Masters layers, yearly refit (round 1) | 1.300 | 53.3% | 71.9% | 0.59 |
| + Masters age groups, head to head, qualification (round 2) | 1.281 | 53.8% | 72.0% | 0.59 |
| + time away, travel for masters, unknown-country leak removed (round 3, final) | **1.280** | **54.3%** | **72.0%** | **0.60** |

What did not help: Weng-Lin online updates (worse than plain Elo here), judge-score margins as
features, home-country advantage, damping later picks when simulating placings (Benter style), and
the round-2 and round-3 ideas listed below.

## The final model

**Stage 1, eight rating views per athlete and division.** Ratings are kept per (athlete, division
type, open or natural circuit).

* The main one is a **time-weighted Plackett-Luce model refitted by MAP on every show date**. Each
  earlier result counts with weight `exp(-age / 730 days)` and every athlete has a normal prior.
  Because it refits, a result is reinterpreted when we learn more about the opponents.
* It is fitted **bottom up ("reverse" PL)**: last place is chosen first. This beat the usual
  top-down order clearly (judges agree most about who is out of contention). The "not placed"
  tie block at 16th is treated as unordered.
* The other seven views:
  * the same PL forward (top down);
  * a short-memory version (300 days) and a long-memory version (1,500 days);
  * a PL that keeps **Masters results apart** from the Open rating;
  * a multiplayer Elo (also Masters apart);
  * a PL and an Elo kept **per Masters age group** (40+ and 50+ rated apart).
* The likelihood is computed in the log domain per contest, so far-apart contests and wide spreads
  inside one contest cannot underflow.

**How shows are weighted (the Olympia vs Arnold vs local pro question).**

* The main lever is field strength. Every result is judged against the ratings of the people on
  stage, so beating an Olympia or Arnold field moves a rating far more than winning a thin local
  show.
* On top of that, the Plackett-Luce views count an Olympia 1.5x and a Masters Olympia 2/3x.
* Tier-2 shows (Arnold, NY Pro, Pittsburgh, Romania, as tier 2 in our qualification rulesets) stay
  at 1.0 there, because raising them did not improve predictions (round 1, and again in round 2). The
  Elo view counts them 1.5x, and the stage-2 layer knows each show's tier.
* Dubai Pro is tier 3 in our rulesets, so it is tier 3 here too.

**Stage 2, a ranking layer over ratings plus context** (Benter style, as in horse racing): a
conditional logit and a LightGBM lambdarank model, blended 50/50, fitted separately for Open-age
and Masters classes.

Inputs for both layers besides the ratings:

* latest Olympia placing (in, top 6, top 3, champion) and ever top 3 at an Olympia;
* last edition of this same show (won it, 1/place);
* pro wins, shows in this division, shows in the last year, years since pro debut;
* layoff since the last show (back to back, short, 1 to 2 years, more than 2 years);
* rookie, or switcher from another division (with their rating there);
* masters-only athlete, recent form and trend, home country;
* the show tier, division and lineup size (boosted model only).

**The Masters layer also gets (round 2):**

* the athlete's record in this exact age group: shows, wins, latest result, first time in it, just
  moved up a bracket;
* the two per-age-group ratings;
* head to head against this lineup over the last three years, and against their highest-rated rival;
* Olympia qualification: top 3 at the latest Olympia or an Open-age pro win since. This is
  qualification for the **Open** Olympia of the division, used as a strength signal; Masters
  Olympia qualification is not modelled. A flag also marks a regular show and the 8 weeks before
  the next Olympia, which is assumed to fall a year after the last one.

**Both layers also get (round 3)** time away: years since the athlete's last show in any division
and in this one, and their base rating scaled by how stale it is. **The masters layer also gets**
travel: the athlete's country differs from the event's, and a non-US athlete at a show in the USA.

An athlete with no country on record counts as being from the event's country. Countries are filled in
after the fact, mostly for athletes who became notable, so "unknown" must not be visible to the model.
Where the head-to-head input picks a rival among tied athletes it goes by athlete id, never by list
position (stored lineups are in finishing order). Tests check both.

**Probabilities.** Win % is the softmax of the stage-2 strength. P(2nd), P(3rd), top 3, top 5 and
expected placing come from 20,000 Plackett-Luce simulations (strength plus Gumbel noise).

## Round 2: the improvement round

**Task:** re-tune with the fixed scoring, try per-division settings, Masters age groups, a fix for
over-confident favourites and new features. Tune on 2018 to 2023, confirm on 2024 to 2026, keep
only what helps. A change counted as helping only if it gained at least 0.002 in log-loss.

**Before and after, 2024 to 2026:**

| | Before (round 1) | After (round 2) |
|---|---|---|
| winner log-loss | 1.300 | **1.281** |
| favourite wins | 53.3% | **53.8%** |
| top-3 overlap | 71.9% | 72.0% |
| Spearman | 0.592 | 0.593 |
| Masters classes: log-loss | 1.218 | **1.178** |
| Masters classes: favourite wins | 53.1% | **53.8%** |
| Open-age classes: log-loss | 1.382 | 1.382 |
| Natural circuit: log-loss | 1.177 | 1.182 |
| calibration error (win %) | 0.35 points | 0.43 points |

By Masters division, favourite wins rose most in:

| Division | Before | After |
|---|---|---|
| Bikini | 47.7% | 51.6% |
| Women's BB | 48.4% | 55.6% |
| Fit Model | 39.5% | 48.6% (33 divisions) |
| Men's Open | 51.4% | 53.2% |

They fell in Classic, Men's Physique, Women's Physique and Wellness (by 1 to 5 points). Log-loss
improved in most Masters divisions.

**On 2018 to 2023 (2022 and 2023 folds):**

| Model | log-loss | Masters log-loss |
|---|---|---|
| Round 1 | 1.406 | 1.288 |
| A: Masters inputs (final) | **1.385** | **1.238** |
| B: A plus re-tuned weights plus per-division settings | 1.384 | 1.235 |
| C: A plus re-tuned weights only | 1.391 | 1.248 |

**Kept:** the Masters inputs (candidate A). Each of the four groups helped the Masters layer alone
and together on 2018 to 2023; they were confirmed as one bundle on 2024 to 2026.

**Rejected:**

* **Re-tuning the main rating.** The rating on its own improved from 1.598 to 1.596 (Olympia
  2.5x, tier 2 1.3x, Masters Olympia 0.4x, stronger prior). In the full model it made things worse
  (C).
* **Per-division decay and prior strength.**
  * On the rating alone, Bikini and Fitness preferred short memory, and Classic, Women's BB, Women's
    Physique and Wellness long memory.
  * In the full model, B beat A by only 0.0007, under the 0.002 bar (and in the run before the bug
    fix it was 0.0026 worse). The model already blends short- and long-memory views.
* **Per-division scaling in stage 2:** worse (1.413 to 1.436 in the logit screen), it overfits
  small Masters fields.
* **The new inputs for Open-age classes:** qualification and head to head did not help the Open-age
  layer.
* **Fixing over-confident favourites, three ways:**
  * a temperature per lineup size: worse, 1.385 to 1.402;
  * blending in uniform probability: it picked no blend in either fold;
  * one fixed shrink of 0.95: it gained 0.0003 on 2018 to 2023 before the bug fix and nothing
    after it, and it lost on 2024 to 2026 (1.281 vs 1.282, calibration error 0.47 vs 0.66 points).
  * Calibration is therefore **not fixed**. Favourites still run a few points high, and some
    middle bins are a little worse than before.

**A bug found on the way.**

* The likelihood code shifted all scores by one global maximum. In one stage-2 fit (the 2025 Masters
  fold) a line-search step underflowed to NaN, and the fit silently returned all-zero coefficients.
  That first made 2024 to 2026 look worse.
* The likelihood now works in the log domain per contest. Stage 2 refuses a fit that ends
  non-finite.
* Three tests guard it: contests far apart, a wide spread inside one contest, and a fit that breaks
  down. Each fails on the old code or with the guard removed.
* The round-1 model's results do not change on the fixed code (2022 and 2023 re-run: log-loss 1.406
  both times), and every round-2 number above comes from the fixed code.

## Round 3: where the model goes wrong

**Method.** I searched for patterns on 2020 to 2023 only: 1,909 divisions and 23,383 athletes, each
year predicted by a model fitted on the years before it. For each kind of athlete and show I
compared expected wins (the sum of their win %) with actual wins. A ratio far from 1 in a large
group is a systematic miss, not an outlier.

**Mostly well calibrated.** Rookies, switchers, Olympia top 6 athletes, defending champions, home
athletes, US athletes and athletes in form all won within about 10% of what the model expected.

**Patterns beyond chance (round-2 model):**

| Pattern | Expected | Actual |
|---|---|---|
| Athletes with no country on record | 43 wins | **0** (a data artefact, see below) |
| Masters athletes from outside the USA | 86 wins | 114 (partly the same artefact) |
| Back after more than 2 years away | 69 wins | 52 |
| Athletes who lost to this lineup before | 126 wins | 100 |
| Model rank 6 to 10 | 155 wins | 133 |
| Model rank 11 and lower | 46 wins | 33 |
| Flat fields (favourite under 20%) | favourite wins 16% | 31% |
| Masters athletes on back-to-back weekends | 180 wins | 208 |
| Masters rookies | 128 wins | 106 |
| Rookies who were favourites at 60%+ | 11.7 wins | 6 |

None of the 195 rookie winners shares a name with an earlier athlete.

**Two leaks, found on the way.**

* **The country on record.** `person.country` is today's value, and it gets filled in mostly for
  athletes who later did well: in 2020 to 2023 the 866 entrants without one won 0 times against 43
  expected.
  * Once "away" joined "home" as an input, the model could tell "unknown" apart. Most of the first
    travel gain came from that, which is also why it vanished on 2024 to 2026 (few unknowns there).
  * An unknown country now counts as the event's own country, and a test checks that such an entrant
    looks exactly like a home athlete. The round-1 "home" input used the same field in a weaker form
    and is fixed by the same change.
* **The finishing order.** Stored lineups are in finishing order.
  * A first "position within the lineup" input broke ties by list order, for example between several
    rookies with the same starting rating, so it leaked the result. It looked like a huge gain
    (Spearman 0.57 to 0.65).
  * The rank-based parts of the analysis broke ties the same way and overstated the back-of-field
    pattern (it was 124 and 25 actual wins before the fix).
  * Ties now go by athlete id, a test checks that inputs do not change when a lineup is reversed, and
    the lineup-position inputs are gone: without the leak they made things worse.

**What was tried, on 2021 to 2023 after removing the leaks (log-loss, lower is better):**

| Idea | Open-age layer | Masters layer | Verdict |
|---|---|---|---|
| Current model | 1.5043 | 1.3366 | |
| Time away (exact years since the last show, stale rating) | **1.5016** | 1.3327 | kept for both |
| Travel (athlete from another country, non-US athlete at a US show) | 1.5036 | 1.3363 | |
| Travel and time away together | 1.5028 | **1.3273** | kept for masters |
| Head to head in the Open-age layer | 1.504 to 1.508 | already in | rejected |
| Position within the lineup, share of the lineup with history | worse | worse | rejected |

* **After the model, two corrections for the probability shape** (each year corrected with
  parameters from earlier years, overall log-loss 1.4300 before):
  * offsets by lineup rank: 1.4329;
  * a temperature per division: 1.4315.

  Both are rejected. The offsets swing too much from year to year.
* **Sixteen stage-2 parameter settings:**
  * what was varied:
    * the boosted ranker: leaves 4, 7, 15; iterations 300, 600, 1000; minimum leaf 40, 100, 200;
      regularisation;
    * logit regularisation 0.3, 1, 3, 10;
    * blend weight 0.4, 0.5, 0.6;
    * training on the last 4 or 6 years or all;
    * a 1- or 2-year calibration holdout;
    * the ranker's label depth: top 5 or top 10.
  * The best gained 0.0014, under the 0.002 bar; the rest were equal or worse. The stage 2 sits on a
    plateau. This sweep ran before the country fix.

**Confirmation on 2024 to 2026, against round 2:**

| | Round 2 | Round 3 |
|---|---|---|
| winner log-loss, all | 1.281 | **1.280** |
| Open-age classes: log-loss | 1.372 | **1.369** |
| masters classes: log-loss | 1.178 | **1.177** |
| favourite wins, all | 53.8% | **54.3%** |
| Spearman | 0.593 | **0.597** |
| masters classes: Spearman | 0.543 | **0.554** |
| calibration error | 0.43 points | **0.31 points** |

The gain is small (log-loss 0.0018) but in the same direction in both layers and on every headline
number. It also includes removing the "home" leak, which on its own would make the scores look a
little worse.

**What is left.**

* Real in total but small, and too unstable from year to year to correct:
  * the back of the field still gets slightly too much win chance;
  * flat-field favourites get too little;
  * strong favourites run a few points high;
  * defending champions and athletes who won their last show do a little worse as strong favourites
    than expected.
* The remaining misses look mostly like judging noise: an outsider having the day of their life, or
  a favourite missing their peak.

## What the data says about bodybuilding

* **Status is real.** Even after the ratings, being top 6 at the latest Olympia adds clearly to the
  chances at the next show; top 3 and the title add more. The literature on judged sports finds
  the same reputation effect.
* **Masters results should not feed the Open rating directly.** Athletes who only ever did
  Masters classes were strongly overrated when Masters and Open shared one rating. The views that
  keep Masters apart became the strongest inputs for Open-age predictions, and a separate stage 2
  for Masters classes improved both Masters and Open accuracy.
* **Within Masters, the age group matters.** An athlete's record in this exact bracket, ratings kept
  per bracket and head-to-head history all add information that the overall rating misses.
* **Ratings carry over between divisions.** An athlete moving from Classic to Open or Bikini to
  Wellness performs in line with their strength in the old division, much better than a rookie.
* **Layoffs:** one year off is neutral to slightly positive (the improvement year); more than two
  years off costs clearly, and the model still overrates returning athletes a little. Back-to-back
  weekends show no consistent effect.
* **In Masters, athletes who travel do a little better than their record.** Once time away is
  known, a masters athlete competing outside their own country (or a non-US athlete at a US show)
  wins slightly more often than the ratings say. In Open-age classes travel adds nothing. An earlier,
  bigger "travel effect" was mostly a data artefact: countries are filled in after the fact for
  notable athletes.
* **No home advantage** shows up once strength is known. **Score margins** (how unanimous a win
  was) add nothing beyond the placings themselves.
* **The biggest shows are the most predictable.** Olympia and tier-2 divisions reach a log-loss
  skill of 0.52 to 0.54 against 0.48 for Open-age classes overall, but upsets at the top still happen.
  Defending Olympia champions who enter keep the title: Women's BB 78%, 212 73%, Fitness 72%,
  Classic 67%, Figure 60%, Men's Open 58%, Men's Physique 54%, **Bikini only 23%**. Returning
  athletes' placings correlate 0.75 (Spearman) from one Olympia to the next.
* **Where it is hardest:**
  * Masters classes: small fields and many first-timers (skill 0.32 vs 0.48 for Open-age);
  * the natural circuit, which has little history;
  * Fit Model, a new division;
  * Wheelchair: tiny sample, and the model is overconfident there.

Per division, Open age, 2024 to 2026: the favourite won 59% in Men's Open, 57% in Women's BB, 63% in
Bikini, 58% in Fitness, and 49 to 56% in Classic, 212, Men's Physique, Figure, Wellness and Women's
Physique.

## Comparison with PRISM

PRISM (Arnold Sports, white paper of 18 July 2026) also fits a field-strength Plackett-Luce model
with recency weighting.

**How it differs from v2:**

* Its published ranking is a hand-weighted points formula, not the model's own prediction.
* It uses one rating model and no second layer, covers six divisions (Open class), and ranks an
  athlete only after three contests.

**On its own published procedure, scored on our data for 2024 to 18 July 2026 only:** six divisions,
"rated" athletes with three or more shows since 2023 or a men's pro win, each show predicted the
day before.

| | PRISM (its paper) | v2 |
|---|---|---|
| pairs ordered correctly | 77% | **82.3%** |
| Spearman | 0.57 | **0.71** |
| Kendall | 0.48 | **0.60** |
| head-to-head log loss | 0.47 | **0.39** |
| favourite wins, actual show winner | 51% | 52.0% |
| favourite wins, best-placed rated athlete | 51% | **61.6%** |

The paper does not say which winner definition it uses, so both are shown. PRISM's code and
contest list are not public to us, so this follows its written rules; it is not a bit-for-bit
replica.

## Olympia 2026, predicted before the show

Stage 2 trained on results up to 2025-12-31, ratings up to the day before the show, win % over the
whole lineup.

| Division | Favourite (win %) | Actual winner (model %, model rank) |
|---|---|---|
| Men's Open | Derek Lunsford (38%) | Nick Walker (5%, 5th) |
| Classic | Ramon Rocha Queiroz (31%) | Niall Darwen (9%, 4th) |
| 212 | **Keone Pearson (46%)** | Keone Pearson |
| Men's Physique | **Ryan Terry (30%)** | Ryan Terry |
| Women's BB | **Andrea Shaw (80%)** | Andrea Shaw |
| Women's Physique | **Natalia Coelho (52%)** | Natalia Coelho |
| Figure | **Lola Montez (29%)** | Lola Montez |
| Wellness | **Eduarda Bezerra (34%)** | Eduarda Bezerra |
| Bikini | Maureen Blanquisco (29%) | Jasmine Gonzalez (6%, 5th) |
| Fitness | **Michelle Mensah (52%)** | Michelle Mensah |
| Fit Model | Gabriela Queiroz (21%) | Shealynn Burnett (16%, 3rd) |
| Wheelchair | James Berger (24%) | Blake Colleton (2%, 11th) |

7 of 12 winners called. In 9 of 12 divisions the whole actual top 3 came from the model's top 5;
in the other three, two of the three did.

## Power ranking

**Method.** For each division, every pro who placed in an Open-age class of the open circuit in the
last 18 months enters a hypothetical Olympia held today. Their stage-2 strength in that context
orders the list. The percentage is their win probability if the top 30 met on that stage.
Inactive athletes fade naturally: old results lose weight and long layoffs cost strength.

As of 2026-10-23 (win % in a hypothetical Olympia today):

| Division | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| Men's Open | Nick Walker (31%) | Derek Lunsford (20%) | Samson Dauda (17%) | Hadi Choopan (10%) | Andrew Jacked (8%) |
| Classic Physique | Niall Darwen (32%) | Mike Sommerfeld (24%) | Ramon Rocha Queiroz (18%) | Josema Muñoz (5%) | Terrence Ruffin (5%) |
| 212 | Keone Pearson (53%) | Shaun Clarida (16%) | Lucas Garcia (14%) | Vitor Porto (4%) | Courage Opara (3%) |
| Men's Physique | Ryan Terry (41%) | Ali Bilal (21%) | Kyron Holden (11%) | Erin Banks (7%) | Brandon Hendrickson (5%) |
| Women's BB | Andrea Shaw (78%) | Ashley Jones (11%) | Angela Yeo (5%) | Leyvina Barros (3%) | Paula Ranta (1%) |
| Women's Physique | Natalia Coelho (57%) | Sarah Villegas (19%) | Zama Benta (15%) | Brooke Walker (2%) | Brittany Herrera (1%) |
| Figure | Lola Montez (45%) | Jessica Reyes Padilla (21%) | Denise Zwinger-Tynek (9%) | Rhea Gayle (8%) | Lena Ramsteiner (5%) |
| Wellness | Eduarda Bezerra (46%) | Isabelle Nunes (22%) | Elisa Alcantara (13%) | Rayane Fogal (5%) | Leonida Ciobu (4%) |
| Bikini | Maureen Blanquisco (26%) | Jasmine Gonzalez (21%) | Aimee Delgado (17%) | Lauralie Chapados (9%) | Ashlyn Little (5%) |
| Fitness | Michelle Mensah (65%) | Jaclyn Baker (23%) | Taylor Learmont (5%) | Anna Fomina (2%) | Allison Kramer (1%) |
| Fit Model | Shealynn Burnett (52%) | Gabriela Queiroz (22%) | Jane Jones (9%) | Liz Trampas (5%) | Carla Andrea Reyes Vargas (3%) |
| Wheelchair | Blake Colleton (33%) | James Berger (26%) | Ephraim Gudgeon (14%) | Kevin Secundino (7%) | Rajesh John (5%) |

Full lists (top 40, with the rank three months earlier, last show and placing) are in
`output/power_rankings.json`. Year-end top 10s since 2018 are in `output/ranking_history.json`.

**Sanity check.** Across Men's Open, Classic, 212, Men's Physique, Women's BB, Figure, Wellness
and Bikini since 2018, the year-end #1 was that year's Olympia champion in 51 of 59 division-years
(every Classic year, Gillon in Figure 2018 to 2024, Shaw in Women's BB since 2020, Mattos, Nunes
and Bezerra in Wellness). The misses:

* Men's Open: Heath ranked above Rhoden (2018) and Curry (2019), Choopan above Dauda (2024);
* 212: Lewis above Elgargni (2019), Elgargni above Clarida (2020);
* Gillon above Gayle (2025);
* Bikini: Teixeira above Pecini (2019), Layug above Dorie (2021).

The year-end lists use the stage-2 layer trained through 2026, so they are a retrospective view,
not what the model would have said live.

## What is not done

* **Nothing is wired into production yet.**
  * v1's batch job (stopped since August 2026) wrote `result.predicted_place`,
    `predicted_win_percentage` and `predicted_confidence`.
  * v2 produces the same fields and more (`output/upcoming.json`).
  * Swapping it in means pointing `batch/batch_predict_and_update.py` at the v2 pipeline, which is
    a write to the database and needs your go.
* **The snapshot is a CSV dump.** A production run should read the same query live (see `dump.py`;
  `dbq.sh` runs it inside a read-only transaction) and retrain weekly (about 7 minutes on this Mac).
* **Rankings exist for the open circuit, Open age only.** There is no Masters (per age group) or
  natural-circuit power ranking yet.
* **Weak spots worth a look:**
  * favourites' win % runs a few points high (rounds 2 and 3 found no fix that held up);
  * the back of the field gets slightly too much win chance (too unstable year to year to correct);
  * Wheelchair is overconfident (22 test divisions);
  * the Masters Olympia is poorly predicted (10 test divisions);
  * natural pros have little history.
* **Ideas not yet tried:**
  * per-judge placings from scorecards (the research rates this the most valuable data upgrade, if
    the cards hold them);
  * amateur results as rookie priors (we hold pro results only);
  * the athlete's age (birthdates are empty);
  * Masters Olympia qualification as its own input;
  * tier lists that change by year (today's tier-2 list is applied to every year).

## Sources

The method choices follow the literature collected in `../research/RESEARCH_NOTES.md` (Plackett-Luce
and reverse PL in Formula One, Powell 2023 "Generalizing Elo", Weng and Lin 2011, Elo-MMR, Benter's
conditional logit, judging bias studies), and the PRISM comparison follows the PRISM white paper
(Brian Powers, Arnold Sports, 18 July 2026). `../research/RESEARCH_NOTES.md` lists all sources.
