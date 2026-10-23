# Predictor v2: research report

As of 2026-10-23. Everything here was computed from a read-only snapshot of the production
database (96,566 pro results, 7,777 divisions, 1947 to 2026). Nothing was written to the database.

## Headline

On every pro division judged from January 2024 to October 2026 (2,610 divisions), predicted
strictly before each show and refitted every January as production would:

| | v2 | uniform guess |
|---|---|---|
| predicted favourite wins | **53.3%** | 15.0% |
| overlap of predicted and actual top 3 | **71.9%** | 41.9% |
| Spearman, predicted vs actual order | **0.59** | 0 |
| winner log-loss (lower is better) | **1.300** | 2.185 |

Against the v1 production predictor on the 259 divisions of 2026 that v1 scored before the show
(both judged on the same athletes):

| | v1 (live) | v2 |
|---|---|---|
| predicted favourite wins | 37.8% | **55.2%** |
| top-3 overlap | 62.5% | **73.1%** |
| Spearman | 0.40 | **0.68** |
| winner log-loss, 79 divisions where v1 stored a win % | 1.503 | **1.476** |

Win percentages are close to honest. Athletes given 10 to 20% won 14.3% of the time (predicted
14.3%), 30 to 40% won 36.9% (predicted 34.8%). **Strong favourites are a little too high:** 70 to 80%
won 71.6% (predicted 74.7%) and 90%+ won 88.7% (predicted 94.3%). The same holds for top-3
chances above 80%. Read a shown 95% as "about 89%".

## How it was tested

* **Walk-forward, no leaks.** Every division is predicted from results dated strictly before it.
  All divisions sharing a date are predicted before any of them is learned from, so an athlete doing
  the Open and the Masters class of one weekend never informs himself. `tests/test_core.py` guards
  this and fails when the guard is sabotaged.
* **The whole lineup counts.** Win % is scored over everyone listed in the division, including
  athletes without a placing (no-shows, unranked; 720 such rows in 2024 to 2026). Field size is the
  lineup size, which is known before the show. A test guards this too.
* **Tuning 2018 to 2023, test 2024 to 2026.** Rating hyperparameters, the stage-2 feature set, the
  boosted model's iteration count, the separate Masters layer and the 50/50 blend were chosen on
  2018 to 2023. The last two were re-checked on 2022 and 2023 alone (each year fitted on the years
  before): separate Masters layer 1.405 vs 1.421 shared; blend weight 0.3 and 0.5 tie at 1.405, 0.7
  is worse (1.413). The stage-2 layer is refitted every January on everything before that year, and
  its temperature is fitted on the last year of its training rows by a model that did not train on
  them.
* **Caveats.** I looked at test numbers while developing (they were printed next to the tuning
  numbers), so they are not a perfectly blind holdout. The v1 comparison is unaffected. The
  coordinate search for the main rating stopped in its second round once gains fell below 0.002;
  `bbpredict/final.py` uses the configuration reached by then.
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
| + boosted ranker, separate Open and Masters layers, yearly refit (final) | **1.300** | **53.3%** | **71.9%** | **0.59** |

What did not help: Weng-Lin online updates (worse than plain Elo here), judge-score margins as
features, home-country advantage, damping later picks when simulating placings (Benter style).

## The final model

**Stage 1, six rating views per athlete and division.** Ratings are kept per (athlete, division
type, open or natural circuit).

* The main one is a **time-weighted Plackett-Luce model refitted by MAP on every show date**. Each
  earlier result counts with weight `exp(-age / 730 days)` and every athlete has a normal prior.
  Because it refits, a result is reinterpreted when we learn more about the opponents.
* It is fitted **bottom up ("reverse" PL)**: last place is chosen first. This beat the usual
  top-down order clearly (judges agree most about who is out of contention). The "not placed"
  tie block at 16th is treated as unordered.
* The other five views are the same PL forward, a short memory version (300 days), a long memory
  version (1,500 days), a PL that keeps **Masters results apart** from the Open rating, and a
  multiplayer Elo (also Masters apart).

**How shows are weighted (the Olympia vs Arnold vs local pro question).**

* The main lever is field strength. Every result is judged against the ratings of the people on
  stage, so beating an Olympia or Arnold field moves a rating far more than winning a thin local
  show.
* On top of that, the Plackett-Luce views count an Olympia 1.5x and a Masters Olympia 2/3x.
* Tier-2 shows (Arnold, NY Pro, Pittsburgh, Romania, as tier 2 in our qualification rulesets) stay
  at 1.0 there, because raising them did not improve predictions. The Elo view counts them 1.5x,
  and the stage-2 layer knows each show's tier.
* Dubai Pro is tier 3 in our rulesets, so it is tier 3 here too.

**Stage 2, a ranking layer over ratings plus context** (Benter style, as in horse racing): a
conditional logit and a LightGBM lambdarank model, blended 50/50, fitted separately for Open-age
and Masters classes. Inputs besides the six ratings:

* latest Olympia placing (in, top 6, top 3, champion) and ever top 3 at an Olympia;
* last edition of this same show (won it, 1/place);
* pro wins, shows in this division, shows in the last year, years since pro debut;
* layoff since the last show (back to back, short, 1 to 2 years, more than 2 years);
* rookie, or switcher from another division (with their rating there);
* masters-only athlete, recent form and trend, home country;
* the show tier, division and lineup size (boosted model only).

**Probabilities.** Win % is the softmax of the stage-2 strength. P(2nd), P(3rd), top 3, top 5 and
expected placing come from 20,000 Plackett-Luce simulations (strength plus Gumbel noise).

## What the data says about bodybuilding

* **Status is real.** Even after the ratings, being top 6 at the latest Olympia adds clearly to the
  chances at the next show; top 3 and the title add more. The literature on judged sports finds
  the same reputation effect.
* **Masters results should not feed the Open rating directly.** Athletes who only ever did
  Masters classes were strongly overrated when Masters and Open shared one rating. The views that
  keep Masters apart became the strongest inputs for Open-age predictions, and a separate stage 2
  for Masters classes improved both Masters and Open accuracy. All Masters age groups (35+ to 70+)
  share that one Masters layer.
* **Ratings carry over between divisions.** An athlete moving from Classic to Open or Bikini to
  Wellness performs in line with their strength in the old division, much better than a rookie.
* **Layoffs:** one year off is neutral to slightly positive (the improvement year); more than two
  years off costs clearly. Back-to-back weekends show no consistent effect.
* **No home advantage** shows up once strength is known. **Score margins** (how unanimous a win
  was) add nothing beyond the placings themselves.
* **The biggest shows are the most predictable.** Olympia and tier-2 divisions reach a log-loss
  skill of 0.52 to 0.54 against 0.48 for Open-age classes overall, but upsets at the top still happen. Defending Olympia champions who enter keep the title:
  Women's BB 78%, 212 73%, Fitness 72%, Classic 67%, Figure 60%, Men's Open 58%, Men's
  Physique 54%, **Bikini only 23%**. Returning athletes' placings correlate 0.75 (Spearman) from one
  Olympia to the next.
* **Where it is hardest:**
  * Masters classes: small fields and many first-timers (skill 0.30 vs 0.48 for Open-age);
  * the natural circuit, which has little history;
  * Fit Model, a new division;
  * Wheelchair: tiny sample, and the model is overconfident there.

Per division, Open age, 2024 to 2026: the favourite won 59% in Men's Open, 62% in Bikini, 61% in
Fitness, and 49 to 54% in Classic, 212, Men's Physique, Figure, Wellness, Women's BB and Women's
Physique.

## Olympia 2026, predicted before the show

Stage 2 trained on results up to 2025-12-31, ratings up to the day before the show, win % over the
whole lineup.

| Division | Favourite (win %) | Actual winner (model %, model rank) |
|---|---|---|
| Men's Open | Derek Lunsford (39%) | Nick Walker (6%, 5th) |
| Classic | Ramon Rocha Queiroz (32%) | Niall Darwen (10%, 4th) |
| 212 | **Keone Pearson (45%)** | Keone Pearson |
| Men's Physique | **Ryan Terry (29%)** | Ryan Terry |
| Women's BB | **Andrea Shaw (80%)** | Andrea Shaw |
| Women's Physique | **Natalia Coelho (51%)** | Natalia Coelho |
| Figure | **Lola Montez (28%)** | Lola Montez |
| Wellness | **Eduarda Bezerra (34%)** | Eduarda Bezerra |
| Bikini | Maureen Blanquisco (29%) | Jasmine Gonzalez (6%, 5th) |
| Fitness | **Michelle Mensah (51%)** | Michelle Mensah |
| Fit Model | Gabriela Queiroz (23%) | Shealynn Burnett (18%, 3rd) |
| Wheelchair | James Berger (23%) | Blake Colleton (2%, 11th) |

7 of 12 winners called. In 8 of 12 divisions the whole actual top 3 came from the model's top 5;
in the other four, two of the three did.

## Power ranking

**Method.** For each division, every pro who placed in an Open-age class of the open circuit in the
last 18 months enters a hypothetical Olympia held today. Their stage-2 strength in that context
orders the list. The percentage is their win probability if the top 30 met on that stage.
Inactive athletes fade naturally: old results lose weight and long layoffs cost strength.

As of 2026-10-23 (win % in a hypothetical Olympia today):

| Division | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| Men's Open | Nick Walker (30%) | Derek Lunsford (23%) | Samson Dauda (17%) | Hadi Choopan (9%) | Andrew Jacked (7%) |
| Classic Physique | Niall Darwen (31%) | Mike Sommerfeld (24%) | Ramon Rocha Queiroz (19%) | Terrence Ruffin (5%) | Wesley Vissers (4%) |
| 212 | Keone Pearson (52%) | Shaun Clarida (17%) | Lucas Garcia (14%) | Vitor Porto (4%) | Courage Opara (3%) |
| Men's Physique | Ryan Terry (40%) | Ali Bilal (20%) | Kyron Holden (12%) | Erin Banks (8%) | Brandon Hendrickson (5%) |
| Women's BB | Andrea Shaw (76%) | Ashley Jones (12%) | Angela Yeo (5%) | Leyvina Barros (3%) | Paula Ranta (1%) |
| Women's Physique | Natalia Coelho (58%) | Sarah Villegas (20%) | Zama Benta (14%) | Brooke Walker (2%) | Brittany Herrera (1%) |
| Figure | Lola Montez (43%) | Jessica Reyes Padilla (21%) | Denise Zwinger-Tynek (11%) | Rhea Gayle (8%) | Lena Ramsteiner (5%) |
| Wellness | Eduarda Bezerra (44%) | Isabelle Nunes (23%) | Elisa Alcantara (14%) | Rayane Fogal (5%) | Leonida Ciobu (4%) |
| Bikini | Maureen Blanquisco (26%) | Jasmine Gonzalez (20%) | Aimee Delgado (18%) | Lauralie Chapados (9%) | Ashlyn Little (5%) |
| Fitness | Michelle Mensah (62%) | Jaclyn Baker (27%) | Taylor Learmont (4%) | Anna Fomina (3%) | Allison Kramer (1%) |
| Fit Model | Shealynn Burnett (50%) | Gabriela Queiroz (22%) | Jane Jones (10%) | Liz Trampas (6%) | Carla Andrea Reyes Vargas (3%) |
| Wheelchair | Blake Colleton (32%) | James Berger (27%) | Ephraim Gudgeon (14%) | Kevin Secundino (7%) | Rajesh John (5%) |

Full lists (top 40, with the rank three months earlier, last show and placing) are in
`output/power_rankings.json`. Year-end top 10s since 2018 are in `output/ranking_history.json`.

**Sanity check.** Across Men's Open, Classic, 212, Men's Physique, Women's BB, Figure, Wellness
and Bikini since 2018, the year-end #1 was that year's Olympia champion in 49 of 59 division-years
(every Classic year, Gillon in Figure 2018 to 2024, Shaw in Women's BB since 2020, Mattos, Nunes
and Bezerra in Wellness). The misses:

* Men's Open: Heath ranked above Rhoden (2018) and Curry (2019), Choopan above Lunsford (2023) and
  Dauda (2024);
* 212: Lewis above Elgargni (2019), Elgargni above Clarida (2020);
* Hendrickson above Terry (2023), Gillon above Gayle (2025);
* Bikini: Teixeira above Pecini (2019), Layug above Dorie (2021).

The year-end lists use the stage-2 layer trained through 2026, so they are a retrospective view,
not what the model would have said live.

## What is not done

* **Nothing is wired into production yet.** v1's batch job still writes `result.predicted_place`,
  `predicted_win_percentage` and `predicted_confidence`. v2 produces the same fields and more
  (`output/upcoming.json`). Swapping it in means pointing `batch/batch_predict_and_update.py` at
  the v2 pipeline, which is a write to the database and needs your go.
* **The snapshot is a CSV dump.** A production run should read the same query live (see `dump.py`;
  `dbq.sh` runs it inside a read-only transaction) and retrain weekly (about 5 minutes on this Mac).
* **Rankings exist for the open circuit, Open age only.** There is no Masters (per age group) or
  natural-circuit power ranking yet, and the Masters layer does not tell 40+ from 60+.
* **Weak spots worth a look:** favourites' win % mostly runs 3 to 5 points high, Wheelchair is
  overconfident (22 test divisions), the Masters Olympia is poorly predicted (10 test divisions),
  natural pros have little history.
* **Ideas not yet tried:**
  * per-judge placings from scorecards (the research rates this the most valuable data upgrade, if
    the cards hold them);
  * amateur results as rookie priors (we hold pro results only);
  * the athlete's age (birthdates are empty);
  * a per-division temperature for small divisions;
  * tier lists that change by year (today's tier-2 list is applied to every year).

## Sources

The method choices follow the literature collected in `../research/RESEARCH_NOTES.md` (Plackett-Luce
and reverse PL in Formula One, Powell 2023 "Generalizing Elo", Weng and Lin 2011, Elo-MMR, Benter's
conditional logit, judging bias studies). That file lists all sources.
