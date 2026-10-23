# Research notes: rating, ranking and prediction for IFBB pro contests

Compiled 2026-10-23 for the Predictor project (pre-show placing prediction plus a per-division power ranking). Web research only: no code was written and no database was touched.

Confidence tags used below:

- (verified): read in a primary or official source during this research.
- (secondary): reported by trade press, a summary page or an abstract only.
- (memory): from prior knowledge, not re-checked today. Treat as likely but confirm before relying on it.
- (unsure): sources conflict or evidence is thin.

---

## 0. Short summary

1. For fields of N with one ranked outcome, the principled core is a dynamic Plackett-Luce (rank-ordered logit) or a Thurstonian (Gaussian) equivalent such as TrueSkill or Elo-MMR. Naive pairwise Elo over a finish order over-counts one bad day N-1 times.
2. Published head-to-head comparisons show that good multiplayer systems differ very little in accuracy (tenths of a percentage point). Dynamics, priors, event weighting and extra features matter more than the core algorithm.
3. Time weighting (down-weighting old results) improved Plackett-Luce forecasts in Formula One. Some variants were competitive with bookmaker odds.
4. Modelling a ranking from the bottom up ("reverse" or "elimination" Plackett-Luce) fitted or predicted better than the usual top-down version in two Formula One studies. Judged sports show more judge consensus at the top of the field. Both forward and reverse PL are worth testing.
5. Judging-bias literature (figure skating, gymnastics, baseball umpires) consistently finds reputation, status and nationality effects. In bodybuilding terms, incumbency and "name" effects are plausible and should be measured as features, not assumed.
6. IFBB Pro League scoring is ordinal: each judge ranks, the extreme placings are dropped, and the rest are summed. Only the top 15 get distinct places; everyone below ties for 16th. The sum of placings is the only margin information.
7. From the 2026 season, Classic Physique, Men's Physique, Bikini and Wellness use an Olympia points race. Five named shows (Arnold Classic USA, Pittsburgh, NY Pro, Romania Muscle Fest, Dubai) are top tier. All other divisions still qualify by winning any open pro show or placing top 3 at the Olympia.
8. Men's Open has had no repeat Mr. Olympia for five straight years (2022 to 2026). Older eras and other divisions show long reigns, so incumbency strength is division and era specific.
9. No peer-reviewed predictive model for pro bodybuilding was found. The public "power rankings" are editorial. One student project (Erdos Institute) used plain Elo.

---

## Part A. Methods for multi-competitor ranked outcomes

### A1. Framing

- One contest = one ranking of N athletes. In our case that is show x division x class/age group x level x natural/open. Typical pro fields are about 5 to 20 athletes. Some Olympia divisions had about 60 qualifiers in 2025 under the any-win qualification era (secondary, BarBend); actual starters are fewer.
- The data are partial rankings with a tied tail. Only the top 15 are scored, the rest are usually tied 16th, and Bikini, Wellness and Fit Model have "top 10 finalists" (secondary, BarBend; verified for the top 10, IFBB Pro rules page).
- Random-utility view (the common ancestor of every method below): athlete i produces a show-day performance X_i = s_i + e_i, and the observed order is X sorted.
  - If e is Gumbel, the order follows Plackett-Luce exactly. Yellott (1977), cited by Powell, shows the shifted Gumbel is the only family that gives PL.
  - If e is Normal, you get Thurstone-Mosteller ordering. TrueSkill, Elo-MMchi and Henery's horse-racing model all work this way.
  - Logistic pairwise comparison gives Bradley-Terry, and online it gives Elo.
  - For the winner probability the choice of noise barely matters. It matters for place probabilities of favourites and longshots (see A6).
- Two different jobs:
  - (a) Forecasting needs filtering: only information available before the show.
  - (b) A "power ranking now" may use smoothing over the whole history up to today. Smoothing also refines opponents' past ratings, which is legitimate as long as it never uses results after today.

### A2. Method families

**Pairwise Elo over a finish order.** Split an N-field into N(N-1)/2 head-to-head games and usually divide K by N-1. It is simple, it was used by the Erdos bodybuilding project, and Moore et al. (2018) did this for races (as cited by Powell).

- Problem: the pairwise results are dependent. A>B and B>C already imply A>C, so one bad show is punished once per opponent. Powell calls this over-counting and argues it is unprincipled (verified, Powell 2023).
- It works acceptably once K is tuned, but it carries no uncertainty, no principled inactivity handling and no rookie prior.

**Bradley-Terry on pairwise decompositions.**

- Batch maximum likelihood via MM algorithms (Hunter 2004) (memory).
- Variants used as baselines in the Elo-MMR repository: "BAR BT-all-pairs" and "Glicko all-pairs" (verified, repo README).
- Weng and Lin (JMLR 2011) give closed-form online Bayesian updates for BT "full pair" (all pairs), BT "partial pair" (adjacent ranks only, which reduces over-counting), Thurstone-Mosteller and Plackett-Luce. Their reported accuracy is competitive with TrueSkill, with much shorter code and run time (secondary, abstract).

**Plackett-Luce (PL, rank-ordered or exploded logit).**

- Likelihood: P(order) = product over places k of exp(s_(k)) / sum over the athletes not yet placed of exp(s_j). The winner probability is a softmax of strengths.
- Batch fitting:
  - The R package PlackettLuce (Turner, van Etten, Firth, Kosmidis 2020) handles ties of any order and partial rankings.
  - It adds "pseudo-rankings" against a hypothetical item so that weakly connected networks get finite estimates. This also shrinks estimates toward average.
  - It supports a multivariate normal prior on log-worths (MAP estimation) (secondary, CRAN and abstract).
- Time-weighted and truncated PL (Henderson and Kirrane, Bayesian Analysis 2018, Formula One 2010 to 2013):
  - Down-weighting past results improved forecasts.
  - Some variants were competitive with forecasts implied by bookmaker odds (secondary, abstract).
  - Truncated PL uses only the top k places, because lower places carry more noise.
  - I could not read the full results table, so which truncation level won is (unsure).
- Reverse PL (build the order from last place upward):
  - Johnson, Henderson and Boys (Extended PL, arXiv 2020) report that RPL-derived models fit F1 better than standard PL (secondary).
  - Powell (JQAS 2023, "why endurance is better than speed") builds an online Elo that is exactly an online PL estimator using elimination rounds. In F1 it beats the selection-round ("speed") version in a betting simulation against each other (verified).
  - His argument: if failures are random, an early exit says little, but outlasting everyone is strong evidence. The model is therefore more sensitive to the best performances.
- Powell's online update (verified, Powell 2023):
  - Elimination probability: P(i eliminated in round) = exp(-R_i) / sum_j exp(-R_j). Update: R_i += k_i * [I(i survives) - P(i survives)].
  - Precision update: 1/k_i += P(survive) * (1 - P(survive)).
  - Between events (h time units apart):
    - Variance relaxes toward the no-information level: k_i += (1 - phi^(2h)) * (k_inf - k_i).
    - The mean reverts toward the population mean: R_i = phi^h * R_i.
  - He suggests a starting K of about 0.25 and tuning from there.
  - This is a complete, cheap recipe for an online PL with time decay and rookie variance.
- Glickman and Hennessy (JQAS 2015), "stochastic rank ordered logit" (secondary, abstract):
  - A Bayesian state-space PL in which each ability follows a Gaussian random walk.
  - Ties are allowed (for example DNF athletes tied last).
  - Inference is by MCMC, plus a filtering algorithm for ongoing updates.
  - Applied to 268 women's Alpine downhill skiers, 2002 to 2013.
  - This is the closest published analogue to an individual sport with a ranked field and athletes coming and going.

**Codeforces and TopCoder ratings.**

- Built for huge programming contests.
- They compute an expected rank from pairwise Elo probabilities and move each rating toward a performance rating.
- TopCoder's volatility term can be exploited by deliberately underperforming (secondary, Ebtekar; DMOJ migration note).
- They carry no clean uncertainty model. Not recommended for us beyond use as a baseline.

**Elo-MMR (Ebtekar and Liu, WWW 2021)** (verified, paper):

- Each contest first estimates a per-round performance from all pairwise wins and losses. The rating is then a robust weighted average of past performances: a logistic factor per past round, combined with a Gaussian prior. A Gaussian variant is called Elo-MMchi.
- Uncertainty grows by gamma^2 per round through a "pseudodiffusion" with a parameter rho.
- The rating is monotone in each past rank, so it is incentive compatible.
- The change caused by one extreme result is bounded. That robustness is attractive for one disastrous show-day conditioning miss.
- The reference implementation supports a per-contest weight field and ties via low/high rank (verified, README).

**Glicko and Glicko-2** (memory plus secondary):

- A Gaussian rating with a rating deviation (RD). RD grows while inactive: RD_new = min(sqrt(RD^2 + c^2 * t), RD_max).
- Glicko-2 adds a volatility sigma, controlled by tau; Glickman recommends 0.3 to 1.2 (secondary).
- Designed for pairwise games batched into rating periods. Multiplayer use needs all-pairs decomposition.
- The volatility can be gamed (Ebtekar).

**TrueSkill** (memory for defaults):

- Factor graph with skill N(mu, sigma^2), performance N(s, beta^2), a dynamics variance tau added per game, and a draw margin.
- Handles multi-player rankings natively through expectation propagation.
- Typical defaults: mu 25, sigma 25/3, beta sigma/2, tau sigma/100. Leaderboards use the conservative score mu - 3 sigma.
- In very large fields with many distinct ranks it lost clearly to Elo-MMR, Codeforces and TopCoder (see A3).
- TrueSkill 2 (Minka, Cleven, Zaykov 2018) adds covariates such as experience and quitting tendency. Halo 5 outcome accuracy rose from 52% to 68% (secondary). The lesson: features beat the core algorithm.

**TrueSkill Through Time (TTT)** (secondary):

- Dangauthier, Herbrich, Minka, Graepel (2007); software paper by Landfried and Mocskos (Journal of Statistical Software 2025), with Julia, Python and R packages.
- Uses the whole history in one Bayesian network: smoothing instead of forward-only filtering.
- Gives reliable early-career estimates and historical comparability.
- For walk-forward backtests it must be refit at each cut-off date. Otherwise it leaks future results into past ratings.

**OpenSkill / Weng-Lin** (secondary):

- Open-source implementation of the Weng-Lin approximations (JOSS 2024, arXiv 2401.05451). Plackett-Luce is the recommended model, and time decay is supported.
- Its own Overwatch benchmark shows TrueSkill slightly more accurate in one split (73.3% vs 70.5%) but slower.

**Whole History Rating (WHR, Coulom 2008)** (secondary):

- A dynamic Bradley-Terry model with a Wiener-process prior on each player's strength, solved as an exact MAP over all history with Newton's method.
- Reported to predict better than Elo, Glicko, TrueSkill and decayed-history methods on Go server data.
- Pairwise only: N-fields need decomposition. Like TTT, it is a smoother.

Summary table (my synthesis):

| Method | Native N-field | Uncertainty | Time dynamics | Ties | Cost | Fit for us |
|---|---|---|---|---|---|---|
| Pairwise Elo (K/(N-1)) | no (decomposed) | no | ad hoc | yes | trivial | baseline only |
| Batch PL with time weights + prior | yes | via prior/Hessian | exponential weights | yes (extension) | refit per date | strong, transparent |
| Online PL (Powell, Weng-Lin, OpenSkill) | yes | yes | variance growth, mean reversion | partly | trivial | strong default |
| Glickman-Hennessy state-space PL | yes | full posterior | random walk | yes | MCMC or filter | gold standard, heavier |
| TrueSkill | yes | yes | tau per game | draw margin | low | good; weaker in huge fields |
| TTT / WHR | TTT yes, WHR no | yes | Wiener / random walk | yes | moderate | best for "ranking now" and history |
| Elo-MMR | yes | yes | diffusion | yes | low | strong, robust to outlier shows |
| Glicko-2 all-pairs | no | yes | c^2 t | yes | low | not preferred |

### A3. Published evidence on predictive accuracy

- **Elo-MMR paper** (verified, Table 2). Metric pairs are % of pairs ordered correctly / mean rank deviation. Hyperparameters were tuned on the first 10% of each dataset and tested on the remaining 90%; players with fewer than 5 contests were excluded.

  | Dataset | Codeforces | TopCoder | TrueSkill | Elo-MMchi | Elo-MMR |
  |---|---|---|---|---|---|
  | Codeforces | 78.3 / 14.9 | 78.5 / 15.1 | 61.7 / 25.4 | 78.5 / 14.8 | 78.6 / 14.7 |
  | TopCoder | 72.6 / 18.5 | 72.3 / 18.7 | 68.7 / 20.9 | 73.0 / 18.3 | 73.1 / 18.2 |
  | Reddit (fields of about 20) | 61.5 / 27.3 | 61.4 / 27.4 | 61.5 / 27.2 | 61.6 / 27.3 | 61.6 / 27.3 |

  Takeaway: among sensible systems the gaps are 0.1 to 0.5 points. On the small-field Reddit data all systems are effectively tied.
- **Weng-Lin vs TrueSkill:** competitive accuracy, faster (secondary).
- **WHR vs Elo, Glicko, TrueSkill:** WHR better on Go data (secondary, author's claim).
- **TrueSkill 2 vs TrueSkill:** 68% vs 52% outcome accuracy from adding covariates (secondary).
- **Tennis, Kovalchik (JQAS 2016):** 11 published models on 2,395 ATP matches (2014).
  - FiveThirtyEight's Elo and ranking-based regressions were the most accurate history-based methods, about 75% for top players.
  - The bookmaker consensus remained the benchmark to beat.
  - All models were 10 to 20 points worse on lower-ranked players (secondary).
- **Formula One, Henderson and Kirrane (2018):** time down-weighting helps, and some PL variants are about as good as bookmaker-implied probabilities (secondary).
- **Formula One, Powell (2023):** elimination-style (reverse) online PL beats selection-style (verified).
- **Golf, Data Golf** (verified, methodology page):
  - Skill is an exponentially time-weighted average of field-adjusted strokes-gained. About 70% of the weight falls on the most recent 50 rounds for a full-schedule player.
  - A time-weighted and a sequence-weighted average are combined.
  - Decay rates are chosen by cross-validation, with a "medium-term" result: rounds from 2 to 3 years ago get non-zero weight, but decay is fairly quick.
  - Win probabilities come from simulation using player-specific variance, shrunk toward the tour average.
  - Their note: models that look equally good overall still disagree on individual predictions. About half of your disagreements with the market are probably your model being wrong.
- **Horse racing:**
  - The Harville formula (equivalent to PL with exponential running times) is systematically biased for place probabilities of the strongest and weakest runners.
  - Henery's normal-times model fits better, but has no closed form (secondary, Hong Kong working papers; Ali 1998).
  - Ali (1998): all models overestimate second and third place probability for horses with a high chance of placing, and underestimate it for low-chance horses (secondary).

Bottom line: choose a principled N-field model, then spend effort on dynamics, priors, event effects and covariates, judged by walk-forward proper scores. Expect markets or expert consensus to remain hard to beat at the Olympia.

### A4. Judged sports: what the bias literature says

- **Nationalistic bias:**
  - Zitzewitz (2006): figure skating and ski jumping judges favour their own country. Skating showed vote trading. Trimming the high and low marks made collusion easier, not harder (secondary).
  - Heiniger and Mercier: FIG gymnastics judges show considerable national bias in aerobic, artistic and rhythmic gymnastics, none in acrobatics or trampoline. The best and worst judges differ greatly in accuracy and bias (secondary).
- **Heteroscedastic judging error** (Heiniger and Mercier, 8 sports including diving, figure skating, gymnastics and ski jumping): judging error follows roughly a quadratic in performance level. Judges agree more about the best athletes, in every sport except dressage (verified, arXiv abstract). Implication: our top placings are probably less noisy than mid-field placings.
- **Reputation and status bias:**
  - Findlay and Ste-Marie (Ottawa thesis): the same skating performances got better ordinals when judges knew the skater. The difference showed in technical and artistic marks, not in error deductions (secondary).
  - Plessner (1999): expectation bias from team order position in gymnastics (secondary).
  - Kim and King (Management Science 2014): MLB umpires expand the strike zone for high-status pitchers. The press release reports about 16% more wrong strike calls for a five-time All-Star (secondary).
  - These are the academic analogues of bodybuilding's "politics" and "you have to beat the champ". Expect autocorrelation in placings beyond true physique change.
- **Bodybuilding specifically:**
  - Gaines (Virginia Tech PhD thesis 2001, 29 men and 9 women):
    - Judges' rankings tracked body fat (women) and muscle proportionality (men).
    - Elite and trained judges were more reliable than untrained judges.
    - Prediction models were explicitly left untested (verified, abstract).
  - I found no peer-reviewed study of bias or predictability in IFBB Pro placings.

### A5. Handling the specific modelling issues

**Event importance and strength of field.**

- In any Bayesian ranked model (PL, TrueSkill, Elo-MMR), opponent strength is handled automatically: beating strong athletes moves you more. A separate "strength of field" multiplier would double count.
- "Importance" in a different sense is still real: whether athletes peak for the show, panel size and quality, and field depth. Options:
  - **Per-contest weight on the likelihood** (Elo-MMR supports this). This is equivalent to a smaller K for less informative events.
  - **Per-tier performance noise beta_tier.** Larger noise means a less informative event. It is learned by maximum likelihood, not hand-set.
  - **Hand-set K by tier.** Precedent: World Football Elo uses K = 60 for World Cup finals, 50 for continental finals and 40 for qualifiers. Values are partly disputed across sources (secondary).
- Recommendation: estimate the noise or weight per tier from data. Tiers: Olympia, Arnold Classic USA and the other named top shows, other open pro shows, masters pro, natural pro, amateur national, amateur regional.
- A displayed "field strength" index can still be computed from entrants' pre-show ratings, for example the mean of the top 5 ratings or the expected rating of the winner. Golf's OWGR-style strength-of-field weighting is an achievement-ranking device, not a predictive one.

**Time decay, inactivity and uncertainty growth.** Four mechanisms, combinable:

1. Variance growth while inactive: Glicko c^2 t, TrueSkill tau, Elo-MMR diffusion, Powell's relaxation toward k_inf.
2. Mean reversion toward the population over long gaps (Powell's phi^h).
3. Exponential down-weighting of old contests in a batch fit (Henderson and Kirrane; Data Golf).
4. Explicit covariates: days since last show, number of contests in the last 12 months, age.

Notes:

- Data Golf uses days since the last round as a predictor. They note that players returning from long layoffs should have estimates that respond more strongly to early results, which is exactly what variance growth does.
- MMA ranking sites apply inactivity penalties after about 18 months. Those are ranking policies, not predictive models (secondary, Combat Edge, Kickboxhub).
- Bodybuilding twist: layoffs are often strategic ("taking a year off to improve"). The Erdos project noted that plain Elo does not reward long off-seasons although these "typically benefit competitors" (secondary).
- So the mean effect of a layoff may be positive for 1 year and negative for 3 years or more. Model it as a learned function in a second stage rather than forcing pure decay.

**Rookies and new entrants.**

- Use a prior mean below the division average and a large prior variance.
- Data Golf regresses low-data players toward about -2 strokes-gained per round, and the baseline differs by tour. Variance for low-data players is borrowed from similar players (verified).
- For us, set the prior mean by pathway: how the pro card was won (NPC national overall vs class win, NPC Worldwide international qualifier, natural qualifier) and the athlete's amateur rating if amateur data are present.
- TTT-style smoothing fixes early-career estimates retrospectively, which helps the history pages.

**Home country and judging panel effect.**

- No bodybuilding evidence was found either way. Panel nationality is not published in the sources I found.
- Given strong effects in skating, gymnastics and breaking, add an "athlete country equals show country" indicator and test it, especially for non-US pro shows.

**Margin information (score points vs pure order).**

- In sports with scores, margin-of-victory versions of Elo help: Kovalchik (IJF 2020); G-Elo (Szczecinski); FiveThirtyEight's multiplier; World Football Elo's goal-difference multiplier of about 1.5x for 2 goals and 1.75x for 3 (secondary).
- In bodybuilding the only margin is the scorecard: the sum of counted judges' placings. Normalised as average placing per counted judge per round, it says whether a win was unanimous (average 1.0) or split (for example 1.4).
- Ways to use it:
  - (a) Weight each PL update by decisiveness.
  - (b) Fit an ordinal or Thurstonian model to the average placings.
  - (c) Best, if per-judge placings exist in the scorecards: treat each counted judge's ranking as a separate noisy ranking with a shared show-day performance. This gives two noise levels, athlete show-day noise and judge noise, and separates "athlete was off" from "the panel was split".

### A6. From ratings to probabilities for a field of N

- **Winner probability, PL closed form:** p_i = exp(s_i / T) / sum_j exp(s_j / T), with a temperature T fit on held-out data. This is the same idea as temperature scaling of classifier outputs.
- **Full placing distribution:** simulate.
  - Gumbel-max trick: add independent Gumbel noise to s_i and sort. This samples exactly from PL. Use Normal noise for a Thurstone model.
  - To include rating uncertainty, first draw s_i from the posterior N(mu_i, sigma_i^2), then add performance noise. That is the TrueSkill-style predictive and the Data Golf simulation approach.
  - Outputs: P(win), P(top 3), P(top 5), expected placing, and the median placing interval.
- **Place-probability bias:** PL top-down (Harville) tends to overstate favourites' chances of second and third (horse racing evidence, A3).
  - Benter's fix raises the remaining runners' probabilities to exponents below 1 for second and third place. My recollection is about 0.81 and 0.65 for Hong Kong (memory, unsure).
  - Alternatively use a Normal (Thurstone) noise model, or the reverse PL, and compare by log score on full orders.
- **Calibration:** check reliability diagrams for win probabilities (deciles) and for P(top 3). Calibrate T per field-size band if needed, because small fields often look overconfident.

### A7. Evaluation

- **Proper scoring rules** (Gneiting and Raftery 2007) reward honest probabilities. Primary choices:
  - Log score (log-loss) of the actual winner: multiclass log-loss over the field.
  - Full-ranking or top-k log-likelihood under the model, the PL likelihood of the observed top 5. This uses more of each contest than the winner alone.
  - Multiclass Brier score as a secondary, bounded check.
- **Ranked Probability Score (RPS):** Wheatcroft (JQAS 2021) shows the "distance sensitivity" argument for RPS adds nothing, and the local log (ignorance) score outperformed RPS and Brier in simulations. I would not make RPS the headline metric for expected placing.
- **Point and ranking metrics** (not proper, but readable): top-1 accuracy, top-3 hit rate, Spearman or Kendall between expected and actual placing, NDCG@5, MAE of placing.
- **Protocol:**
  - Walk-forward by date: the rating state uses only results before the show date.
  - Tune hyperparameters on an early slice and report on later years (as in the Elo-MMR paper).
  - Report separately for the Olympia, top-tier shows, small pro shows and masters, and by field size. Prediction difficulty varies hugely, and winner log-loss depends on N.
  - Normalise by comparing to a uniform baseline (log N) and to simple heuristics: last placing, most pro wins, previous Olympia placing.
- **External benchmark:**
  - No sportsbook lines for the 2026 Mr. Olympia were found.
  - Expert prediction round-ups (Fitness Volt, Generation Iron) are a usable "public" baseline. In 2026 most experts picked Lunsford, Jacked or Dauda, and Walker won (secondary). This shows how uncertain the top of the Open division is.

### A8. Benter-style conditional logit (horse racing)

- Bolton and Chapman (1986) and Benter (1994) model each race as a multinomial or conditional logit over the field. Each horse has a linear score from many features, and the win probability is the softmax within the race (secondary).
- Benter's second stage combines the fundamental model with the public (market) probabilities: c_i = exp(alpha * ln f_i + beta * ln pi_i) / sum_j exp(alpha * ln f_j + beta * ln pi_j) (secondary).
  - Reported out-of-sample pseudo R^2 on 2,313 Hong Kong races: public odds .1237; fundamental model alone .1016; combined .1327 (secondary).
  - The fundamental model alone was worse than the market, but the combination beat both.
- Direct translation for us:
  - Stage 1: the rating system gives mu and sigma per athlete.
  - Stage 2: a conditional (or rank-ordered) logit on each contest field. Candidate features: rating mean and sigma; placing percentile at the last show; days since the last show; shows in the last 12 months; defending champion of this show or of the Olympia; last Olympia placing; pro wins to date; age; division switch in the last 2 years; same-country-as-show; back-to-back weekend; rookie pathway.
  - Optional stage 3: blend with expert consensus where it exists (Olympia, Arnold).
- Fit the stage-2 logit by maximum likelihood on walk-forward data. A rank-ordered version uses the full top 5 with lower places down-weighted, which is Benter's discounting logic.

---

## Part B. Bodybuilding domain specifics

### B1. How IFBB Pro League judging and scoring work

**Panel and scoring.**

- Panels always have an odd number of judges: 5 at small shows, 7 at others, 9 at the Arnold-level shows (secondary, BarBend).
- Steve Weinberger says panels can be 5, 7, 9 or 11. The Olympia uses 11 with 7 scores counted (secondary, Fitness Volt).
- Each judge ranks the athletes, first place = 1 point, and the counted placings are summed. Lowest total wins.
- High and low placings are dropped: one high and one low with 5 or 7 judges, two of each with 9 (secondary; sources slightly inconsistent).
- BarBend's examples fit 3 counted judges at the 2025 NY Pro (perfect score 3 per round) and 5 counted at the 2025 Pittsburgh Pro (perfect 5 per round).

**Rounds** (verified, IFBB Pro rules page):

- Men's Open: Judging (prejudging) and Finals, each 50%.
- Fitness: Judging 33%, Finals 67%.
- All other divisions: Judging 100%. The head judge may make the finals callouts a scored "confirmation round" that replaces the Judging scores.

**Ties.** The official rule was not found. Reported practice is that the finals winner may be deemed the winner, or judges take a further vote. At the 2025 Pittsburgh Pro, Walker won prejudging and Lunsford won finals, both totalled 15, and Lunsford won after a third vote (secondary, BarBend).

**Scored depth.** The top 15 are scored; others are usually tied 16th, though international shows sometimes score deeper (secondary). Scores are final and cannot be appealed.

**Multiple divisions.** A pro may not enter more than one division per contest unless the extra entries are Masters classes. Open plus Masters of the same division at one show is allowed (verified). So the same athlete can appear in two rankings on the same day: model these as two observations of one show-day performance, not two independent ones.

**What matters for modelling:**

- The data are ordinal.
- Margins exist only as the summed placings.
- Only the Open and Fitness finals add a second, partly independent observation.
- Callouts at prejudging reveal the judges' provisional top group. That information is useful if it is ever captured.

### B2. Olympia qualification, past and present

**2026 season** (15 Sep 2025 to 30 Aug 2026) (verified, ifbbpro.com):

- Classic Physique, Men's Physique, Bikini and Wellness: points race. The top 25 in the standings qualify, and two open contest wins qualify automatically. Points:

  | Contest | 1st | 2nd | 3rd | 4th | 5th |
  |---|---|---|---|---|---|
  | Olympia | qualified | qualified | qualified | 12 | 10 |
  | Arnold Classic USA, Pittsburgh Pro, NY Pro, Romania Muscle Fest Pro, Dubai Pro | qualified | 14 | 11 | 9 | 7 |
  | All other open IFBB Pro contests | 10 | 4 | 3 | 2 | 1 |

  Ties in the standings are broken by contest level and placing.
- Men's Open, 212, Women's Bodybuilding, Fitness, Figure, Women's Physique and Fit Model: top 3 from the previous Olympia plus the winner of each open contest in the window. For Women's Bodybuilding the Rising Phoenix top 3 also qualify.
- Wheelchair: top 3 at the Olympia, plus contest winners, plus the top 5 in a points table (2nd 10, 3rd 9, 4th 8, 5th 7).
- Former Olympia winners have lifetime eligibility, but need league approval if more than 5 years have passed since their last win.
- Ben Weider Naturals Pro division winners qualify for the Olympia, except in Men's Bodybuilding, 212 and Women's Bodybuilding.
- Stated rationale: some 2025 divisions had about 60 Olympia qualifiers, and the Olympia must keep its prestige (secondary, BarBend quoting Olympia president Dan Solomon).

**2027 season** (31 Aug 2026 to 12 Sep 2027): same points tables, but only the top 20 qualify in the four points divisions. The wheelchair points route drops to the top 3 (secondary, Fitness Volt and Generation Iron; the IFBB page currently shown still describes 2026). Generation Iron printed a different points table; the IFBB table above is the authoritative one.

**2024 and 2025:**

- All divisions: top 3 at the previous Olympia (their place cannot be lost) plus the winner of any IFBB Pro League contest.
- Rare special invites, for example Levrone 2016 and Lunsford 2022 (secondary, BarBend).

**2023 cycle:** Men's Open top 5 from the 2022 Olympia; other divisions top 3; plus every open contest winner (secondary, Wikipedia).

**2012 season:** top 4 at the Olympia, top 3 at the Arnold, top 2 at the NY Pro, plus the winner of every other contest (secondary, Iron Man magazine). Earlier eras used various points systems that I could not document reliably (unsure).

**Arnold Classic USA (Columbus, Ohio)** is invitation-only (memory). It is now formally a top-tier show for the points divisions. Arnold Classic UK, Europe and other brand offshoots are separate, ordinary pro shows (memory).

**Masters Olympia:**

- Held 1994 to 2003 and in 2012, revived in 2023, not held in 2024, held in Tokyo in 2025 (secondary).
- The 2023 entry was by application and invitation. Age floors were 40+ in seven divisions and 45+ in Men's Bodybuilding, 212 and Women's Bodybuilding.
- Natural Masters 40+ winners do not qualify for the Masters Olympia (verified, natural rules).

**What this implies for show strength:**

- The Olympia has always been the best field in the world for its division, but its depth changed with the rules:
  - Large fields of small-show winners in the any-win era (roughly the late 2010s to 2025).
  - Smaller, deeper fields expected for the points divisions from 2026.
- The Arnold USA is invite-only and strong, often next to the Olympia in depth.
- The named top-tier shows attract qualification hunters in the points divisions.
- Small open pro shows are often weak, and their winners were Olympia-qualified only by winning there.
- From 2026, athletes in the four points divisions have a reason to do several small shows (two wins, or 10 points per win). Expect more repeat matchups and somewhat deeper small-show fields in those divisions.
- So treat show strength as time-varying and estimate it from entrants' ratings, not from a fixed tier label.
- "Qualified for the Olympia" means different things in different years. Do not use it as a cross-era athlete feature without adjustment.

### B3. Known patterns that should shape the model

**Incumbency and "politics"** (division and era specific):

- Long consecutive reigns: Men's Open Haney 8, Coleman 8, Heath 7, Schwarzenegger 6, Yates 6; Ms. Olympia Iris Kyle 9, Andrea Shaw 7, Murray 6, Everson 6 (verified, Fitness Volt studies); Classic Physique Bumstead 6, 2019 to 2024, and 202/212 Flex Lewis 7, 2012 to 2018 (memory).
- But in Men's Open the defending champion lost five years in a row: Choopan 2022, Lunsford 2023, Dauda 2024, Lunsford 2025, Walker 2026 (secondary, Bleacher Report). Walker went from 6th at the 2025 Olympia and 2nd at the 2026 Arnold to winning the 2026 Olympia. The 2026 Arnold and Romania winner Andrew Jacked placed 4th. The 2026 placings are provisional pending official scorecards (secondary).
- The league line, from IFBB Pro League VP Tyler Manion: there is no "knock out the champ" rule, and you just have to look best on the day (secondary).
- Commentators describe a tradition that the Arnold winner becomes the Olympia front-runner (secondary, Cicherillo).
- Model consequence: let the data decide.
  - Include the incumbent and prior-placing features in stage 2, with division-specific (possibly era-specific) coefficients.
  - Let the dynamics variance per division absorb how volatile each division's top is.

**Improvement over years and age:**

- Mr. Olympia winners average 32.6 years on contest day (1965 to 2026). The decade average peaked at 37.2 in the 2000s and is 34.3 in the 2020s. Ms. Olympia winners average 33.4 (verified, Fitness Volt).
- No Mr. Olympia under 30 since Haney (secondary, TheBarbell).
- Implication: in Open bodybuilding expect a positive drift over a pro career into the mid 30s. Physique divisions are probably younger (not researched).
- Add age and years-as-pro as drift covariates, or allow a positive mean drift for young pros in the dynamics.

**Comebacks:**

- Mostly anecdotal. Phil Heath returned in 2020 after a 2018 loss and a year off, and placed 3rd (secondary). Ex-champions returning after long layoffs rarely regain the title, but the evidence is thin.
- Variance growth plus mild mean reversion handles this. Measure the layoff effect in our data.

**Division switching:**

- Fitness Volt counts 69% of division switchers winning (a show) in their new division (verified, Fitness Volt study). This is a heavily selected sample: athletes switch when they expect to do better, and many moves go to smaller divisions.
- Most common moves: Women's Physique to Women's Bodybuilding (18), Men's Open to 212 (18), Men's Physique to Classic Physique (12), Classic Physique to 212 (10), Figure to Women's Physique (9).
- Implication: initialise the new-division rating from a learned mapping of the old-division rating, with inflated variance, rather than as a rookie.

**Conditioning variance:**

- Show-day condition (flat, spilled, holding water, missed peak) is the dominant noise source. It is athlete-specific: some athletes are consistent, others volatile.
- Elo-MMR's robustness to one extreme result, and athlete-specific performance variance (Data Golf shrinks player variance toward the tour mean), are the relevant tools.
- Back-to-back weekends and multiple shows in short succession are plausible negative factors. They are untested, so treat them as a candidate feature.

**Year-to-year stability of Olympia placings:**

- No published statistic was found. Compute it ourselves: Spearman between consecutive Olympia placings for repeat entrants, by division and era.

**Field sizes and careers** (Fitness Volt Men's Open database):

- Average career 5.6 years and 4.6 contests per athlete, counting athletes with 2 or more contests (verified, Fitness Volt analytics).
- Their "average field 7.9" figure looks inconsistent with their own season table (unsure).
- Sparse histories per athlete mean priors and pooling matter a lot.

**Division history** (memory, approximate):

- Olympia contests started in different years: Fitness 1995, Figure 2003, Bikini 2010, 202/212 2012, Men's Physique and Women's Physique 2013, Classic Physique 2016, Wellness around 2020 to 2021 (Fitness Volt's "on this day" page says the division was introduced in 2020), Fit Model 2026.
- Ms. Olympia was not held 2015 to 2019. The Rising Phoenix was the top Women's Bodybuilding show in those years (memory).
- Ms. International at the Arnold was a major Women's Bodybuilding show until about 2013 (memory).
- Event tier mapping must follow this history.

### B4. Masters, Natural, Amateur

**Masters:**

- Pro shows run Masters classes (35+, 40+, 45+, 50+ and so on, varying by show). Athletes may enter Open and Masters of the same division at one show (verified, rules).
- Masters fields are mostly weaker and older, but they share athletes with the Open division.
- Best treatment: one rating pool per division across Open and Masters. The model then calibrates Masters field strength through the shared athletes. Add age as a covariate rather than a separate pool.

**Natural (IFBB Pro League natural pro circuit)** (verified, ifbbpro.com natural rules):

- Testing by urinalysis and polygraph; the top two finalists in each division must submit to urinalysis.
- Clean periods of 6 months to 3 years by substance class. HRT/TRT users are ineligible.
- Natural pro division winners qualify for the Ben Weider Naturals Pro. Its winners qualify for the (open) Olympia in the physique-type divisions.
- So a natural-to-open bridge exists but is tiny. The natural and enhanced populations differ systematically.
- Keep a separate natural pool per division. Optionally link the few crossover athletes with an offset parameter.
- Sources differ on testing details at amateur natural qualifiers (unsure).

**Amateur (NPC and NPC Worldwide):**

- Pro cards come from national-level and international pro-qualifier shows. A division needs at least 5 competitors for a card to be awarded at NPC Worldwide (USA) pro qualifiers (secondary).
- Amateur classes (height or weight classes) are separate rankings. The "overall" is a contest among class winners only, a selected subset, so model it as its own small ranking or skip it.
- Amateur results are a separate, larger pool. Their main value is as the prior for a pro debut.

**Federation lineage** (memory, unsure):

- The IFBB (amateur) and the IFBB Pro League separated in 2017. The IFBB then started its own "Elite Pro" circuit.
- If our historical "IFBB pro" data include post-2017 Elite Pro results, treat them as a different, weaker circuit, not as IFBB Pro League shows.

### B5. Existing public rankings, predictions and research

- **Fitness Volt:** editorial Olympia predictions and power rankings based on recent form, Olympia history and momentum. The rankings come with a disclaimer that past performance is not indicative. They also aggregate expert picks, run an Olympia points calculator, publish contest databases with division analytics, and publish small "studies" on winning age, reigns, countries and division switches (verified and secondary).
- **Generation Iron, BarBend, Muscle and Fitness:** prediction articles and qualification explainers, all editorial (secondary).
- **Official:** the IFBB Pro League publishes qualification rules and point standings. Athletes must track their own points (verified). This is an achievement standing, not a skill rating.
- **Erdos Institute "Muscle Metrics" (2024, De Silva and Prokhorov):**
  - Plain Elo on NPC News Online results.
  - Claims Elo-based predictions beat alphabetical ordering, but publishes no metrics.
  - Notes the off-season effect.
  - Proposes scorecard OCR as future work.
  - Code at github.com/jessicadesilva/bodybuilding-ranking (secondary).
- **Academic:** Gaines (2001) on anthropometrics vs judges' scores (see A4). No peer-reviewed study predicting pro bodybuilding placings, and no bias study of IFBB Pro judging, was found. The closest rigorous work is the gymnastics, skating and diving judge-evaluation literature.

---

## Implications for our model

1. **Core rating:** use a dynamic Plackett-Luce model.
   - Run it online (Weng-Lin/OpenSkill-style or Powell's online PL with per-athlete variance and forgetting) for walk-forward forecasts.
   - Run a smoothed version (TTT-style, or a batch MAP PL with exponential time weights and a normal prior) for the "power ranking now" and history pages.
   - Keep pairwise Elo with K/(N-1) and Elo-MMR as baselines. Expect small differences between principled systems and invest effort in dynamics and features.
2. **Unit of observation:** one ranking per contest (show x division x class or age group x level x natural/open).
   - Encode the 16th-place tail as a tie group, or truncate at the scored depth (15, or 10 for Bikini, Wellness and Fit Model).
   - Fit both forward and reverse PL and keep whichever wins on held-out top-5 log-likelihood. Literature and judging consensus at the top both favour trying reverse PL.
3. **Event weighting:**
   - Do not hand-set "Olympia counts triple". Opponent strength is already in the likelihood.
   - Learn a per-tier performance noise (or contest weight) by maximum likelihood. Tiers: Olympia; Arnold USA plus the named top shows; other open pro; masters pro; natural pro; amateur national; amateur regional.
   - Show a separate, descriptive field-strength index (for example the mean top-5 pre-show rating) for the app.
4. **Time:**
   - Variance grows per day of inactivity, capped at the rookie variance, with mild mean reversion for multi-year gaps.
   - Put the layoff effect on the mean in stage 2 as a learned curve. A 1-year break may help; 3 or more years probably hurts.
   - Tune decay by walk-forward log score (Data Golf found "medium-term" decay best in golf).
5. **Rookies:**
   - Prior mean by entry pathway (NPC national overall vs class win, international qualifier, natural) and by amateur rating where available.
   - Use a large prior variance so the first 2 to 3 pro shows move the rating quickly.
   - Calibrate the prior from the observed first-season placing percentile per pathway.
6. **Division switches:** carry the rating over through a learned linear map per (from, to) pair, with inflated variance. Do not reset to a rookie prior.
7. **Masters:** same rating pool as the Open of that division, with age as a covariate. Same-day Open and Masters results of one athlete share one show-day performance; do not count them as fully independent evidence.
8. **Natural and Amateur:** separate pools. The amateur pool feeds pro debut priors. Natural crossovers into the open circuit (Ben Weider Naturals Pro to Olympia) get an estimated offset.
9. **Incumbency and reputation:** test stage-2 features (defending Olympia champion, last Olympia placing, defending champion of this show, pro wins to date) with division and era specific coefficients. The literature predicts a positive reputation effect, but Men's Open 2022 to 2026 shows it can be weak.
10. **Margins:**
    - Use scorecard totals (average placing per counted judge) to weight updates or as an ordinal target.
    - If per-judge placings exist, model judge noise separately from show-day noise. This is the most valuable data upgrade available.
11. **Probabilities:** winner from a softmax with a calibrated temperature; placings, P(top 3) and expected placing from Monte Carlo (Gumbel-max or Normal) including posterior rating uncertainty. Check place-probability bias for favourites and correct with Benter-style exponents or a Thurstone noise model if needed.
12. **Evaluation:** strict walk-forward. Headline metrics are winner log-loss and top-5 PL log-likelihood, each relative to uniform. Secondary: Brier, top-1 and top-3 accuracy, Spearman on expected placing, NDCG@5, reliability diagrams. Slice by tier and field size, and benchmark Olympia predictions against aggregated expert picks.
13. **Power ranking:**
    - Rank by a conservative score (for example mu minus 1 to 2 sigma), with an activity rule (for example competed in the last 18 to 24 months). Keep the raw predictive rating separate.
    - Show uncertainty and "last competed".
    - Keep Open-age and Masters-only athletes in one division list, with a filter.
14. **Qualification-era awareness:** Olympia field size and depth changed with the rules (any-win era vs the 2026 points system and the 2027 top 20). Use rating-based field strength, never "is Olympia qualified", as a cross-era signal.

## Things worth measuring first in our data

- Defending-champion retention and Spearman of consecutive Olympia placings, by division and decade.
- Arnold USA winner's Olympia finish distribution, by division.
- Placing percentile vs pre-show rating expectation as a function of layoff length (0 to 6 months, 6 to 12, 12 to 24, more than 24).
- First-season pro placing percentile by pro-card pathway (rookie prior).
- First result in a new division vs the last rating in the old one (switch mapping).
- Same-country-as-show effect at non-US pro shows.
- Back-to-back weekend effect.
- Learned noise scale per event tier (is a small show more or less "random" than the Olympia?).
- Share of contests with a 16th-place tie block, and how often scorecards provide per-judge placings.

---

## Sources

Methods and ratings
- https://ar5iv.labs.arxiv.org/html/2101.00400 (Ebtekar and Liu, Elo-MMR, WWW 2021)
- https://github.com/EbTech/Elo-MMR (Elo-MMR implementations, contest weights, ties)
- https://dmoj.ca/post/206-ratings-migrated-to-elo-mmr (DMOJ migration from TopCoder-style ratings)
- https://www.glicko.net/research/multicompetitor.pdf (Glickman and Hennessy, stochastic rank ordered logit, preprint)
- https://ideas.repec.org/a/bpj/jqsprt/v11y2015i3p131-144n5.html (same, JQAS 2015 record)
- https://projecteuclid.org/journals/bayesian-analysis/volume-13/issue-2/A-Comparison-of-Truncated-and-Time-Weighted-PlackettLuce-Models-for/10.1214/17-BA1048.full (Henderson and Kirrane 2018)
- https://eprints.ncl.ac.uk/231830 (Henderson and Kirrane, repository record)
- https://arxiv.org/pdf/2002.05953 (Johnson, Henderson, Boys, Extended Plackett-Luce)
- https://eprints.whiterose.ac.uk/200471 (Powell 2023, generalizing Elo, JQAS)
- https://www-users.york.ac.uk/~bp787/Generalizing_Elo_arxiv.pdf (Powell, full text)
- https://jmlr.org/papers/v12/weng11a.html (Weng and Lin 2011, Bayesian approximation for online ranking)
- https://arxiv.org/pdf/2401.05451 (OpenSkill)
- https://joss.theoj.org/papers/10.21105/joss.05901.pdf (OpenSkill, JOSS)
- https://www.microsoft.com/en-us/research/publication/trueskilltm-a-bayesian-skill-rating-system/ (TrueSkill)
- https://www.microsoft.com/en-us/research/publication/trueskill-2-improved-bayesian-skill-rating-system/ (TrueSkill 2)
- https://www.jstatsoft.org/article/view/v112i06 (Landfried and Mocskos, TrueSkill Through Time, JSS 2025)
- https://cran.r-project.org/web/packages/TrueSkillThroughTime/index.html (TTT R package)
- https://www.remi-coulom.fr/WHR/ (Coulom, Whole-History Rating)
- https://www.glicko.net/glicko/glicko2.pdf (Glickman, Glicko-2)
- https://www.rdocumentation.org/packages/PlayerRatings/versions/1.1-0/topics/glicko2 (Glicko-2 parameters)
- https://arxiv.org/abs/1810.12068 (Turner et al., PlackettLuce R package)
- https://cran.r-project.org/web/packages/PlackettLuce/index.html (PlackettLuce on CRAN)
- https://vuir.vu.edu.au/34652/ (Kovalchik 2016, tennis win prediction)
- https://www.tennisabstract.com/blog/?p=2129 (tennis model accuracy check)
- https://ar5iv.arxiv.org/html/2010.11187 (Szczecinski, G-Elo, margin of victory)
- https://arxiv.org/pdf/1802.00527 (margin-based outcome prediction, cites Kovalchik 2020)
- https://www.stat.berkeley.edu/~aldous/Talks/Elo_talk.pdf (World Football Elo K values)
- https://theconversation.com/what-are-the-world-football-elo-ratings-27851 (World Football Elo goal multiplier)
- https://datagolf.com/predictive-model-methodology (Data Golf methodology)
- https://www.Gwern.net/doc/statistics/decision/1994-benter.pdf (Benter 1994, scanned)
- https://oddspapi.io/blog/?p=3174 (summary of Benter's combination formula and R^2 values)
- https://www.stat.berkeley.edu/~aldous/157/Papers/ali.pdf (Ali, probability models on horse race outcomes)
- https://hub.hku.hk/handle/10722/64961 (Lo and Bacon-Shone, Harville vs Henery and Stern)
- https://combat-edge.com/news/ufc-new-ranking-system-combat-edge-elo/ (MMA inactivity handling)
- https://kickboxhub.com/methodology (tiered K and inactivity decay)
- https://arxiv.org/pdf/1908.08980 (Wheatcroft, case against the RPS)
- https://apps.dtic.mil/sti/pdfs/ADA459827.pdf (Gneiting and Raftery, strictly proper scoring rules, tech report)
- https://arxiv.org/abs/2608.23859 (Szczecinski 2026, ranking by points and ordinal models)

Judging and bias
- https://arxiv.org/abs/1807.10055v1 (Heiniger and Mercier, judging error across 8 sports)
- https://ar5iv.arxiv.org/html/1807.10021 (Heiniger and Mercier, judging the judges)
- https://ar5iv.arxiv.org/html/1807.10033 (national bias of gymnastics judges 2013 to 2016)
- https://www.gsb.stanford.edu/faculty-research/working-papers/nationalism-winter-sports-judging-its-lessons-organizational (Zitzewitz)
- https://ruor.uottawa.ca/handle/10393/9076 (Findlay, reputation bias in figure skating)
- https://heibib.ub.uni-heidelberg.de/search/Record/1564441474 (Plessner 1999, expectation biases in gymnastics)
- https://www.haas.berkeley.edu/wp-content/uploads/Kim-and-King-Matthew-Effects.pdf (Kim and King, status bias in umpiring)
- https://business.columbia.edu/press-release/cbs-press-releases/all-star-pitchers-will-hate-instant-replay-according-new-research (Kim and King press release)
- https://wbldb.lievers.net/10346865.html (Gaines 2001 thesis abstract, anthropometrics vs judges)
- https://jahrbib.sulb.uni-saarland.de/bitstream/20.500.11880/40955/1/National%20bias%20in%20international%20sports%20judging%20%20a%20scoping%20review.pdf (national bias scoping review)

Bodybuilding rules, qualification and history
- https://www.ifbbpro.com/olympia-qualification-rules/ (2026 Olympia qualification system)
- https://barbend.com/news/2026-olympia-qualification-points-system/ (2026 changes and rationale)
- https://fitnessvolt.com/ifbb-pro-league-announces-2026-olympia-qualification-system (2026 system)
- https://fitnessvolt.com/ifbb-pro-league-2027-mr-olympia-qualification-new-rule/ (2027 top 20 change)
- https://generationiron.com/2027-olympia-qualification-rule-change/ (2027 explainer, conflicting points table)
- https://barbend.com/news/2025-olympia-qualification-rules/ (2025 rules)
- https://en.wikipedia.org/wiki/International_Federation_of_Bodybuilding_%26_Fitness_Professional_League (2023 cycle, league history)
- https://www.ironmanmagazine.com/?p=27651 (2012 qualification rules)
- https://www.ifbbpro.com/rules/ (IFBB Pro rules: round weighting, one division per contest)
- https://barbend.com/news/how-scoring-in-bodybuilding-works/ (scoring, judges, dropped placings, ties, examples)
- https://fitnessvolt.com/steve-weinberger-talks-current-judging (panel sizes, Olympia 11 judges)
- https://www.ifbbpro.com/natural-pro (IFBB Pro natural rules)
- https://npcnewsonline.com/npc-ifbb-pro-natural-contest-rules/ (NPC natural rules)
- https://fitnessvolt.com/masters-olympia-return-2023/ (Masters Olympia revival)
- https://fitnessvolt.com/2025-masters-olympia-results/ (2025 Masters Olympia)
- https://en.wikipedia.org/wiki/Professional_bodybuilding (pro card pathway)
- https://fitnessvolt.com/2026-mr-olympia-bodybuilding-results/ (2026 Mr. Olympia results)
- https://bleacherreport.com/articles/25503439-mr-olympia-2026-results-prize-money-payouts-winner-and-top-contestants (2026 results, no repeat champion streak)
- https://fitnessvolt.com/2026-mr-olympia-predictions/ (2026 expert predictions)
- https://fitnessvolt.com/tyler-manion-2023-mr-olympia-recap/ (Manion on beating the champion)
- https://fitnessvolt.com/bob-cicherillo-says-dauda-derek-threat-to-hadi-at-2023-olympia/ (Arnold winner as front-runner)
- https://barbend.com/?p=94000 (Phil Heath 2020 comeback)
- https://fitnessvolt.com/bodybuilding/studies/ (Fitness Volt studies: age, reigns, countries, division switches)
- https://fitnessvolt.com/bodybuilding/studies/mr-olympia-winning-age/ (Mr. Olympia winning age)
- https://thebarbell.com/?p=5201 (Olympia vs Arnold winner ages)
- https://fitnessvolt.com/bodybuilding/divisions/analytics/ (division analytics, careers, field sizes)
- https://fitnessvolt.com/bodybuilding/on-this-day/ (division introduction years)
- https://fitnessvolt.com/bodybuilding/predictions/ (Fitness Volt predictions and power rankings)
- https://fitnessvolt.com/bodybuilding/calculators/olympia-points/ (Olympia points calculator)
- https://www.erdosinstitute.org/project-database/spring-2024/data-science-boot-camp/muscle-metrics%3A-revolutionizing-bodybuilding-competitions-with-elo-ratings (Erdos Institute bodybuilding Elo project)
- https://www.erdosinstitute.org/_files/ugd/6cd34c_0d00d1dc66874541abc46746cc6bce3e.pdf (Erdos project poster)
