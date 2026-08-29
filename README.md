# Brownlow Medal prediction

[![CI](https://github.com/julia-latrobe/Brownlow-medal/actions/workflows/ci.yml/badge.svg)](https://github.com/julia-latrobe/Brownlow-medal/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Predicts AFL Brownlow Medal votes from player match statistics, and projects the
season's final count.

**New to this project, or to Git and VS Code?** Start with
**[GETTING_STARTED.md](GETTING_STARTED.md)** — it goes from an empty computer to a
running model. This file explains how the model works.

📊 **[Live results](https://julia-latrobe.github.io/Brownlow-medal/)**

---

## The problem

After every home-and-away match the three field umpires confer and award **3, 2
and 1 votes** to the players they judged best. Add them up over the season and the
player with the most wins the Brownlow Medal. Finals matches award no votes.

So each match hands out exactly six votes, to exactly three players, in a
particular order. That structure is the whole problem, and it is what the model
is built around.

## How the model works

### The idea

A natural first attempt is to fit a logistic regression for "did this player
poll?", score every player, and give 3-2-1 to the top three scores in each match.
That works. This project includes it as `WeightedLogisticModel`, weighting a
3-vote game more heavily than a 1-vote game during fitting.

But it models the wrong thing in two ways:

1. **It scores players against the whole league rather than against their own
   match.** Every match awards six votes, whether it was a 150-point blowout or a
   wet 40-point slog. A global model has to learn "25 disposals is good" as an
   absolute claim. What actually decides votes is being the best player *in this
   particular game*.
2. **The vote weighting has to be guessed.** Weighting 3-vote games more is
   sensible, but the size of the weight is arbitrary.

`PlackettLuceModel` fixes both by modelling what the umpires actually do. They
pick a best player from the 44 on the ground, then a second best from those
remaining, then a third. That is a **rank-ordered conditional logit**, also called
a Plackett–Luce model. Each player gets a latent quality score `v = x · β`, and:

```
P(3 votes = a)          = softmax over everyone in the match
P(2 votes = b | a)      = softmax over everyone except a
P(1 vote  = c | a, b)   = softmax over everyone except a and b
```

The model maximises the likelihood of the orderings that actually happened.

Two things fall out of this for free:

- **The relative worth of 3 versus 1 votes is derived, not tuned.** The 3-vote
  player has to beat the field three times over; the 1-vote player only has to
  survive the final draw.
- **Match context cancels exactly.** Because every softmax is taken within a
  match, anything constant across that match — the weather, the era, how generous
  the umpires were that day — divides out. For the same reason the model has no
  intercept: a constant added to every player in a match would cancel too.

### From probabilities to a projection

For each match the model computes each player's probability of taking 3, 2 and 1
votes in closed form, and combines them into an **expected vote count**. These sum
to exactly 6 per match, matching reality.

For the question everyone actually asks — *what are the odds this player wins?* —
expected votes are not enough. Winning needs an outlier, not an average, so
`simulate_season` runs the whole season 10,000 times, drawing each match's 3-2-1
from the fitted probabilities, and counts how often each player finishes on top.
Suspended players can be marked ineligible: they still poll votes, because the
umpires do not know about the tribunal, but they cannot win.

### Features

Raw counting stats (disposals, goals, tackles, clearances, contested possessions,
time on ground and the rest), plus:

- **Within-match comparisons** — each stat as a z-score and a percentile rank
  against the other players in the same game. These travel far better across eras
  and conditions than raw totals: the best player in a wet, low-scoring game still
  polls.
- **Team context** — whether the player's team won, and by how much (clipped,
  since an umpire's read of "this team won" saturates well before a 100-point
  margin).
- **Win interactions** — the same performance is worth more in a win than in a
  loss, so the key stats get an explicit interaction term.
- **Share of team** — a player's share of their team's disposals, goals and
  contested possessions.
- **Composites** — AFL Fantasy points, goal involvements, clean disposals.

All of it is configurable through `FeatureConfig`, so turning a group off is one
line in an experiment file.

## Does it work?

Six-fold walk-forward cross-validation on seasons 2015–2025 — each fold trains
only on seasons *before* the one it is scored on:

| Metric | Rank model | Weighted logistic | Random guessing |
| --- | --- | --- | --- |
| Picked the 3-vote getter | **57.3%** | 56.3% | ~2.3% |
| Vote-getters inside our top 3 | **64.8%** | 64.1% | ~6.8% |
| All three, in the right order | **8.5%** | 7.8% | ~0.001% |
| Log-likelihood per match | **−5.40** | −5.75 | −11.40 |
| Season leaderboard, all players | 0.740 | **0.793** | 0 |
| Season leaderboard, players who polled | **0.796** | 0.760 | 0 |

The rank model is better at the job it is built for — picking the right players
in a given match — and substantially better calibrated, which is what the
log-likelihood measures.

The one row where the simpler model wins deserves explaining rather than hiding.
Across *all* players, 71% of whom finish the season on zero votes, the logistic
model scores better on rank correlation. That is largely an artefact of the
metric: the logistic assigns hard integer 3-2-1s, so most players get exactly
zero, which reproduces the real distribution's mass of ties. The rank model gives
everyone a small non-zero expectation, and the resulting long tail of near-ties
costs it. Restrict the comparison to players who actually polled and the ordering
reverses. Take both readings with the caveat that all-player Spearman is a weak
metric on data this zero-inflated.

Expect roughly 55–60% top-1 accuracy in a normal season. The model called
Nick Daicos as 2025's leading vote-getter and had him within a vote of his real
total, but the actual medallist was Matt Rowell, whom it had 7th. That is the
honest shape of this problem: the leaderboard is broadly predictable, the winner
often is not.

## Installation

```bash
git clone https://github.com/julia-latrobe/Brownlow-medal.git
cd Brownlow-medal

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1

pip install -e ".[data,dev]"
```

`[data]` adds the packages that read the source dataset; `[dev]` adds the test and
lint tools. The model itself only needs numpy, pandas and scipy.

## Usage

The workflow is three steps: **build**, **hold out and check**, then **predict a
season nobody knows the answer to.**

```bash
# Download the data (about 14 MB, cached afterwards)
brownlow fetch

# 1 & 2. Train on past seasons, score against seasons the model never saw
brownlow backtest --train 2015-2023 --test 2024-2025 --compare

# 3. Project the current season and rebuild the results page
brownlow predict --seasons 2015-2026 --train 2015-2025 --test 2025 --predict 2026
```

Other commands:

```bash
brownlow cv --min-train 5                  # walk-forward cross-validation
brownlow tune --alphas 0.1,1,10            # grid-search the regularisation
brownlow run experiments/rank-model.json   # run a saved experiment
brownlow compare                           # table of every run so far
brownlow report                            # rebuild docs/index.html
```

Every command takes the same data filters, and all of them accept a single value,
a list or a range:

```bash
brownlow backtest --seasons 2018-2025 --rounds 1-23 --teams Geelong,Carlton
```

From Python:

```python
from brownlow import load_player_matches, PlackettLuceModel, simulate_season

history = load_player_matches(seasons="2015-2025", require_complete_votes=True)
current = load_player_matches(seasons="2026")

model = PlackettLuceModel(alpha=1.0).fit(history)
predictions = model.predict(current)

print(model.season_totals(predictions).head(10))
print(simulate_season(predictions, n_simulations=10_000).head(10))
```

`examples/quickstart.py` walks through the whole thing step by step. It is written
in `# %%` cells, so VS Code can run it a piece at a time like a notebook while
staying a plain Python file that diffs cleanly.

## Running experiments

An experiment is a small JSON file in `experiments/`. Because it is a file, trying
an idea is a one-file change you can put in a pull request, and comparing ideas is
reading the metrics back off disk.

```json
{
  "name": "rank-model",
  "model": "plackett_luce",
  "alpha": 1.0,
  "train_seasons": "2015-2024",
  "test_seasons": "2025",
  "predict_seasons": "2026",
  "n_simulations": 10000,
  "features": {},
  "notes": "Default rank model over the full feature set."
}
```

```bash
brownlow run experiments/rank-model.json --seasons 2015-2026
brownlow compare
```

Each run writes four things to `data/output/<name>/`:

| File | What it holds |
| --- | --- |
| `predictions.csv` | One row per player per match: vote probabilities and expected votes. |
| `leaderboard.csv` | One row per player: projected total, win probability, likely range. |
| `metrics.json` | Holdout accuracy plus the exact config used — self-describing. |
| `coefficients.csv` | What the model learned, largest effect first. |

### Cross-validation

**Never split this data randomly.** Rows from the same match are not independent —
if the 3-vote getter lands in train and the 2-vote getter in test, the model has
already seen the answer. The game also changes over time. Every split here is by
season:

```bash
brownlow cv --method rolling --min-train 5   # walk-forward: train on the past only
brownlow cv --method loso                    # leave one season out: more folds
```

Use walk-forward for an honest estimate of live accuracy. Leave-one-season-out has
more folds so it is steadier for tuning a hyperparameter, but it lets the future
leak into the past — never quote it as expected accuracy.

## The results page

`brownlow report` reads every run in `data/output/` and writes a single
self-contained `docs/index.html` — no build step, no JavaScript dependencies, no
CDN.

It has four linked views, so you can start anywhere and click through:

| View | What it shows |
| --- | --- |
| **Season** | The projected leaderboard, win probabilities, votes by team, the fitted coefficients, and the holdout comparison. |
| **Player** | One player's season match by match: expected votes per round, the chance of taking 3, 2 and 1 votes in each game, their likely range and their best projected game. |
| **Round** | Every match in a round, in the order it was played, with the projected 3-2-1 and the runners-up. |
| **Team** | A club's whole squad ranked, its fixtures, and the player most likely to poll in each one. |

Every name is a link — players, teams and round numbers all navigate — and each
view links back to the season page, as does the title in the header. Views are
addressed by URL fragment (`#player=Nick Daicos|Collingwood`, `#round=12`,
`#team=Geelong`), so any page can be linked to or shared directly.

There is also a run selector, a team filter, and a type-to-search box for
players. Choosing a team narrows the search box to that club's players.

### Labelling a player

`annotations` in an experiment config puts a short label beside a player's name
wherever it appears — useful for marking someone as unavailable:

```json
"annotations": { "Some Player": "Omitted" }
```

Annotations are presentational only. They never reach the model, the projection
or the simulation, so adding one cannot change a number on the page. To actually
exclude a player from *winning* the medal while still having them poll votes —
what a suspension does — use `ineligible` instead.

The page is committed to the repository, so the numbers always correspond to a
specific commit. Merging a change to `docs/` on `main` triggers the **Publish
site** workflow and the live page updates.

> One-time setup: **Settings → Pages → Source: GitHub Actions**.

`.github/workflows/refresh-projections.yml` re-runs every experiment against fresh
data on a weekly schedule and opens a pull request when the projections move.

## Project layout

```
src/brownlow/
  data.py        Download, tidy and filter the AFL Tables data
  features.py    Turn raw statistics into model features
  model.py       PlackettLuceModel and WeightedLogisticModel
  simulate.py    Monte Carlo the season for win probabilities
  evaluate.py    Metrics, backtesting and season-wise cross-validation
  experiment.py  Run configs, write artefacts, compare runs
  report.py      Build the results website
  synthetic.py   Generate fake seasons with a known answer, for tests
  cli.py         The `brownlow` command
experiments/     Experiment configs, one JSON file each
examples/        A step-by-step walkthrough
tests/           The test suite
docs/            The generated results page (published by GitHub Pages)
```

## Data and credit

All player statistics and historical Brownlow votes come from
**[AFL Tables](https://afltables.com)**, which has recorded them since 1897.

They are read through the public data mirror maintained by the
**[fitzRoy](https://github.com/jimmyday12/fitzRoy)** project — an R package for AFL
data by James Day and contributors — which scrapes AFL Tables with permission and
republishes a tidy snapshot at
[jimmyday12/fitzroy_data](https://github.com/jimmyday12/fitzroy_data) on a
scheduled job.

No fitzRoy code is used here. `src/brownlow/data.py` is an independent Python
reader for the dataset that project publishes, written because fitzRoy itself is
an R package. If you use this data, please credit both AFL Tables and fitzRoy, and
read their terms before redistributing it.

The AFL and Brownlow Medal names and marks belong to the Australian Football
League. This project is not affiliated with or endorsed by the AFL.

## Known limitations

Worth knowing before trusting a number, and each one is a reasonable thing to
work on:

- **Umpire identity is ignored.** The dataset names the umpires for each match and
  the model does not use them, though individual umpires plausibly reward
  different things.
- **No within-season form.** Every match is treated independently. Reputation and
  recent form probably influence votes.
- **No suspension data.** Ineligible players have to be passed in by hand via
  `--ineligible`.
- **Timing within a match is invisible.** A goal in the last two minutes of a
  close game counts the same as one in garbage time.
- **All-player rank correlation is a weak metric here**, for the zero-inflation
  reason described above. A better season-level metric would be worth having.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Issues and pull requests are welcome —
`.github/ISSUE_TEMPLATE/model_idea.md` is the right template for "I think this
would predict better", and it asks you to say in advance which metric should move.

## License

[MIT](LICENSE).
