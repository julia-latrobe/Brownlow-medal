# Changelog

Notable changes to this project. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[semantic versioning](https://semver.org/).

## [Unreleased]

### Added

- **Player pages** on the results site: a player's season match by match, with
  expected votes per round, the chance of taking 3, 2 and 1 votes in each game,
  their likely range, season rank and best projected game.
- **Round pages**: every match in a round, ordered by the actual date and start
  time it was played, with the projected 3-2-1 and the runners-up.
- **Team pages**: a club's squad ranked by projected votes, its fixtures, and the
  player most likely to poll in each one.
- **Navigation throughout**: every player, team and round name is a link; each
  view links back to the season page and the header title is a home link. Views
  are addressed by URL fragment, so any page can be linked to directly.
- **Player search** with a type-to-filter box that narrows to the selected team.
- **`annotations`** in an experiment config, for a short label beside a player's
  name (e.g. `"Omitted"`). Presentational only — never an input to the model.
- Match date, local start time and venue are now carried through the pipeline
  and written to `predictions.csv`, so matches can be shown in playing order.

### Fixed

- **Two players who share a name are no longer treated as one person.** The
  simulation keyed on the player's name alone, so the two Bailey Williamses (West
  Coast and Western Bulldogs) had their votes pooled, which inflated the combined
  total and gave both the same wrong win probability. Identity is now the name
  *and* the club, everywhere. `season_totals` was already correct; the simulation
  and the leaderboard join were not.

## [0.1.0] - 2026-08-29

First working version.

### Added

- **Data loading** (`brownlow.data`) from the AFL Tables dataset, with filters for
  seasons, rounds, teams and players. Finals are excluded by default, since no
  Brownlow votes are awarded in them. Rebuilds player names where the upstream
  mirror has gaps.
- **Feature engineering** (`brownlow.features`): raw counting stats, within-match
  z-scores and percentile ranks, team context and margin, win interactions,
  share-of-team stats, and composites including AFL Fantasy points. Configurable
  through `FeatureConfig`.
- **`PlackettLuceModel`** — a rank-ordered conditional logit fitted by maximum
  likelihood, modelling the umpires' sequential 3-2-1 choice within each match.
  Produces exact closed-form expected votes.
- **`WeightedLogisticModel`** — weighted logistic regression ranked to 3-2-1, kept
  as a comparison point.
- **Season simulation** (`brownlow.simulate`) via Gumbel top-k sampling, giving
  win probabilities, top-5 probabilities and credible intervals. Supports marking
  players ineligible.
- **Evaluation** (`brownlow.evaluate`): per-match and season metrics, held-out
  log-likelihood, and season-wise cross-validation (walk-forward and
  leave-one-season-out).
- **Experiment harness** (`brownlow.experiment`): JSON run configs, four standard
  output artefacts per run, cross-run comparison, and regularisation tuning.
- **Results website** (`brownlow.report`): a self-contained `docs/index.html` with
  a run selector, a team filter, projected totals by team, fitted coefficients and
  the holdout comparison.
- **`brownlow` command line** covering fetch, backtest, predict, cv, tune, run,
  report and compare.
- **Synthetic data generator** (`brownlow.synthetic`) so the test suite runs
  without network access and model recovery can be tested against a known answer.
- GitHub Actions for CI, publishing the site, and refreshing projections weekly.
- VS Code configuration: recommended extensions, test discovery, and debug
  configurations for each command.

[Unreleased]: https://github.com/julia-latrobe/Brownlow-medal/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/julia-latrobe/Brownlow-medal/releases/tag/v0.1.0
