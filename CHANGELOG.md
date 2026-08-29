# Changelog

Notable changes to this project. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[semantic versioning](https://semver.org/).

## [Unreleased]

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
