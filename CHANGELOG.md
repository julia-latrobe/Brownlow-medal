# Changelog

Notable changes to this project. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[semantic versioning](https://semver.org/).

## [Unreleased]

### Added

- **An `umpire-overlay` scenario**, whose range shows the umpires' standing view
  rather than the randomness of the count. The low end is where a player lands
  on his statistics alone, the high end where he lands if the umpires keep
  treating him as they have for years — one-sided by nature, and for 2026 the
  live question, since umpires now see the statistics before voting and no
  season the adjustment was measured on worked that way. Cripps spans 20.8 to
  27.2 votes, Newcombe 17.2 to 22.7, Gawn 17.0 to 22.4; Ashcroft moves 0.2.
  Projections and ranking are identical to `player-adjusted`, so the two read
  side by side. Adds `interval` to an experiment config, and
  `predict_at_strength` / `bias_band` / `bias_overlay` to the model.

### Fixed

- **The widened simulated range was centred too low.** Softening the
  probabilities moves the leaders' share of the three votes to the field, since
  a match always awards six, so every leading player drifted down — five votes
  for a runaway favourite. It read as a long tail below each of them and almost
  none above. Ranges now slide back onto the projection with their shape intact:
  the simulated mean was out by 1.9 votes on a top-40 player and is now out by
  0.8, matching the projection it sits on, with none of the width given back.

- **Honest confidence on the simulated results.** The season simulation was
  stating ranges far tighter than the evidence supports: measured on 240
  held-out player-seasons, a stated 95% range held the truth 84% of the time and
  a 50% range held it 46%. Two corrections, both fitted on held-out seasons and
  both in `simulate.py`: `simulation_temperature` (default 0.8) softens the
  fitted probabilities, and `player_shock` (default 0.5) allows that the model's
  read on a player may simply be wrong, drawn once per simulated season and held
  across that player's matches. Together they bring coverage to 96% and 58%.
  Neither changes a projected vote total or the order of the leaderboard — those
  come from the exact marginals, which no simulation touches. Both are settable
  per experiment; 1.0 and 0.0 restore the previous behaviour.
- **A `player-adjusted` scenario**, giving each player a standing adjustment for
  how the umpires have actually treated him rather than how his statistics read.
  A player's gap between actual and predicted votes one season correlates +0.32
  with the next across 244 held-out player-seasons, so the effect is a trait, not
  noise. Half the measured adjustment carries forward by default, discounted
  again for players seen in only a season or two; `strength: 0.0` reproduces the
  plain rank model exactly. Cuts the error on a top-40 player's season total by
  about a tenth of a vote on walk-forward seasons, most of it in the recent ones,
  and leaves per-match accuracy unchanged. Learns from completed seasons only.
- **`model_options` in an experiment config**, passing settings straight through
  to the model for knobs that belong to one model rather than all of them.
  Options the chosen model does not accept raise rather than silently doing
  nothing.
- **Five new model scenarios**, each testing one hypothesis about what earns
  votes: `past-polling` (previous seasons' votes), `recent-form` (a lagged
  rolling rating), `interactions` (products of paired features and match-best
  flags), `midfield-focus` (ball-winning stats only) and `everything`.
- **History features** — prior-season votes, career votes and votes per game.
  Completed seasons only: Brownlow votes are not published until count night, so
  using within-season votes would leak information nobody has. Tested.
- **Form features** — a decaying average of recent matches, lagged one game.
  These may use earlier rounds of the same season, since match statistics are
  public immediately.
- **Interaction features** — configurable products of feature pairs, plus
  indicators for leading a match in a statistic. The model is linear in its
  features, so combinations have to be built explicitly.
- **A contested-work composite**, weighting hard-won possession above cheap
  possession. A proxy from public statistics — not SuperCoach or the AFL Player
  Ratings, both of which are proprietary and absent from this dataset.
- **`cross_validate`** in an experiment config: the run also scores itself with
  walk-forward CV and stores the result in `metrics.json`.
- **The results page opens on the best model**, ranked by cross-validated top-3
  recall where available and the held-out season otherwise. This is not
  cosmetic: on the single 2025 holdout `recent-form` looks best, while over six
  folds it is indistinguishable from noise.
- `--detail-players` on `brownlow report`, capping how many players keep
  round-by-round detail, to hold the page size down as scenarios accumulate.

### Changed

- Cross-validation inside a run now uses every season with a complete result,
  not just the training window, so scenarios using history features are not
  handicapped by having their look-back period cut off.
- Run ordering and the "best" marker moved from `render_site` into
  `collect_runs`, so ranking is a property of collecting runs and is testable.

### Findings

Six-fold walk-forward CV (2020–2025) put `interactions` top on top-3 recall at
0.6544 against `rank-model`'s 0.6489. Paired across folds the gains are small:
`interactions` +0.55pp (t=2.61) and `past-polling` +0.16pp (t=2.74) are real but
marginal, while `everything` (+0.45pp, t=1.94) and `recent-form` (+0.11pp,
t=0.29) are not distinguishable from noise.

Prior polling correlates 0.29 with votes on its own — second only to disposals —
yet adds almost nothing to the model, because players who polled last year are
polling now largely *because they are still playing well*, which this match's
statistics already capture.

### Earlier in this release

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
