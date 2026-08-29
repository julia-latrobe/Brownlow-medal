"""A guided tour of the model, one step at a time.

Run it as a script::

    python examples/quickstart.py

Or open it in VS Code and run it cell by cell: each ``# %%`` marks a cell, and
VS Code shows a "Run Cell" button above it. Output appears in an interactive
window beside the code, the way a notebook does -- but the file stays plain
Python, so it diffs cleanly in git.
"""

# %% Load the data
from brownlow import build_features, load_player_matches
from brownlow.evaluate import backtest, rolling_origin_cv, summarise_cv
from brownlow.model import PlackettLuceModel, WeightedLogisticModel
from brownlow.simulate import simulate_season

# Downloads about 14 MB the first time, then reads from the local cache.
# require_complete_votes drops any match without a full 3-2-1 result.
history = load_player_matches(seasons="2015-2025", require_complete_votes=True)

print(f"{len(history):,} player-match rows")
print(f"{history['match_id'].nunique():,} matches")
print(history.groupby("season")["votes"].sum().rename("votes awarded"))

# %% Look at what a row actually contains
columns = ["season", "round", "player", "team", "disposals", "goals", "margin", "votes"]
print(history[history["votes"] == 3][columns].head(10).to_string(index=False))

# %% Build the features
# Nothing is learned here -- it is a deterministic transformation, so features
# for a future round are built exactly the same way as for history.
featured = build_features(history)
print(f"{len(featured.attrs['feature_names'])} features")
print(featured.attrs["feature_names"])

# %% Step 1 and 2: train on the past, score on a season held out
result = backtest(
    history,
    PlackettLuceModel(alpha=1.0),
    train_seasons=range(2015, 2024),
    test_seasons=[2024, 2025],
)
for name, value in result["metrics"].items():
    print(f"{name:<30} {value}")

# %% Compare against the simpler approach
baseline = backtest(
    history,
    WeightedLogisticModel(alpha=1.0),
    train_seasons=range(2015, 2024),
    test_seasons=[2024, 2025],
)
print(f"{'metric':<24} {'rank model':>12} {'logistic':>12}")
for name in ("top1_accuracy", "top3_recall", "exact_order_accuracy", "spearman"):
    print(f"{name:<24} {result['metrics'][name]:>12.4f} {baseline['metrics'][name]:>12.4f}")

# %% What did the model learn?
print(result["model"].coefficient_table().head(15).to_string(index=False))

# %% Cross-validate properly -- one fold per season, never training on the future
folds = rolling_origin_cv(history, min_train_seasons=5)
fold_columns = ["fold_test_season", "top1_accuracy", "top3_recall", "spearman"]
print(folds[fold_columns].to_string(index=False))
print("\nAverage across folds:")
print(summarise_cv(folds).round(4).to_string())

# %% Step 3: predict a season whose votes nobody knows yet
current = load_player_matches(seasons="2026")

model = PlackettLuceModel(alpha=1.0).fit(history)
predictions = model.predict(current)

leaderboard = model.season_totals(predictions)
print(leaderboard.head(15).to_string(index=False))

# %% How likely is each player to actually win?
# Expected votes rank players by their average. Winning takes an outlier, so
# simulate the whole season repeatedly and count how often each player tops it.
simulated = simulate_season(predictions, n_simulations=10_000, seed=0)
print(
    simulated[simulated["win_probability"] > 0.001][
        ["player", "team", "mean_votes", "p10_votes", "p90_votes", "win_probability"]
    ].to_string(index=False)
)

# %% Suspended players poll votes but cannot win the medal
# Pass their names and the simulation redistributes the win probability.
eligible_only = simulate_season(
    predictions, n_simulations=10_000, ineligible=["Some Player"], seed=0
)
print(eligible_only.head(5).to_string(index=False))
