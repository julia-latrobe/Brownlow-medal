"""Running and comparing model experiments.

The workflow this is built for
------------------------------
1. **Build** the model on past seasons.
2. **Hold out** the most recent completed season(s) and score against them, so
   you know what the model is worth before you trust it.
3. **Predict** the season in progress, where the votes are genuinely unknown.

Every run drops the same four files into ``data/output/<experiment name>/``:

``predictions.csv``
    One row per player per match: the vote probabilities and expected votes.
    The raw material -- everything else is an aggregate of this.
``leaderboard.csv``
    One row per player: projected total votes, and (for the rank model) the
    simulated probability of winning the medal.
``metrics.json``
    Holdout accuracy, plus the exact config used. Self-describing, so a run
    from six months ago is still interpretable.
``coefficients.csv``
    What the model learned, largest effect first.

Because an experiment is just a small JSON config, trying a new idea is a
one-file change you can put in a pull request, and comparing ideas is
:func:`compare_experiments` reading the metrics back off disk.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from brownlow.data import parse_spec
from brownlow.evaluate import (
    evaluate_season,
    leave_one_season_out_cv,
    rolling_origin_cv,
    summarise_cv,
)
from brownlow.features import FeatureConfig
from brownlow.model import PlackettLuceModel, WeightedLogisticModel
from brownlow.simulate import simulate_season

MODELS = {
    "plackett_luce": PlackettLuceModel,
    "logistic": WeightedLogisticModel,
}

_FRIENDLY_NAMES = {
    "plackett_luce": "Rank model (Plackett-Luce)",
    "logistic": "Weighted logistic",
}


def default_output_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "output"


@dataclass
class ExperimentConfig:
    """Everything that defines a run. Save it, diff it, put it in a PR."""

    name: str = "baseline"
    model: str = "plackett_luce"
    alpha: float = 1.0
    train_seasons: Any = "2015-2023"
    test_seasons: Any = "2024-2025"
    predict_seasons: Any = None
    n_simulations: int = 10_000
    seed: int = 0
    ineligible: List[str] = field(default_factory=list)
    compare_baseline: bool = True
    #: Short labels shown beside a player's name on the results page, e.g.
    #: ``{"Some Player": "Omitted"}``. Purely presentational -- annotations
    #: never affect the model, the projection or the simulation.
    annotations: Dict[str, str] = field(default_factory=dict)
    #: Also score the scenario with walk-forward cross-validation. Slower (one
    #: fit per fold), but a single held-out season is a noisy way to rank
    #: scenarios whose differences are small -- which these are. The results
    #: page ranks by this when it is present.
    cross_validate: bool = False
    cv_min_train_seasons: int = 7
    features: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def build_model(self):
        if self.model not in MODELS:
            raise ValueError(f"Unknown model {self.model!r}. Options: {sorted(MODELS)}")
        feature_config = FeatureConfig(**self.features) if self.features else FeatureConfig()
        return MODELS[self.model](alpha=self.alpha, feature_config=feature_config)

    def to_json(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))
        return path

    @classmethod
    def from_json(cls, path: Path) -> ExperimentConfig:
        return cls(**json.loads(Path(path).read_text()))


def _seasons(spec, available: Sequence[int]) -> List[int]:
    parsed = parse_spec(spec)
    if parsed is None:
        return sorted(int(s) for s in available)
    return sorted(int(s) for s in parsed if int(s) in set(int(a) for a in available))


def run_experiment(
    config: ExperimentConfig,
    df: pd.DataFrame,
    output_dir: Optional[Path] = None,
    simulate: bool = True,
) -> Dict[str, Any]:
    """Train, evaluate on the holdout, optionally predict an unscored season.

    Returns a dict of everything produced; also writes it to disk.
    """
    available = sorted(int(s) for s in df["season"].unique())
    # A null season list means "none", not "all" -- a config with no holdout
    # must not silently test on every season it can find.
    test_seasons = _seasons(config.test_seasons, available) if config.test_seasons else []
    predict_seasons = (
        _seasons(config.predict_seasons, available) if config.predict_seasons else []
    )
    if config.train_seasons:
        train_seasons = _seasons(config.train_seasons, available)
    else:
        train_seasons = sorted(set(available) - set(test_seasons) - set(predict_seasons))

    overlap = set(train_seasons) & set(test_seasons)
    if overlap:
        raise ValueError(f"Train and test seasons overlap: {sorted(overlap)}")
    if not train_seasons:
        raise ValueError("No training seasons available after filtering.")

    model = config.build_model()
    model.fit(df[df["season"].isin(train_seasons)])

    results: Dict[str, Any] = {
        "config": asdict(config),
        "train_seasons": train_seasons,
        "test_seasons": test_seasons,
        "predict_seasons": predict_seasons,
        "model": model,
    }

    # -- Step 2: the holdout ------------------------------------------
    if test_seasons:
        test_predictions = model.predict(df[df["season"].isin(test_seasons)])
        results["test_predictions"] = test_predictions
        results["holdout_metrics"] = evaluate_season(test_predictions)

        # Fit the simpler baseline on the same split so the report can show an
        # honest side-by-side rather than a number with nothing to compare to.
        if config.compare_baseline:
            comparison = {_FRIENDLY_NAMES.get(config.model, config.model):
                          results["holdout_metrics"]}
            other = "logistic" if config.model != "logistic" else "plackett_luce"
            try:
                baseline = MODELS[other](alpha=config.alpha)
                baseline.fit(df[df["season"].isin(train_seasons)])
                comparison[_FRIENDLY_NAMES.get(other, other)] = evaluate_season(
                    baseline.predict(df[df["season"].isin(test_seasons)])
                )
                results["comparison"] = comparison
            except Exception:  # a baseline failure must never sink the main run
                pass

    # -- Cross-validation, when the scenario asks for it ---------------
    if config.cross_validate:
        # Every season with a known result, not just the training window. The
        # extra early seasons matter: history features look back a season or
        # two, so cutting them off would quietly handicap the scenarios that
        # use them. The season being projected has no votes and is excluded.
        counted = df.groupby("season")["votes"].apply(lambda s: s.notna().all())
        scored_seasons = [
            int(season) for season, complete in counted.items()
            if complete and int(season) not in set(predict_seasons)
        ]
        cv_frame = df[df["season"].isin(scored_seasons)]

        def factory():
            return config.build_model()

        try:
            folds = rolling_origin_cv(
                cv_frame, factory, min_train_seasons=config.cv_min_train_seasons
            )
            results["cv_folds"] = folds
            results["cv_metrics"] = {
                **{k: float(v) for k, v in summarise_cv(folds).items()},
                "n_folds": int(len(folds)),
                "fold_seasons": [int(s) for s in folds["fold_test_season"]],
            }
        except ValueError as error:
            # Not enough seasons to fold. Say so rather than failing the run.
            results["cv_metrics"] = {"error": str(error)}

    # -- Step 3: the unknown season -----------------------------------
    if predict_seasons:
        future = df[df["season"].isin(predict_seasons)]
        predictions = model.predict(future)
        results["predictions"] = predictions
        leaderboard = model.season_totals(predictions)
        if simulate and "p_3_votes" in predictions.columns:
            simulated = simulate_season(
                predictions,
                n_simulations=config.n_simulations,
                ineligible=config.ineligible,
                seed=config.seed,
            )
            # Join on name and club together: joining on name alone would
            # give two players who share a name each other's numbers.
            join_keys = ["player"]
            if "team" in leaderboard.columns and "team" in simulated.columns:
                join_keys.append("team")
            leaderboard = leaderboard.merge(
                simulated.drop(columns=["rank"], errors="ignore")
                if "team" in join_keys
                else simulated.drop(columns=["rank", "team"], errors="ignore"),
                on=join_keys,
                how="left",
            ).sort_values("predicted_votes", ascending=False)
            leaderboard["rank"] = np.arange(1, len(leaderboard) + 1)
            results["simulation"] = simulated
        results["leaderboard"] = leaderboard.reset_index(drop=True)

    if output_dir is not False:
        results["output_dir"] = str(write_outputs(config, results, output_dir))
    return results


def write_outputs(
    config: ExperimentConfig,
    results: Dict[str, Any],
    output_dir: Optional[Path] = None,
) -> Path:
    """Write the four standard artefacts for a run."""
    root = Path(output_dir) if output_dir else default_output_dir()
    run_dir = root / config.name
    run_dir.mkdir(parents=True, exist_ok=True)

    model = results["model"]
    model.save(run_dir / "model.json")
    model.coefficient_table().to_csv(run_dir / "coefficients.csv", index=False)
    config.to_json(run_dir / "config.json")

    frame = results.get("predictions")
    if frame is None:
        frame = results.get("test_predictions")
    if frame is not None:
        columns = [
            c
            for c in (
                "season", "round", "match_id", "date", "local_start_time",
                "venue", "player", "team", "opponent", "is_home", "votes",
                "predicted_votes", "p_3_votes", "p_2_votes",
                "p_1_vote", "p_any_votes", "score",
            )
            if c in frame.columns
        ]
        frame[columns].to_csv(run_dir / "predictions.csv", index=False)

    leaderboard = results.get("leaderboard")
    if leaderboard is None and "test_predictions" in results:
        leaderboard = model.season_totals(results["test_predictions"])
    if leaderboard is not None:
        leaderboard.to_csv(run_dir / "leaderboard.csv", index=False)

    metrics = {
        "name": config.name,
        "config": results["config"],
        "train_seasons": results["train_seasons"],
        "test_seasons": results["test_seasons"],
        "predict_seasons": results["predict_seasons"],
        "holdout_metrics": results.get("holdout_metrics"),
        "cv_metrics": results.get("cv_metrics"),
        "comparison": results.get("comparison"),
        "optimisation": getattr(model, "optimisation_", None),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    return run_dir


def compare_experiments(output_dir: Optional[Path] = None) -> pd.DataFrame:
    """Read every run's metrics.json back off disk into one comparison table.

    This is the iteration loop: change a config, rerun, call this, see whether
    the number moved.
    """
    root = Path(output_dir) if output_dir else default_output_dir()
    rows: List[Dict[str, Any]] = []
    for metrics_file in sorted(root.glob("*/metrics.json")):
        payload = json.loads(metrics_file.read_text())
        holdout = payload.get("holdout_metrics") or {}
        config = payload.get("config", {})
        rows.append(
            {
                "experiment": payload.get("name", metrics_file.parent.name),
                "model": config.get("model"),
                "alpha": config.get("alpha"),
                "train": config.get("train_seasons"),
                "test": config.get("test_seasons"),
                **{
                    k: holdout.get(k)
                    for k in (
                        "top1_accuracy",
                        "top3_recall",
                        "exact_order_accuracy",
                        "spearman",
                        "top5_overlap",
                        "winner_correct",
                        "log_likelihood_per_match",
                    )
                },
                "notes": config.get("notes", ""),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("top3_recall", ascending=False).reset_index(drop=True)


def tune_alpha(
    df: pd.DataFrame,
    alphas: Sequence[float] = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0),
    model: str = "plackett_luce",
    method: str = "rolling",
    min_train_seasons: int = 4,
    features: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Grid-search the regularisation strength using season-wise CV."""
    feature_config = FeatureConfig(**features) if features else FeatureConfig()
    rows = []
    for alpha in alphas:
        def factory(alpha=alpha):
            return MODELS[model](alpha=alpha, feature_config=feature_config)

        folds = (
            rolling_origin_cv(df, factory, min_train_seasons=min_train_seasons)
            if method == "rolling"
            else leave_one_season_out_cv(df, factory)
        )
        rows.append({"alpha": alpha, **summarise_cv(folds).to_dict()})
    return pd.DataFrame(rows).sort_values("top3_recall", ascending=False).reset_index(drop=True)
