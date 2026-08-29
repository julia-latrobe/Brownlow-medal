"""Combining the rank model with a gradient-boosted ranker.

Why this and not more features
------------------------------
Pooling eight variants of the linear model gained nothing, because they agreed
with each other: their within-match vote probabilities correlated 0.94 to 0.999,
so they made the same mistakes and averaging the same mistake changes nothing.

Two genuinely different model *classes* correlate about 0.96, and pooling them
does help. Across six walk-forward folds this beat the linear model alone on
top-3 recall in all six, gaining about 0.9 of a percentage point -- the only
change tried on this project that cleared statistical significance.

The two members are good at different things, which is the point:

* :class:`~brownlow.model.PlackettLuceModel` produces genuinely calibrated
  within-match probabilities, which the season simulation depends on. Its score
  is a straight weighted sum, so it cannot express a combination.
* A gradient-boosted ranker searches out combinations and thresholds by itself
  -- "plenty of the ball *and* on the winning side" -- but its raw output is not
  a probability of anything.

Pooling is done on the log of the probabilities (a geometric mean), which is the
natural combination for softmax models and keeps the result sharp; a plain
average flattens it.

LightGBM is an optional dependency. Install it with ``pip install
'brownlow[boost]'``; without it this model raises a clear error and every other
part of the project carries on unaffected.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from brownlow.features import FeatureBuilder, FeatureConfig
from brownlow.model import (
    BaseVoteModel,
    MatchIndex,
    PlackettLuceModel,
    allocate_votes,
    plackett_luce_marginals,
    segment_softmax,
)

_EPS = 1e-300


def _require_lightgbm():
    try:
        import lightgbm
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "The ensemble needs LightGBM, which is an optional extra. "
            "Install it with:  pip install 'brownlow[boost]'"
        ) from exc
    return lightgbm


class EnsembleModel(BaseVoteModel):
    """The rank model and a gradient-boosted ranker, pooled.

    Parameters
    ----------
    alpha:
        Regularisation for the linear member.
    n_estimators, learning_rate, num_leaves, min_child_samples:
        The booster's shape. The defaults are deliberately small -- there are
        only about 2,000 matches a decade, and a bigger booster memorises them.
    weights:
        Relative weight of (linear, booster) when pooling. Equal by default;
        there is no evidence in the data for anything else.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        num_leaves: int = 15,
        min_child_samples: int = 40,
        weights: tuple = (0.5, 0.5),
        random_state: int = 0,
        feature_config: Optional[FeatureConfig] = None,
    ) -> None:
        super().__init__(feature_config)
        self.alpha = float(alpha)
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.num_leaves = int(num_leaves)
        self.min_child_samples = int(min_child_samples)
        total = sum(weights)
        self.weights = (weights[0] / total, weights[1] / total)
        self.random_state = int(random_state)
        self.linear_: Optional[PlackettLuceModel] = None
        self.booster_ = None

    # -- fitting -------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> EnsembleModel:
        lightgbm = _require_lightgbm()

        self.linear_ = PlackettLuceModel(alpha=self.alpha,
                                         feature_config=self.feature_config)
        self.linear_.fit(df)
        self.feature_names_ = list(self.linear_.feature_names_)
        self.coefficients_ = self.linear_.coefficients_
        self.scaler_ = self.linear_.scaler_

        builder = FeatureBuilder(self.feature_config)
        built = builder.transform(df).sort_values("match_id", kind="stable")
        # LightGBM's ranking objective needs the rows grouped by match, and the
        # size of each group in order.
        groups = built.groupby("match_id", sort=False).size().to_numpy()

        self.booster_ = lightgbm.LGBMRanker(
            objective="lambdarank",
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples,
            subsample=0.9,
            colsample_bytree=0.8,
            random_state=self.random_state,
            verbose=-1,
            # A 3-vote game is worth more than a 1-vote game, and the gains say
            # by how much. They rise faster than 3/2/1 because topping a match
            # is a far stronger statement than scraping into it.
            label_gain=[0, 1, 3, 7],
        )
        self.booster_.fit(built[self.feature_names_],
                          built["votes"].astype(int).to_numpy(),
                          group=groups)
        self.optimisation_ = {
            "linear": getattr(self.linear_, "optimisation_", None),
            "booster_trees": int(self.booster_.n_estimators_),
            "weights": list(self.weights),
        }
        return self

    # -- prediction ----------------------------------------------------
    def _member_probabilities(self, df: pd.DataFrame, index: MatchIndex):
        """Each member's within-match probability of taking the three votes.

        Reducing both to the same currency is what makes them poolable: the
        booster's raw output is not a probability, but a softmax over the match
        turns any score into one.
        """
        linear_scores = np.asarray(self.linear_.predict_scores(df), dtype=float)
        linear_p, _ = segment_softmax(linear_scores, index)

        builder = FeatureBuilder(self.feature_config)
        built = builder.transform(df)
        booster_scores = np.asarray(
            self.booster_.predict(built[self.feature_names_]), dtype=float)
        booster_p, _ = segment_softmax(booster_scores, index)
        return linear_p, booster_p

    def predict_scores(self, df: pd.DataFrame) -> np.ndarray:
        if self.booster_ is None:
            raise ValueError("Model is not fitted yet.")
        ordered = df.sort_values("match_id", kind="stable")
        index = MatchIndex(ordered["match_id"].to_numpy())
        linear_p, booster_p = self._member_probabilities(ordered, index)

        # Geometric mean of the two probabilities. In score space that is just a
        # weighted average of log-probabilities, which is what a softmax model
        # wants; the softmax downstream renormalises it.
        pooled = (self.weights[0] * np.log(np.maximum(linear_p, _EPS))
                  + self.weights[1] * np.log(np.maximum(booster_p, _EPS)))
        return pd.Series(pooled, index=ordered.index).reindex(df.index).to_numpy()

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.booster_ is None:
            raise ValueError("Model is not fitted yet.")
        prepared = df.sort_values("match_id", kind="stable")
        index = MatchIndex(prepared["match_id"].to_numpy())
        scores = np.asarray(self.predict_scores(prepared), dtype=float)
        p3, p2, p1 = plackett_luce_marginals(scores, index)

        out = prepared.copy()
        out["score"] = scores
        out["p_3_votes"], out["p_2_votes"], out["p_1_vote"] = p3, p2, p1
        out["p_any_votes"] = p3 + p2 + p1
        out["expected_votes"] = 3.0 * p3 + 2.0 * p2 + 1.0 * p1
        out["predicted_votes"] = allocate_votes(scores, index)
        return out

    # -- what it learned -----------------------------------------------
    def coefficient_table(self) -> pd.DataFrame:
        """The linear member's weights, with the booster's usage beside them.

        Only the linear half has coefficients in any meaningful sense. The
        booster contributes a split count instead, which says how often it found
        a feature worth branching on -- a different question, shown alongside so
        the two views can be compared.
        """
        table = self.linear_.coefficient_table()
        importance = pd.Series(self.booster_.feature_importances_,
                               index=self.feature_names_, name="booster_splits")
        table["booster_splits"] = table["feature"].map(importance).fillna(0).astype(int)
        return table

    # -- persistence ---------------------------------------------------
    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        booster_path = path.with_suffix(".booster.txt")
        self.booster_.booster_.save_model(str(booster_path))
        payload = {
            "model": type(self).__name__,
            "feature_names": self.feature_names_,
            "coefficients": np.asarray(self.coefficients_).tolist(),
            "scaler_mean": self.scaler_.mean_.tolist(),
            "scaler_scale": self.scaler_.scale_.tolist(),
            "alpha": self.alpha,
            "weights": list(self.weights),
            "booster_file": booster_path.name,
        }
        path.write_text(json.dumps(payload, indent=2))
        return path

    @classmethod
    def load(cls, path: Path) -> EnsembleModel:
        lightgbm = _require_lightgbm()
        path = Path(path)
        payload = json.loads(path.read_text())

        model = cls(alpha=payload.get("alpha", 1.0),
                    weights=tuple(payload.get("weights", (0.5, 0.5))))
        model.feature_names_ = payload["feature_names"]
        model.coefficients_ = np.asarray(payload["coefficients"], dtype=float)

        linear = PlackettLuceModel(alpha=model.alpha)
        linear.feature_names_ = list(model.feature_names_)
        linear.coefficients_ = model.coefficients_
        from brownlow.model import _StandardScaler

        scaler = _StandardScaler()
        scaler.mean_ = np.asarray(payload["scaler_mean"], dtype=float)
        scaler.scale_ = np.asarray(payload["scaler_scale"], dtype=float)
        linear.scaler_ = scaler
        model.linear_ = linear
        model.scaler_ = scaler

        booster_path = path.parent / payload["booster_file"]
        model.booster_ = lightgbm.LGBMRanker()
        model.booster_._Booster = lightgbm.Booster(model_file=str(booster_path))
        model.booster_.fitted_ = True
        model.booster_._n_features = len(model.feature_names_)
        return model


def lightgbm_available() -> bool:
    """Whether the optional booster dependency is installed."""
    try:
        import lightgbm  # noqa: F401
    except ImportError:
        return False
    return True


__all__: List[str] = ["EnsembleModel", "lightgbm_available"]
