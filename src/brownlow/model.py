"""Models for predicting Brownlow votes.

Why a rank model rather than plain logistic regression
------------------------------------------------------
A natural first attempt is: fit a logistic regression for "did this player
poll?", score every player, then hand 3-2-1 to the top three scores in each
match. That works, and :class:`WeightedLogisticModel` implements it (including
the usual trick of weighting a 3-vote game more heavily than a 1-vote game).

But it models the wrong thing in two ways:

1. **It scores players against the whole league, not against their own match.**
   Every match awards exactly 6 votes, whether it was a 150-point blowout or a
   wet 40-point slog. A global logistic model has to learn "25 disposals is
   good" as an absolute statement. What actually matters is being the best
   player *in this game*.
2. **The vote weighting has to be guessed.** Weighting 3-vote games more is a
   sensible hand-adjustment, but the size of the weight is arbitrary.

:class:`PlackettLuceModel` fixes both. It models what the umpires actually do:
they pick a best player from the 44 on the ground, then a second best from
those remaining, then a third. That is a *rank-ordered (exploded) conditional
logit*, also known as a Plackett-Luce model. Each player gets a latent quality
score ``v = x . beta``, and

    P(3 votes = a) = softmax over everyone in the match
    P(2 votes = b | a) = softmax over everyone except a
    P(1 vote  = c | a, b) = softmax over everyone except a and b

The relative importance of 3 versus 1 votes is not a knob -- it falls out of
that factorisation, because the 3-vote player has to beat the field three times
over while the 1-vote player only has to survive the last draw. The softmax
being *within match* also means anything constant across a match (the weather,
the umpires' generosity, the era) cancels out exactly.

The model has no intercept, for the same reason: a constant added to every
player's score in a match cancels in every softmax.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from brownlow.features import FeatureBuilder, FeatureConfig

_EPS = 1e-12


class MatchIndex:
    """Bookkeeping for "rows grouped into matches".

    The design matrix is one long table of players, but every likelihood term
    is computed within a match. This class holds the offsets that let us do
    per-match softmaxes on the flat array without a Python loop.
    """

    def __init__(self, match_ids: Sequence) -> None:
        codes, uniques = pd.factorize(pd.Series(match_ids), sort=False)
        if np.any(np.diff(codes) < 0):
            raise ValueError(
                "Rows must be grouped by match (all rows of a match contiguous). "
                "Sort by match_id first."
            )
        self.codes = codes
        self.match_ids = uniques
        self.n_matches = len(uniques)
        self.starts = np.concatenate(([0], np.flatnonzero(np.diff(codes)) + 1))
        self.sizes = np.diff(np.concatenate((self.starts, [len(codes)])))

    def __len__(self) -> int:
        return self.n_matches


def segment_softmax(values: np.ndarray, index: MatchIndex, mask: np.ndarray | None = None):
    """Softmax within each match, plus the log-sum-exp of each match.

    ``mask`` marks which rows are still in contention -- used to drop the
    3-vote player before computing the 2-vote probabilities, and so on.
    """
    masked = values if mask is None else np.where(mask, values, -np.inf)
    peak = np.maximum.reduceat(masked, index.starts)
    shifted = np.exp(masked - peak[index.codes])
    totals = np.add.reduceat(shifted, index.starts)
    probabilities = shifted / np.maximum(totals[index.codes], _EPS)
    return probabilities, peak + np.log(np.maximum(totals, _EPS))


def allocate_votes(scores: np.ndarray, index: MatchIndex) -> np.ndarray:
    """Hand out 3, 2, 1 and 0 to the players in each match, by rank.

    This is the model's actual call: the best-rated player in the game gets the
    3, the next the 2, the next the 1, everyone else nothing -- exactly the shape
    of a real umpire's card.

    It is not the same thing as expected votes, and the two answer different
    questions. Expected votes (``3*P(3) + 2*P(2) + P(1)``) is the better estimate
    of a season total, because it keeps the near-misses: a player who is second
    favourite in fifteen games is worth a lot of votes even if he never quite
    tops one. This allocation throws that away, and in exchange tells you who the
    model would actually name.
    """
    allocated = np.zeros(len(scores), dtype=float)
    for start, size in zip(index.starts, index.sizes):
        block = scores[start:start + size]
        top = np.argsort(-block)[:3]
        for rank, position in enumerate(top):
            allocated[start + position] = 3.0 - rank
    return allocated


class _StandardScaler:
    """Centre and scale features so the L2 penalty treats them comparably."""

    def fit(self, X: np.ndarray) -> _StandardScaler:
        self.mean_ = X.mean(axis=0)
        scale = X.std(axis=0)
        self.scale_ = np.where(scale < _EPS, 1.0, scale)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.scale_


class BaseVoteModel:
    """Shared plumbing: feature building, fitting checks, season aggregation."""

    def __init__(self, feature_config: Optional[FeatureConfig] = None) -> None:
        self.feature_config = feature_config or FeatureConfig()
        self.feature_names_: List[str] = []
        self.coefficients_: Optional[np.ndarray] = None

    # -- helpers -------------------------------------------------------
    def _prepare(self, df: pd.DataFrame, fit: bool):
        if "match_id" not in df.columns:
            raise ValueError("Input needs a 'match_id' column.")
        if not any(name in df.columns for name in ("disposals_z", "fantasy_points")):
            builder = FeatureBuilder(self.feature_config)
            df = builder.transform(df)
            names = builder.feature_names_
        else:
            names = list(df.attrs.get("feature_names", []))
            if not names:
                builder = FeatureBuilder(self.feature_config)
                df = builder.transform(df)
                names = builder.feature_names_
        if fit:
            self.feature_names_ = names
        missing = [n for n in self.feature_names_ if n not in df.columns]
        if missing:
            raise ValueError(f"Features missing from input: {missing}")
        df = df.sort_values("match_id", kind="stable")
        X = df[self.feature_names_].to_numpy(dtype=float)
        return df, X

    @staticmethod
    def _vote_row_indices(df: pd.DataFrame, index: MatchIndex):
        """Row positions of the 3, 2 and 1 vote getters in each match."""
        votes = df["votes"].to_numpy(dtype=float)
        picks = []
        for value in (3.0, 2.0, 1.0):
            rows = np.flatnonzero(votes == value)
            if len(rows) != index.n_matches:
                # Group positionally: the frame's index is the caller's, not a
                # fresh range, so aligning on it would not be safe here.
                counts = pd.Series(votes == value).groupby(
                    df["match_id"].to_numpy()).sum()
                bad = counts[counts != 1].index.tolist()[:5]
                raise ValueError(
                    f"Expected exactly one {int(value)}-vote player per match. "
                    f"{len(rows)} found across {index.n_matches} matches. "
                    f"First offending matches: {bad}. "
                    "Filter to complete matches with require_complete_votes=True."
                )
            picks.append(rows)
        return picks

    def predict_scores(self, df: pd.DataFrame) -> np.ndarray:
        """Each player's latent quality score, in the caller's own row order.

        Scoring sorts rows internally, so the result is realigned before it is
        returned. Getting that wrong hands back a correctly-computed array of
        scores attached to the wrong players, which is the sort of bug that
        looks like a bad model rather than a bad join.
        """
        raise NotImplementedError

    def _scores_in_input_order(self, df: pd.DataFrame, prepared: pd.DataFrame,
                               scores: np.ndarray) -> np.ndarray:
        """Put scores computed on the sorted frame back into the input's order."""
        return (
            pd.Series(scores, index=prepared.index)
            .reindex(df.index)
            .to_numpy(dtype=float)
        )

    # -- prediction ----------------------------------------------------
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return per-player predicted votes for every match in ``df``.

        Adds two columns, which answer different questions:

        ``predicted_votes``
            The model's actual call: 3, 2, 1 or 0, awarded by rank within the
            match, the way an umpire fills in a card.
        ``expected_votes``
            A real number between 0 and 3, being ``3*P(3) + 2*P(2) + P(1)``.
            The better estimate of a season total, because it keeps the
            near-misses that the hard allocation discards.

        Plus the probability of taking each vote count.
        """
        raise NotImplementedError

    def season_totals(self, predictions: pd.DataFrame, by: str = "player") -> pd.DataFrame:
        """Add up votes across a season into a leaderboard.

        Carries both totals. ``predicted_votes`` sums the 3-2-1 allocations, so
        it is a whole number and reads like a real count. ``expected_votes`` sums
        the expectations, so it is smoother and usually the better estimate --
        it credits the games a player nearly won votes in.

        Rows are ordered by expected votes, with the tally breaking ties. That
        ordering is the one the charts and the season simulation agree with --
        the simulated spread around a player is a spread around their
        *expectation*, so ranking by the hard allocation instead would put bars
        out of step with their own uncertainty bands. It is also the better
        estimate: against the real 2025 count it was about a vote closer per
        player across the top 20.

        Grouping keeps the season and the player's team alongside the name, so a
        multi-season frame produces one row per player per season rather than
        silently merging them.
        """
        keys = [by]
        if "season" in predictions.columns:
            keys.insert(0, "season")
        if by == "player" and "team" in predictions.columns:
            keys.append("team")

        aggregations = {"games": ("match_id", "count")}
        for column in ("predicted_votes", "expected_votes"):
            if column in predictions.columns:
                aggregations[column] = (column, "sum")
        grouped = predictions.groupby(keys, dropna=False).agg(**aggregations)

        if "votes" in predictions.columns and predictions["votes"].notna().any():
            grouped["actual_votes"] = predictions.groupby(keys, dropna=False)["votes"].sum()

        sort_by = [c for c in ("expected_votes", "predicted_votes") if c in grouped.columns]
        out = grouped.reset_index().sort_values(sort_by, ascending=False)
        out.insert(0, "rank", np.arange(1, len(out) + 1))
        return out.reset_index(drop=True)

    # -- persistence ---------------------------------------------------
    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": type(self).__name__,
            "feature_names": self.feature_names_,
            "coefficients": np.asarray(self.coefficients_).tolist(),
            "scaler_mean": self.scaler_.mean_.tolist(),
            "scaler_scale": self.scaler_.scale_.tolist(),
            "alpha": getattr(self, "alpha", None),
        }
        path.write_text(json.dumps(payload, indent=2))
        return path

    @classmethod
    def load(cls, path: Path) -> BaseVoteModel:
        payload = json.loads(Path(path).read_text())
        model = cls()
        model.feature_names_ = payload["feature_names"]
        model.coefficients_ = np.asarray(payload["coefficients"], dtype=float)
        scaler = _StandardScaler()
        scaler.mean_ = np.asarray(payload["scaler_mean"], dtype=float)
        scaler.scale_ = np.asarray(payload["scaler_scale"], dtype=float)
        model.scaler_ = scaler
        if payload.get("alpha") is not None:
            model.alpha = payload["alpha"]
        return model

    def coefficient_table(self) -> pd.DataFrame:
        """Fitted weights, largest absolute effect first.

        Features are standardised before fitting, so these are directly
        comparable: a coefficient of 0.4 moves a player's score twice as much
        per standard deviation as one of 0.2.
        """
        if self.coefficients_ is None:
            raise ValueError("Model is not fitted yet.")
        table = pd.DataFrame(
            {"feature": self.feature_names_, "coefficient": self.coefficients_}
        )
        table["abs_coefficient"] = table["coefficient"].abs()
        return table.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)


class PlackettLuceModel(BaseVoteModel):
    """Rank-ordered conditional logit over the players in each match.

    Parameters
    ----------
    alpha:
        L2 regularisation strength. Features are standardised first, so the
        same alpha means roughly the same thing regardless of the feature set.
    max_iter:
        Maximum L-BFGS iterations.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        max_iter: int = 500,
        feature_config: Optional[FeatureConfig] = None,
    ) -> None:
        super().__init__(feature_config)
        self.alpha = float(alpha)
        self.max_iter = int(max_iter)

    # -- fitting -------------------------------------------------------
    def _negative_log_likelihood(self, beta, X, index, picks):
        scores = X @ beta
        available = np.ones(len(scores), dtype=bool)
        total = 0.0
        gradient = np.zeros_like(beta)

        for rows in picks:
            probabilities, log_norm = segment_softmax(scores, index, available)
            total += scores[rows].sum() - log_norm.sum()
            gradient += X[rows].sum(axis=0) - probabilities @ X
            available[rows] = False

        penalty = self.alpha * float(beta @ beta)
        objective = -total + penalty
        return objective, -gradient + 2.0 * self.alpha * beta

    def fit(self, df: pd.DataFrame) -> PlackettLuceModel:
        """Fit by maximum likelihood on matches with a complete 3-2-1 result."""
        df, X = self._prepare(df, fit=True)
        if "votes" not in df.columns or df["votes"].isna().all():
            raise ValueError("Training data needs a 'votes' column with known results.")

        index = MatchIndex(df["match_id"].to_numpy())
        picks = self._vote_row_indices(df, index)

        self.scaler_ = _StandardScaler().fit(X)
        X_scaled = self.scaler_.transform(X)

        result = minimize(
            self._negative_log_likelihood,
            x0=np.zeros(X_scaled.shape[1]),
            args=(X_scaled, index, picks),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": self.max_iter},
        )
        self.coefficients_ = result.x
        self.optimisation_ = {
            "success": bool(result.success),
            "message": str(result.message),
            "iterations": int(result.nit),
            "log_likelihood": float(-(result.fun - self.alpha * float(result.x @ result.x))),
            "n_matches": index.n_matches,
            "n_players": int(len(df)),
        }
        return self

    # -- prediction ----------------------------------------------------
    def predict_scores(self, df: pd.DataFrame) -> np.ndarray:
        if self.coefficients_ is None:
            raise ValueError("Model is not fitted yet.")
        prepared, X = self._prepare(df, fit=False)
        scores = self.scaler_.transform(X) @ self.coefficients_
        return self._scores_in_input_order(df, prepared, scores)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Expected votes per player, using the exact Plackett-Luce marginals.

        For every player we work out the probability of taking 3, 2 and 1 votes
        and combine them into an expected vote count. These sum to exactly 6
        per match, the same as reality -- no simulation needed.
        """
        if self.coefficients_ is None:
            raise ValueError("Model is not fitted yet.")
        prepared, X = self._prepare(df, fit=False)
        scores = self.scaler_.transform(X) @ self.coefficients_
        index = MatchIndex(prepared["match_id"].to_numpy())
        probabilities, _ = segment_softmax(scores, index)

        p3 = probabilities
        p2 = np.zeros_like(p3)
        p1 = np.zeros_like(p3)

        for start, size in zip(index.starts, index.sizes):
            block = slice(start, start + size)
            p = probabilities[block]
            rest = np.maximum(1.0 - p, _EPS)

            # Second pick: someone else wins the first draw, then this player
            # wins the second from what is left.
            ratio = p / rest
            p2[block] = p * (ratio.sum() - ratio)

            # Third pick: sum over every ordered pair of earlier winners.
            pair_remaining = np.maximum(1.0 - p[:, None] - p[None, :], _EPS)
            g = (p[:, None] * p[None, :]) / (rest[:, None] * pair_remaining)
            np.fill_diagonal(g, 0.0)
            p1[block] = p * (g.sum() - g.sum(axis=1) - g.sum(axis=0))

        out = prepared.copy()
        out["score"] = scores
        out["p_3_votes"] = p3
        out["p_2_votes"] = p2
        out["p_1_vote"] = p1
        out["p_any_votes"] = p3 + p2 + p1
        # Two different answers, both worth having. See allocate_votes.
        out["expected_votes"] = 3.0 * p3 + 2.0 * p2 + 1.0 * p1
        out["predicted_votes"] = allocate_votes(scores, index)
        return out


class WeightedLogisticModel(BaseVoteModel):
    """The simpler baseline: weighted logistic regression, then rank the scores.

    This is the "score every player, give 3-2-1 to the top three" approach, with
    3-vote games weighted more heavily than 1-vote games during fitting. It is
    included as an honest comparison point for :class:`PlackettLuceModel` -- see
    ``brownlow backtest --compare``.

    Parameters
    ----------
    vote_weights:
        Sample weight applied to a player who polled 3, 2 and 1 votes.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        vote_weights: Sequence[float] = (3.0, 2.0, 1.0),
        max_iter: int = 500,
        feature_config: Optional[FeatureConfig] = None,
    ) -> None:
        super().__init__(feature_config)
        self.alpha = float(alpha)
        self.vote_weights = tuple(float(w) for w in vote_weights)
        self.max_iter = int(max_iter)

    def _objective(self, beta, X, y, weights):
        logits = X @ beta
        # log(1 + exp(z)) computed stably.
        log_terms = np.logaddexp(0.0, logits)
        loss = float(weights @ (log_terms - y * logits))
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        gradient = X.T @ (weights * (probabilities - y))
        penalty = self.alpha * float(beta[1:] @ beta[1:])
        gradient_penalty = np.concatenate(([0.0], 2.0 * self.alpha * beta[1:]))
        return loss + penalty, gradient + gradient_penalty

    def fit(self, df: pd.DataFrame) -> WeightedLogisticModel:
        df, X = self._prepare(df, fit=True)
        if "votes" not in df.columns or df["votes"].isna().all():
            raise ValueError("Training data needs a 'votes' column with known results.")

        votes = df["votes"].to_numpy(dtype=float)
        y = (votes > 0).astype(float)
        weights = np.ones_like(votes)
        for value, weight in zip((3.0, 2.0, 1.0), self.vote_weights):
            weights[votes == value] = weight

        self.scaler_ = _StandardScaler().fit(X)
        # Unlike the rank model, a global logistic regression does need an
        # intercept: it has to learn the base rate of polling at all.
        X_scaled = np.column_stack([np.ones(len(X)), self.scaler_.transform(X)])

        result = minimize(
            self._objective,
            x0=np.zeros(X_scaled.shape[1]),
            args=(X_scaled, y, weights),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": self.max_iter},
        )
        self.intercept_ = float(result.x[0])
        self.coefficients_ = result.x[1:]
        self.optimisation_ = {"success": bool(result.success), "iterations": int(result.nit)}
        return self

    def predict_scores(self, df: pd.DataFrame) -> np.ndarray:
        prepared, X = self._prepare(df, fit=False)
        scores = self.intercept_ + self.scaler_.transform(X) @ self.coefficients_
        return self._scores_in_input_order(df, prepared, scores)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        prepared, X = self._prepare(df, fit=False)
        scores = self.intercept_ + self.scaler_.transform(X) @ self.coefficients_
        index = MatchIndex(prepared["match_id"].to_numpy())

        out = prepared.copy()
        out["score"] = scores
        out["predicted_votes"] = allocate_votes(scores, index)
        # This model scores players against the league rather than against the
        # others in their match, so it has no per-match vote probabilities to
        # take an expectation over. Its expected votes are therefore just its
        # allocation -- reported for a consistent interface, not because the
        # model has anything smoother to say.
        out["expected_votes"] = out["predicted_votes"]
        out["p_any_votes"] = 1.0 / (1.0 + np.exp(-scores))
        return out

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": type(self).__name__,
            "feature_names": self.feature_names_,
            "coefficients": np.asarray(self.coefficients_).tolist(),
            "intercept": self.intercept_,
            "scaler_mean": self.scaler_.mean_.tolist(),
            "scaler_scale": self.scaler_.scale_.tolist(),
            "alpha": self.alpha,
        }
        path.write_text(json.dumps(payload, indent=2))
        return path

    @classmethod
    def load(cls, path: Path) -> WeightedLogisticModel:
        model = super().load(path)
        payload = json.loads(Path(path).read_text())
        model.intercept_ = float(payload["intercept"])
        return model
