"""A model that remembers which players the umpires read differently.

Two players can produce the same line in the stats and poll very differently.
Some catch the eye -- they win the ball in traffic, they lift a game -- and
poll more than their numbers deserve. Others accumulate quietly and poll less.
Nothing in a box score separates them, so a model built only on box scores gets
the same players wrong the same way year after year.

That repeatability is measurable. Across 244 player-seasons of held-out
predictions, how far a player's total sat from his prediction one season
correlated +0.32 with the following season (p < 0.00001). Matt Rowell polled
more than predicted in every season he was in contention, by nine votes on
average; Rowan Marshall polled fewer in every one of his.

So this model fits the ordinary rank model first, then gives each player a
standing adjustment for how the umpires have actually treated him, and applies
a fraction of it to future seasons.

Two cautions, both deliberate:

*The adjustment is deliberately partial.* A player's past misses are a mix of
something real and plain luck, and only the real part should carry forward.
Both ``strength`` and the number of seasons behind an estimate hold it back, so
a player seen once barely moves and one seen six times moves most of the way.

*It learns from completed seasons only.* Votes are not published until the count,
so an adjustment for a season in progress would be reading an answer sheet
nobody has yet. The offsets come from the training seasons and stay fixed
across the season being predicted.

It is a bet that the umpires who voted before will vote the same way again.
Where the way votes are cast changes, that bet is weaker than the backtest
suggests, and the ordinary rank model is the more conservative choice.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from brownlow.features import FeatureConfig, player_identity
from brownlow.model import (
    BaseVoteModel,
    MatchIndex,
    PlackettLuceModel,
    allocate_votes,
    plackett_luce_marginals,
)

__all__ = ["PlayerAdjustedModel"]


class PlayerAdjustedModel(BaseVoteModel):
    """Rank model plus a standing per-player adjustment.

    Parameters
    ----------
    alpha:
        L2 regularisation for the underlying rank model.
    strength:
        How much of a player's measured adjustment to carry forward, before the
        further discount for how little we have seen of him. ``0.0`` reproduces
        the plain rank model exactly; ``1.0`` applies the estimate in full.
        The default of ``0.5`` was the best of the settings tried on held-out
        seasons, and the flatness of that comparison is itself a reason not to
        push it higher.
    prior_seasons:
        Governs how quickly a player earns his full adjustment. An estimate
        drawn from ``n`` seasons is scaled by ``n / (n + prior_seasons)``, so
        with the default of 2 a player seen once gets a third of it and one seen
        six times gets three quarters.
    rounds:
        Passes of the fitting loop. Each pass nudges every player's offset
        towards the votes he really polled and recomputes the effect, since
        raising one player in a match lowers everyone else in it.
    max_offset:
        A ceiling on any single adjustment, in score units. Guards against a
        player with one freak season being handed an implausible standing bonus.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        strength: float = 0.5,
        prior_seasons: float = 2.0,
        rounds: int = 8,
        max_offset: float = 1.5,
        max_iter: int = 500,
        feature_config: Optional[FeatureConfig] = None,
    ) -> None:
        super().__init__(feature_config)
        self.alpha = float(alpha)
        self.strength = float(strength)
        self.prior_seasons = float(prior_seasons)
        self.rounds = int(rounds)
        self.max_offset = float(max_offset)
        self.max_iter = int(max_iter)
        self.base_: Optional[PlackettLuceModel] = None
        self.offsets_: dict = {}
        self.seasons_seen_: dict = {}

    # -- fitting -------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> PlayerAdjustedModel:
        self.base_ = PlackettLuceModel(
            alpha=self.alpha, max_iter=self.max_iter,
            feature_config=self.feature_config,
        ).fit(df)
        # Mirror the base model's fitted state so save/load and the feature
        # table work the same way as for any other model here.
        self.feature_names_ = self.base_.feature_names_
        self.coefficients_ = self.base_.coefficients_
        self.scaler_ = self.base_.scaler_

        frame = df.sort_values("match_id", kind="stable")
        keys = pd.Series(player_identity(frame), index=frame.index).astype(str)
        codes, players = pd.factorize(keys.to_numpy(), sort=True)
        index = MatchIndex(frame["match_id"].to_numpy())
        base_scores = self.base_.predict_scores(frame)

        actual = np.bincount(codes, weights=frame["votes"].to_numpy(dtype=float),
                             minlength=len(players))
        games = np.bincount(codes, minlength=len(players)).astype(float)

        # Nudge each player until the model reproduces what he actually polled.
        # One pass is not enough: lifting a player takes probability from the
        # others in his matches, so the whole field has to settle.
        offsets = np.zeros(len(players), dtype=float)
        for _ in range(self.rounds):
            p3, p2, p1 = plackett_luce_marginals(base_scores + offsets[codes], index)
            expected = np.bincount(codes, weights=3.0 * p3 + 2.0 * p2 + p1,
                                   minlength=len(players))
            offsets = np.clip(offsets + (actual - expected) / np.maximum(games, 1.0),
                              -self.max_offset, self.max_offset)

        # Discount for how little we have seen: a player with one season behind
        # his estimate has barely earned it.
        if "season" in frame.columns:
            seasons = (pd.DataFrame({"code": codes, "season": frame["season"].to_numpy()})
                       .groupby("code")["season"].nunique()
                       .reindex(range(len(players))).fillna(0).to_numpy(dtype=float))
        else:
            seasons = np.ones(len(players), dtype=float)
        shrunk = offsets * self.strength * (seasons / (seasons + self.prior_seasons))

        self.offsets_ = {p: float(v) for p, v in zip(players, shrunk) if v != 0.0}
        self.seasons_seen_ = {p: int(n) for p, n in zip(players, seasons)}
        return self

    # -- prediction ----------------------------------------------------
    def _offsets_for(self, df: pd.DataFrame) -> np.ndarray:
        """A player never seen in training gets no adjustment, which is right:
        we have no evidence about how the umpires read him."""
        keys = pd.Series(player_identity(df), index=df.index).astype(str)
        return keys.map(self.offsets_).fillna(0.0).to_numpy(dtype=float)

    def predict_scores(self, df: pd.DataFrame) -> np.ndarray:
        if self.base_ is None:
            raise ValueError("Model is not fitted yet.")
        return self.base_.predict_scores(df) + self._offsets_for(df)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.base_ is None:
            raise ValueError("Model is not fitted yet.")
        prepared = df.sort_values("match_id", kind="stable")
        scores = self.base_.predict_scores(prepared) + self._offsets_for(prepared)
        index = MatchIndex(prepared["match_id"].to_numpy())
        p3, p2, p1 = plackett_luce_marginals(scores, index)

        out = prepared.copy()
        out["score"] = scores
        out["player_adjustment"] = self._offsets_for(prepared)
        out["p_3_votes"] = p3
        out["p_2_votes"] = p2
        out["p_1_vote"] = p1
        out["p_any_votes"] = p3 + p2 + p1
        out["expected_votes"] = 3.0 * p3 + 2.0 * p2 + 1.0 * p1
        out["predicted_votes"] = allocate_votes(scores, index)
        return out

    def predict_unadjusted(self, df: pd.DataFrame) -> pd.DataFrame:
        """What the model would say with the umpires' standing view removed.

        The pair of projections is the useful output. One says where a player
        lands on his statistics alone; the other says where he lands once how
        the umpires have actually treated him is allowed for. The gap between
        them is the part of his season that is reputation rather than output,
        and it is worth seeing as a movement rather than hidden inside a range.
        """
        if self.base_ is None:
            raise ValueError("Model is not fitted yet.")
        return self.base_.predict(df)

    def predict_at_strength(self, df: pd.DataFrame, strength: float) -> pd.DataFrame:
        """Predict with a different amount of the adjustment applied.

        No refitting: the measured gaps are already stored, so this only
        rescales them. ``0.0`` is the plain rank model, ``1.0`` the whole
        measured adjustment.
        """
        if self.base_ is None:
            raise ValueError("Model is not fitted yet.")
        if self.strength == 0.0 and strength != 0.0:
            raise ValueError(
                "This model was fitted with strength 0.0, so it measured no "
                "adjustments to rescale. Refit with a strength above zero."
            )
        keep = self.offsets_
        scale = 0.0 if self.strength == 0.0 else float(strength) / self.strength
        try:
            self.offsets_ = {k: v * scale for k, v in keep.items()}
            return self.predict(df)
        finally:
            self.offsets_ = keep

    def bias_band(self, df: pd.DataFrame) -> pd.DataFrame:
        """Where each player lands with none of the umpires' standing view
        applied, and with all of it.

        This is a different question from the usual range. The simulated range
        asks how the votes might fall if we have the players right. This asks
        what happens if we have the umpires wrong -- if a player who has been
        rewarded beyond his statistics for years either keeps being rewarded or
        stops. It is one-sided by nature: a player carrying a standing credit
        has room above him and little below, and one marked down has the
        reverse.

        For 2026 the distinction is live. The umpires now see the statistics
        before voting, which had never been true of any season the adjustment
        was measured on. Whether that leaves their habits intact, sharpens them
        or dissolves them is not knowable in advance, so both ends are shown
        rather than a single answer.
        """
        keys = [c for c in ("season", "player", "team")
                if c in df.columns or c == "player"]
        ends = {}
        for label, strength in (("none", 0.0), ("all", 1.0)):
            board = self.season_totals(self.predict_at_strength(df, strength))
            ends[label] = board.set_index(
                [c for c in keys if c in board.columns])["expected_votes"]
        low = np.minimum(ends["none"], ends["all"])
        high = np.maximum(ends["none"], ends["all"])
        out = pd.DataFrame({
            "expected_votes_on_statistics": ends["none"],
            "expected_votes_full_umpire_bias": ends["all"],
            "bias_low": low, "bias_high": high,
            "bias_votes": ends["all"] - ends["none"],
        }).reset_index()
        return out.sort_values("expected_votes_full_umpire_bias",
                               ascending=False).reset_index(drop=True)

    def bias_overlay(self, df: pd.DataFrame) -> pd.DataFrame:
        """Where each player lands before and after the umpires' standing view.

        Returns one row per player: the projection from his statistics alone,
        the projection with the adjustment applied, and the movement between
        them. Positive movement means the umpires have historically given this
        player more than his numbers earn.
        """
        plain = self.base_.season_totals(self.predict_unadjusted(df))
        adjusted = self.season_totals(self.predict(df))
        keys = [c for c in ("season", "player", "team") if c in plain.columns]
        merged = plain[keys + ["expected_votes"]].merge(
            adjusted[keys + ["expected_votes"]], on=keys,
            suffixes=("_on_statistics", "_with_umpire_bias"))
        merged["bias_votes"] = (merged["expected_votes_with_umpire_bias"]
                                - merged["expected_votes_on_statistics"])
        return merged.sort_values("expected_votes_with_umpire_bias",
                                  ascending=False).reset_index(drop=True)

    # -- inspection ----------------------------------------------------
    def adjustment_table(self) -> pd.DataFrame:
        """Who the model has learned to push up or down, biggest effect first.

        A positive adjustment means the umpires have consistently given this
        player more than his statistics alone would earn him.
        """
        table = pd.DataFrame({
            "player": list(self.offsets_),
            "adjustment": [self.offsets_[p] for p in self.offsets_],
        })
        table["seasons_seen"] = [self.seasons_seen_.get(p, 0) for p in table["player"]]
        table["direction"] = np.where(table["adjustment"] > 0,
                                      "polls above their statistics",
                                      "polls below their statistics")
        return (table.reindex(table["adjustment"].abs().sort_values(ascending=False).index)
                .reset_index(drop=True))

    # -- persistence ---------------------------------------------------
    def save(self, path: Path) -> Path:
        path = Path(path)
        super().save(path)
        payload = json.loads(path.read_text())
        payload.update({
            "model": type(self).__name__,
            "strength": self.strength,
            "prior_seasons": self.prior_seasons,
            "rounds": self.rounds,
            "max_offset": self.max_offset,
            "offsets": self.offsets_,
            "seasons_seen": self.seasons_seen_,
        })
        path.write_text(json.dumps(payload, indent=2))
        return path

    @classmethod
    def load(cls, path: Path) -> PlayerAdjustedModel:
        payload = json.loads(Path(path).read_text())
        model = super().load(path)
        model.offsets_ = {str(k): float(v) for k, v in payload.get("offsets", {}).items()}
        model.seasons_seen_ = {str(k): int(v)
                               for k, v in payload.get("seasons_seen", {}).items()}
        for field in ("strength", "prior_seasons", "rounds", "max_offset"):
            if payload.get(field) is not None:
                setattr(model, field, payload[field])
        # Rebuild the underlying rank model from the same saved coefficients.
        base = PlackettLuceModel(alpha=model.alpha)
        base.feature_names_ = model.feature_names_
        base.coefficients_ = model.coefficients_
        base.scaler_ = model.scaler_
        model.base_ = base
        return model
