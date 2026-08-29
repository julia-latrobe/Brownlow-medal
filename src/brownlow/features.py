"""Turning raw match statistics into model features.

The guiding idea: umpires do not ask "was this a good game by historical
standards?", they ask "who were the best three players *on this ground today*?".
A wet Saturday where nobody breaks 25 disposals still awards 3-2-1. So alongside
the raw counting stats we build **within-match** features -- how far above the
rest of this game's field a player was -- which travel much better across eras,
venues and conditions than raw totals do.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd

#: AFL Fantasy scoring weights. A useful single summary of a player's game.
FANTASY_WEIGHTS = {
    "kicks": 3.0,
    "handballs": 2.0,
    "marks": 3.0,
    "tackles": 4.0,
    "frees_for": 1.0,
    "frees_against": -3.0,
    "goals": 6.0,
    "behinds": 1.0,
    "hit_outs": 1.0,
}

#: Stats worth expressing relative to the rest of the players in the same match.
DEFAULT_WITHIN_MATCH_STATS = (
    "disposals",
    "contested_possessions",
    "clearances",
    "marks",
    "tackles",
    "inside_50s",
    "goals",
    "fantasy_points",
)

#: Stats whose vote value depends on whether the team won. Umpires reward the
#: best player in a win far more readily than the same game in a loss, so these
#: get an explicit interaction term rather than relying on the model to infer it.
DEFAULT_WIN_INTERACTIONS = (
    "disposals",
    "goals",
    "contested_possessions",
    "fantasy_points",
)


@dataclass
class FeatureConfig:
    """Which features to build. Tweak this to experiment with the model."""

    base_stats: Sequence[str] = (
        "kicks",
        "handballs",
        "disposals",
        "marks",
        "goals",
        "behinds",
        "hit_outs",
        "tackles",
        "rebounds_50",
        "inside_50s",
        "clearances",
        "clangers",
        "frees_for",
        "frees_against",
        "contested_possessions",
        "uncontested_possessions",
        "contested_marks",
        "marks_inside_50",
        "one_percenters",
        "goal_assists",
        "time_on_ground",
    )
    within_match_stats: Sequence[str] = DEFAULT_WITHIN_MATCH_STATS
    win_interactions: Sequence[str] = DEFAULT_WIN_INTERACTIONS
    include_fantasy: bool = True
    include_team_context: bool = True
    include_shares: bool = True
    include_ranks: bool = True
    extra: List[str] = field(default_factory=list)


def add_derived_stats(df: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    """Add composite stats that are combinations of the raw counts."""
    out = df.copy()

    if config.include_fantasy:
        fantasy = np.zeros(len(out), dtype=float)
        for column, weight in FANTASY_WEIGHTS.items():
            if column in out.columns:
                fantasy += weight * out[column].to_numpy(dtype=float)
        out["fantasy_points"] = fantasy

    if {"goals", "goal_assists"}.issubset(out.columns):
        out["goal_involvements"] = out["goals"] + out["goal_assists"]
    if {"goals", "behinds"}.issubset(out.columns):
        out["shots"] = out["goals"] + out["behinds"]
        # Accuracy only means something for players who actually had shots.
        out["goal_accuracy"] = np.where(
            out["shots"] > 0, out["goals"] / out["shots"].replace(0, np.nan), 0.0
        )
    if {"frees_for", "frees_against"}.issubset(out.columns):
        out["free_kick_differential"] = out["frees_for"] - out["frees_against"]
    if {"disposals", "clangers"}.issubset(out.columns):
        out["clean_disposals"] = out["disposals"] - out["clangers"]

    return out


def _within_match_zscore(values: pd.Series, match_id: pd.Series) -> np.ndarray:
    """Standardise a stat against the other players in the same match."""
    grouped = values.groupby(match_id)
    mean = grouped.transform("mean")
    std = grouped.transform("std")
    # A match where everyone recorded the same value carries no signal.
    return ((values - mean) / std.replace(0, np.nan)).fillna(0.0).to_numpy(dtype=float)


def _within_match_rank(values: pd.Series, match_id: pd.Series) -> np.ndarray:
    """Percentile rank of a stat within its match, in [0, 1]."""
    return (
        values.groupby(match_id)
        .rank(pct=True, method="average")
        .fillna(0.5)
        .to_numpy(dtype=float)
    )


class FeatureBuilder:
    """Builds the model's design matrix from tidy player-match rows.

    Deterministic and stateless: the same row always produces the same features,
    and nothing is learned from the data here. That means you can build features
    for a future round exactly the same way you built them for history, with no
    risk of leaking information backwards.
    """

    def __init__(self, config: FeatureConfig | None = None) -> None:
        self.config = config or FeatureConfig()
        self.feature_names_: List[str] = []

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if "match_id" not in df.columns:
            raise ValueError("Input needs a 'match_id' column; use load_player_matches.")

        config = self.config
        out = add_derived_stats(df, config)
        match_id = out["match_id"]
        names: List[str] = []

        for stat in config.base_stats:
            if stat in out.columns:
                names.append(stat)

        for stat in ("fantasy_points", "goal_involvements", "clean_disposals",
                     "free_kick_differential", "shots"):
            if stat in out.columns:
                names.append(stat)

        if config.include_team_context:
            if "win" in out.columns:
                names.append("win")
            if "margin" in out.columns:
                # Raw margin is dominated by blowouts; the umpire's read of "this
                # team won" saturates long before a 100-point margin.
                out["margin_scaled"] = np.clip(out["margin"].to_numpy(dtype=float), -60, 60) / 60.0
                names.append("margin_scaled")
            if "is_home" in out.columns:
                names.append("is_home")

        for stat in config.within_match_stats:
            if stat not in out.columns:
                continue
            out[f"{stat}_z"] = _within_match_zscore(out[stat], match_id)
            names.append(f"{stat}_z")
            if config.include_ranks:
                out[f"{stat}_rank"] = _within_match_rank(out[stat], match_id)
                names.append(f"{stat}_rank")

        if config.include_shares and "team" in out.columns:
            team_key = out["match_id"].astype(str) + "|" + out["team"].astype(str)
            for stat in ("disposals", "goals", "contested_possessions"):
                if stat not in out.columns:
                    continue
                team_total = out[stat].groupby(team_key).transform("sum")
                out[f"{stat}_team_share"] = (
                    (out[stat] / team_total.replace(0, np.nan)).fillna(0.0)
                )
                names.append(f"{stat}_team_share")

        if "win" in out.columns:
            for stat in config.win_interactions:
                if stat not in out.columns:
                    continue
                out[f"{stat}_x_win"] = out[stat].to_numpy(dtype=float) * out["win"].to_numpy(
                    dtype=float
                )
                names.append(f"{stat}_x_win")

        for stat in config.extra:
            if stat in out.columns and stat not in names:
                names.append(stat)

        self.feature_names_ = list(dict.fromkeys(names))
        out[self.feature_names_] = out[self.feature_names_].astype(float).fillna(0.0)
        return out


def build_features(df: pd.DataFrame, config: FeatureConfig | None = None) -> pd.DataFrame:
    """Convenience wrapper around :class:`FeatureBuilder`.

    Returns the input frame with feature columns added. The list of feature
    names is stored on ``result.attrs["feature_names"]``.
    """
    builder = FeatureBuilder(config)
    out = builder.transform(df)
    out.attrs["feature_names"] = builder.feature_names_
    return out
