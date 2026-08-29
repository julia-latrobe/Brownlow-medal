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

#: Pairs multiplied together to give the model combinations it cannot otherwise
#: express. The score is linear in its features, so "lots of the ball *and*
#: kicked goals" is a different claim from "lots of the ball" plus "kicked
#: goals", and only an explicit product can make it.
DEFAULT_INTERACTION_PAIRS = (
    ("disposals_z", "goals_z"),                 # did everything
    ("contested_possessions_z", "goals_z"),     # won it and used it
    ("clearances_z", "goals_z"),                # drove the game forward
    ("disposals_z", "win"),                     # dominance that mattered
    ("fantasy_points_z", "win"),
    ("goals_z", "margin_scaled"),               # goals in a tight game
    ("disposals_z", "margin_scaled"),
    ("hit_outs", "clearances_z"),               # a ruckman's game
    ("tackles_z", "contested_possessions_z"),   # two-way midfield work
)

#: Stats where topping the match is worth an explicit flag.
DEFAULT_MATCH_BEST_STATS = (
    "disposals",
    "goals",
    "contested_possessions",
    "clearances",
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
    #: Products of pairs of features -- see :data:`DEFAULT_INTERACTION_PAIRS`.
    include_interactions: bool = False
    interaction_pairs: Sequence[Sequence[str]] = ()
    #: Indicators for topping the match in a statistic. The model is linear in
    #: its features, so "led the game for disposals" is something it cannot
    #: express from the raw count alone.
    include_match_best: bool = False
    #: Prior-season and career Brownlow polling. Needs a season or two of
    #: history loaded before the training window to be worth anything.
    include_history: bool = False
    #: A decaying average of recent performances, lagged by one match.
    include_form: bool = False
    form_halflife: float = 4.0
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

    # A contested-work score, in the spirit of the scoring systems that weight
    # hard-won possession above cheap possession. It is a proxy built from public
    # statistics -- SuperCoach's actual formula is proprietary and not in this
    # dataset, so this is not that number and should not be read as it.
    contested_weights = {
        "contested_possessions": 3.0,
        "clearances": 4.0,
        "contested_marks": 4.0,
        "tackles": 3.0,
        "goals": 6.0,
        "goal_assists": 3.0,
        "inside_50s": 2.0,
        "uncontested_possessions": 1.0,
        "clangers": -2.0,
    }
    if any(column in out.columns for column in contested_weights):
        score = np.zeros(len(out), dtype=float)
        for column, weight in contested_weights.items():
            if column in out.columns:
                score += weight * out[column].to_numpy(dtype=float)
        out["contested_score"] = score

    return out


def player_identity(df: pd.DataFrame) -> pd.Series:
    """A key that follows one player across seasons and club changes.

    Uses the AFL Tables player ID where it exists (it is missing on a handful of
    rows) and falls back to name plus club, which keeps two players who share a
    name apart.
    """
    fallback = df["player"].astype(str)
    if "team" in df.columns:
        fallback = fallback + "|" + df["team"].astype(str)
    if "player_id" not in df.columns:
        return fallback
    identifier = df["player_id"]
    return np.where(identifier.notna(), "id:" + identifier.astype(str), fallback)


def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """How much this player has polled in *previous* seasons.

    Prior polling is one of the strongest signals available: umpires reward the
    same kinds of players year after year, and a player who polled heavily last
    season is usually still in that role.

    **This looks at completed seasons only, never the current one.** Brownlow
    votes are not published until count night, so a model at round 12 does not
    know what anybody polled in round 11 of the same season. Using within-season
    votes would leak information nobody actually has. Performance-based form
    (see :func:`add_form_features`) is different, because those statistics are
    public the moment the game ends.
    """
    out = df.copy()
    if "votes" not in out.columns or "season" not in out.columns:
        return out

    out["_player_key"] = player_identity(out)

    per_season = (
        out.groupby(["_player_key", "season"], dropna=False)
        .agg(season_votes=("votes", "sum"), season_games=("votes", "size"))
        .reset_index()
        .sort_values(["_player_key", "season"])
    )

    grouped = per_season.groupby("_player_key", sort=False)
    # shift(1) is what makes this causal: season S sees up to S-1 and no further.
    per_season["prior_season_votes"] = grouped["season_votes"].shift(1).fillna(0.0)
    prior_games = grouped["season_games"].shift(1).fillna(0.0)
    per_season["prior_season_votes_per_game"] = np.where(
        prior_games > 0, per_season["prior_season_votes"] / prior_games.replace(0, np.nan), 0.0
    )
    per_season["career_votes_before"] = (
        grouped["season_votes"].cumsum() - per_season["season_votes"]
    )
    career_games = grouped["season_games"].cumsum() - per_season["season_games"]
    per_season["career_votes_per_game_before"] = np.where(
        career_games > 0,
        per_season["career_votes_before"] / career_games.replace(0, np.nan),
        0.0,
    )
    per_season["seasons_before"] = grouped.cumcount()
    per_season["has_polled_before"] = (per_season["career_votes_before"] > 0).astype(float)

    columns = [
        "_player_key", "season", "prior_season_votes", "prior_season_votes_per_game",
        "career_votes_before", "career_votes_per_game_before", "seasons_before",
        "has_polled_before",
    ]
    # Transforming an already-transformed frame is a reasonable thing for a
    # caller to do. Without this the merge would suffix the new columns _x/_y
    # and the features would silently go missing under their expected names.
    out = out.drop(columns=[c for c in columns if c != "_player_key" and c != "season"
                            and c in out.columns])
    # merge() returns a fresh RangeIndex. A left merge keeps row order, so the
    # caller's index can be put straight back -- and it must be, because callers
    # realign on it. Losing it here silently turns their scores into NaN.
    original_index = out.index
    out = out.merge(per_season[columns], on=["_player_key", "season"], how="left")
    out.index = original_index
    return out.drop(columns=["_player_key"])


#: Statistics whose recent average says something about a player's current form.
FORM_STATS = ("fantasy_points", "disposals", "contested_possessions", "goals")


def add_form_features(df: pd.DataFrame, halflife: float = 4.0) -> pd.DataFrame:
    """A rolling rating of how a player has been going lately.

    This is the "lagging" idea: rather than judging a game only on its own
    statistics, carry a decaying average of the player's recent matches. A
    halflife of four games means a match four rounds ago counts half as much as
    last week's.

    Every value is shifted by one match, so a row never sees its own game. Match
    statistics are public as soon as a game finishes, so unlike vote history this
    can legitimately use earlier rounds of the same season.

    This is a proxy built from public statistics. It is not the AFL's official
    Player Ratings, which are proprietary and not in this dataset.
    """
    out = add_derived_stats(df, FeatureConfig())
    if "season" not in out.columns:
        return out

    out["_player_key"] = player_identity(out)
    # Computing a rolling average needs the rows in career order, but callers
    # expect their rows back in the order they handed them over -- anything else
    # silently misaligns the features from the players they belong to. Remember
    # the incoming order and restore it before returning.
    out["_input_order"] = np.arange(len(out))
    order = ["_player_key", "season"]
    if "round_number" in out.columns:
        order.append("round_number")
    # match_id completes the ordering. Without it two rows for the same player in
    # the same round tie, and a tie is broken by whatever order the rows happened
    # to arrive in -- which makes the rolling average depend on the caller's row
    # order rather than on the player's career.
    if "match_id" in out.columns:
        order.append("match_id")
    out = out.sort_values(order, kind="stable")

    grouped = out.groupby("_player_key", sort=False)
    for stat in FORM_STATS:
        if stat not in out.columns:
            continue
        # shift(1) first, so the rating entering a match excludes that match.
        lagged = grouped[stat].shift(1)
        out[f"{stat}_form"] = (
            lagged.groupby(out["_player_key"])
            .transform(lambda s: s.ewm(halflife=halflife, ignore_na=True).mean())
            .fillna(0.0)
        )

    out["games_played_before"] = grouped.cumcount().astype(float)
    out = out.sort_values("_input_order")
    return out.drop(columns=["_player_key", "_input_order"])


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
        if config.include_form:
            out = add_form_features(out, halflife=config.form_halflife)
        if config.include_history:
            out = add_history_features(out)
        match_id = out["match_id"]
        names: List[str] = []

        for stat in config.base_stats:
            if stat in out.columns:
                names.append(stat)

        for stat in ("fantasy_points", "contested_score", "goal_involvements",
                     "clean_disposals", "free_kick_differential", "shots"):
            if stat in out.columns:
                names.append(stat)

        if config.include_form:
            for stat in FORM_STATS:
                if f"{stat}_form" in out.columns:
                    names.append(f"{stat}_form")
            if "games_played_before" in out.columns:
                names.append("games_played_before")

        if config.include_history:
            for stat in ("prior_season_votes", "prior_season_votes_per_game",
                         "career_votes_before", "career_votes_per_game_before",
                         "seasons_before", "has_polled_before"):
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

        # Match-best flags and interactions go last, because they are built from
        # the within-match columns created above.
        if config.include_match_best:
            for stat in DEFAULT_MATCH_BEST_STATS:
                if stat not in out.columns:
                    continue
                best = out.groupby(match_id)[stat].transform("max")
                column = f"{stat}_match_best"
                out[column] = ((out[stat] >= best) & (best > 0)).astype(float)
                names.append(column)

        if config.include_interactions:
            pairs = config.interaction_pairs or DEFAULT_INTERACTION_PAIRS
            for left, right in pairs:
                if left not in out.columns or right not in out.columns:
                    continue
                column = f"{left}_x_{right}"
                if column in names:
                    continue
                out[column] = (
                    out[left].to_numpy(dtype=float) * out[right].to_numpy(dtype=float)
                )
                names.append(column)

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
