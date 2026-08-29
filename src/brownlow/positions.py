"""Working out where a player plays, from what they do.

AFL Tables records no position, but a player's statistical profile gives it away
almost unambiguously: rucks contest the ball-ups, defenders rebound and spoil,
forwards take marks inside 50 and kick goals, midfielders win clearances and rack
up possessions.

This matters because the Brownlow is lopsided by position. Across 2015-2025
midfielders are 29% of the players and take 66% of the votes, while defenders are
36% of the players and take 9%. A midfielder is roughly seven times likelier to
poll than a defender, so "twenty disposals" means something completely different
depending on who recorded it -- which a model with no notion of position cannot
express.

The scores below are built from *ratios* rather than standardised counts, so a
player's classification does not depend on which other rows happen to be in the
frame. The same player gets the same position whether you classify one season or
twenty, which keeps training and prediction consistent.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

#: The four positions, in the order their dummy columns are emitted.
POSITIONS: Sequence[str] = ("MID", "FWD", "DEF", "RUC")

#: Hit-outs a game above which a player is a ruck, whatever else they do. Rucks
#: are the one role no other position ever imitates.
RUCK_HIT_OUTS = 8.0

#: Games a player needs in a season before their profile is trusted. Below this
#: they keep the neutral fallback rather than a label drawn from two matches.
MIN_GAMES_FOR_PROFILE = 4

_PROFILE_STATS = (
    "hit_outs", "clearances", "contested_possessions", "disposals", "inside_50s",
    "rebounds_50", "goals", "behinds", "marks_inside_50", "one_percenters", "tackles",
    "marks",
)


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float),
                     where=denominator > 0)


def position_scores(profile: pd.DataFrame) -> pd.DataFrame:
    """Score a player's average game against each of the four roles.

    ``profile`` holds per-game averages. Every term is a rate or a ratio, so the
    scores do not shift when the surrounding data changes.
    """
    out = pd.DataFrame(index=profile.index)
    disposals = profile["disposals"].to_numpy(dtype=float)
    involvement = np.maximum(disposals, 1.0)

    # Rucks: hit-outs are the giveaway and nothing else produces them.
    out["ruck"] = profile["hit_outs"].to_numpy(dtype=float) / 10.0

    # Forwards: score, and take the ball inside 50 rather than out of it.
    out["forward"] = (
        1.6 * profile["goals"].to_numpy(dtype=float)
        + 0.8 * profile["behinds"].to_numpy(dtype=float)
        + 1.2 * profile["marks_inside_50"].to_numpy(dtype=float)
        - 0.8 * profile["rebounds_50"].to_numpy(dtype=float)
        - 0.5 * profile["clearances"].to_numpy(dtype=float)
    )

    # Defenders: rebound, spoil, and rarely go forward or score.
    out["defence"] = (
        1.3 * profile["rebounds_50"].to_numpy(dtype=float)
        + 0.5 * profile["one_percenters"].to_numpy(dtype=float)
        - 0.9 * profile["inside_50s"].to_numpy(dtype=float)
        - 1.6 * profile["goals"].to_numpy(dtype=float)
        - 0.5 * profile["clearances"].to_numpy(dtype=float)
    )

    # Midfielders: win the contested ball and move it, in volume.
    out["midfield"] = (
        1.5 * profile["clearances"].to_numpy(dtype=float)
        + 12.0 * _safe_ratio(profile["contested_possessions"].to_numpy(dtype=float),
                             involvement)
        + 0.10 * disposals
        + 0.3 * profile["inside_50s"].to_numpy(dtype=float)
        - 3.0
    )
    return out


def season_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """Average each player's season, one row per player, club and season."""
    columns = [c for c in _PROFILE_STATS if c in df.columns]
    grouped = df.groupby(["player", "team", "season"], dropna=False)
    profile = grouped[columns].mean()
    profile["games"] = grouped.size()
    return profile.reset_index()


def assign_positions(df: pd.DataFrame, leave_one_out: bool = True) -> pd.Series:
    """Give every row the position that player was filling that season.

    With ``leave_one_out`` the profile for a row excludes that row's own match.
    A player who happens to kick four goals in one game should not be reclassified
    as a forward *for that game* -- position is what they do across a season, and
    leaving the match out keeps the feature strictly independent of the game being
    predicted.
    """
    columns = [c for c in _PROFILE_STATS if c in df.columns]
    keys = ["player", "team", "season"]
    grouped = df.groupby(keys, dropna=False)

    totals = grouped[columns].transform("sum")
    counts = grouped[columns[0]].transform("size").to_numpy(dtype=float)

    if leave_one_out:
        remaining = np.maximum(counts - 1.0, 1.0)
        averages = pd.DataFrame(
            {c: (totals[c].to_numpy(dtype=float) - df[c].to_numpy(dtype=float))
             / remaining for c in columns},
            index=df.index,
        )
        # A player with a single game has nothing left over; use that game.
        single = counts <= 1
        for column in columns:
            averages.loc[single, column] = df.loc[single, column].to_numpy(dtype=float)
    else:
        averages = pd.DataFrame(
            {c: totals[c].to_numpy(dtype=float) / np.maximum(counts, 1.0)
             for c in columns},
            index=df.index,
        )

    scores = position_scores(averages)
    labels = {"ruck": "RUC", "forward": "FWD", "defence": "DEF", "midfield": "MID"}
    position = scores.idxmax(axis=1).map(labels)

    # Rucks are unmistakable, so hit-outs override the comparison entirely.
    position = position.mask(averages["hit_outs"] >= RUCK_HIT_OUTS, "RUC")

    # Too few games to judge: fall back to the commonest role rather than guess.
    position = position.mask(counts < MIN_GAMES_FOR_PROFILE, "MID")
    return pd.Series(position.to_numpy(), index=df.index, name="position")


def add_position_features(df: pd.DataFrame, stats: Sequence[str] = (),
                          leave_one_out: bool = True) -> pd.DataFrame:
    """Add the position, its dummy columns, and position-by-statistic products.

    The dummies let the model hold different baselines for each role. The
    products let it hold different *slopes*: twenty-five disposals from a
    defender is a remarkable game, and from a midfielder an ordinary one, and
    only an interaction can say both at once.
    """
    out = df.copy()
    out["position"] = assign_positions(out, leave_one_out=leave_one_out)

    for name in POSITIONS:
        out[f"is_{name.lower()}"] = (out["position"] == name).astype(float)

    for stat in stats:
        if stat not in out.columns:
            continue
        values = out[stat].to_numpy(dtype=float)
        for name in POSITIONS:
            out[f"{stat}_x_{name.lower()}"] = (
                values * out[f"is_{name.lower()}"].to_numpy(dtype=float)
            )
    return out
