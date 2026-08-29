"""Brownlow Medal vote prediction.

A small, dependency-light toolkit that predicts AFL Brownlow Medal votes from
player-by-match statistics.

The umpires award 3, 2 and 1 votes after every home-and-away match, so the
target is not "did this player poll?" but "who were the best three players in
this particular game, in order?". That is a *ranking within a match*, and the
model in :mod:`brownlow.model` is built to match that structure exactly.

Typical use::

    from brownlow import load_player_matches, build_features, PlackettLuceModel

    df = load_player_matches(seasons=range(2015, 2026))
    feats = build_features(df)
    model = PlackettLuceModel().fit(feats)
    votes = model.predict_expected_votes(feats)
"""

from brownlow.data import (
    FEATURE_STAT_COLUMNS,
    load_csv,
    load_player_matches,
)
from brownlow.evaluate import backtest, evaluate_season
from brownlow.features import build_features
from brownlow.model import PlackettLuceModel, WeightedLogisticModel
from brownlow.simulate import simulate_season

__version__ = "0.1.0"

__all__ = [
    "FEATURE_STAT_COLUMNS",
    "PlackettLuceModel",
    "WeightedLogisticModel",
    "backtest",
    "build_features",
    "evaluate_season",
    "load_csv",
    "load_player_matches",
    "simulate_season",
]
