"""Scoring the model honestly, and cross-validating it the right way.

A note on splits
----------------
Do **not** split this data randomly. Rows from the same match are not
independent -- if the 3-vote getter lands in train and the 2-vote getter in
test, the model has already seen the answer. And the game changes over time
(rule changes, interchange caps, the 2020 shortened quarters), so a model
validated on shuffled rows will flatter itself.

Every split here is therefore **by season**, and the cross-validation is
walk-forward: train on the past, test on the next season, roll forward. That is
the same shape as the real task -- fit on completed seasons, predict a season
you have not seen.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from brownlow.model import BaseVoteModel, MatchIndex, PlackettLuceModel, segment_softmax


def _top_k_by_match(df: pd.DataFrame, column: str, k: int = 3) -> pd.DataFrame:
    order = df.sort_values(["match_id", column], ascending=[True, False])
    return order.groupby("match_id").head(k)


def _ranking_column(predictions: pd.DataFrame, preferred: Optional[str] = None) -> str:
    """Which column orders the players within a match.

    Never the 3-2-1 allocation: everyone outside the top three shares a zero, so
    ranking on it would be settled by row order rather than by the model. The
    continuous score is the model's real opinion, so use that where it exists.
    """
    if preferred is not None:
        return preferred
    for column in ("score", "expected_votes", "predicted_votes"):
        if column in predictions.columns:
            return column
    raise ValueError("No column to rank players by.")


def match_metrics(predictions: pd.DataFrame, score_column: Optional[str] = None) -> Dict:
    """How well did we pick the vote-getters, match by match?

    ``top1_accuracy``
        Share of matches where our highest-rated player actually got the 3.
    ``top3_recall``
        Of the three players who polled, what share did we have in our top 3?
        This is the fairest single number: getting the right three players in
        the wrong order is nearly right, and this metric says so.
    ``exact_order_accuracy``
        Share of matches where we got all three players *and* their order right.
        This is deliberately brutal -- even excellent models sit low here.
    ``mean_absolute_error``
        On expected votes, which is the estimate; the hard allocation is scored
        by the ranking metrics above.
    """
    known = predictions[predictions["votes"].notna()].copy()
    if known.empty:
        raise ValueError("No matches with known votes to score against.")

    score_column = _ranking_column(known, score_column)
    error_column = "expected_votes" if "expected_votes" in known.columns else score_column

    known["predicted_rank"] = (
        known.groupby("match_id")[score_column].rank(ascending=False, method="first")
    )
    n_matches = known["match_id"].nunique()

    top1 = known[(known["predicted_rank"] == 1) & (known["votes"] == 3)]
    top3 = known[(known["predicted_rank"] <= 3) & (known["votes"] > 0)]

    exact = known[known["votes"] > 0].copy()
    exact["actual_rank"] = 4.0 - exact["votes"]
    exact_hits = (
        exact.assign(hit=exact["predicted_rank"] == exact["actual_rank"])
        .groupby("match_id")["hit"]
        .all()
    )

    return {
        "n_matches": int(n_matches),
        "top1_accuracy": float(len(top1) / n_matches),
        "top3_recall": float(len(top3) / (3 * n_matches)),
        "exact_order_accuracy": float(exact_hits.mean()),
        "mean_absolute_error": float((known[error_column] - known["votes"]).abs().mean()),
    }


def plackett_luce_log_likelihood(predictions: pd.DataFrame) -> Dict:
    """Held-out log-likelihood under the fitted model, per match.

    This is the strictest test: it asks whether the *probabilities* are right,
    not just the ordering. A useful reference point is the log-likelihood of
    guessing at random, which for a 44-player match is about -11.3.
    """
    if "score" not in predictions.columns:
        raise ValueError("Needs the 'score' column from PlackettLuceModel.predict.")
    df = predictions[predictions["votes"].notna()].sort_values("match_id", kind="stable")
    df = df.reset_index(drop=True)
    index = MatchIndex(df["match_id"].to_numpy())

    scores = df["score"].to_numpy(dtype=float)
    votes = df["votes"].to_numpy(dtype=float)
    available = np.ones(len(scores), dtype=bool)
    total = 0.0

    for value in (3.0, 2.0, 1.0):
        rows = np.flatnonzero(votes == value)
        if len(rows) != index.n_matches:
            raise ValueError("Every match must have exactly one 3, 2 and 1 vote getter.")
        _, log_norm = segment_softmax(scores, index, available)
        total += float(scores[rows].sum() - log_norm.sum())
        available[rows] = False

    n_random = float(np.mean(index.sizes))
    random_baseline = -(np.log(n_random) + np.log(n_random - 1) + np.log(n_random - 2))
    return {
        "log_likelihood": total,
        "log_likelihood_per_match": total / index.n_matches,
        "random_baseline_per_match": float(random_baseline),
    }


def season_metrics(predictions: pd.DataFrame, top_n: int = 10) -> Dict:
    """Did we get the *count* right -- the leaderboard, not the single match?

    Scored on expected votes rather than the 3-2-1 allocation. Expected votes
    keep the near-misses, so they order a season far better: a player who is
    runner-up in fifteen games has earned a high projection, while the
    allocation would give him nothing at all.
    """
    known = predictions[predictions["votes"].notna()]
    if known.empty:
        raise ValueError("No matches with known votes to score against.")

    column = "expected_votes" if "expected_votes" in known.columns else "predicted_votes"
    totals = (
        known.groupby("player")
        .agg(predicted=(column, "sum"), actual=("votes", "sum"))
        .reset_index()
    )
    predicted_order = totals.sort_values("predicted", ascending=False)
    actual_order = totals.sort_values("actual", ascending=False)

    predicted_winner = predicted_order.iloc[0]["player"]
    actual_winner = actual_order.iloc[0]["player"]
    actual_top = set(actual_order.head(top_n)["player"])
    predicted_top = set(predicted_order.head(top_n)["player"])

    return {
        "n_players": int(len(totals)),
        "spearman": float(spearmanr(totals["predicted"], totals["actual"]).statistic),
        "kendall_tau": float(kendalltau(totals["predicted"], totals["actual"]).statistic),
        "winner_correct": bool(predicted_winner == actual_winner),
        "predicted_winner": str(predicted_winner),
        "actual_winner": str(actual_winner),
        f"top{top_n}_overlap": float(len(actual_top & predicted_top) / top_n),
        "top5_overlap": float(
            len(
                set(actual_order.head(5)["player"]) & set(predicted_order.head(5)["player"])
            )
            / 5
        ),
        "season_total_mae": float((totals["predicted"] - totals["actual"]).abs().mean()),
    }


def team_leader_metrics(predictions: pd.DataFrame) -> Dict:
    """How often do we name the right club's leading vote-getter?

    A different and more forgiving question than picking the medallist: there are
    eighteen of them a season rather than one, and it is the market most often
    offered on a club-by-club basis.

    Ties in the real count are common -- two players finishing level tops their
    club together -- so naming either of them counts as correct.
    """
    known = predictions[predictions["votes"].notna()]
    if known.empty:
        raise ValueError("No matches with known votes to score against.")

    column = "expected_votes" if "expected_votes" in known.columns else "predicted_votes"
    keys = ["season", "team"] if "season" in known.columns else ["team"]
    totals = known.groupby(keys + ["player"]).agg(
        predicted=(column, "sum"), actual=("votes", "sum")
    ).reset_index()

    hits, ties, clubs = 0, 0, 0
    for _, group in totals.groupby(keys):
        if group["actual"].max() <= 0:
            continue  # a club nobody polled for gives nothing to be right about
        clubs += 1
        picked = group.loc[group["predicted"].idxmax(), "player"]
        leaders = set(group.loc[group["actual"] == group["actual"].max(), "player"])
        hits += picked in leaders
        ties += len(leaders) > 1

    return {
        "team_leader_accuracy": float(hits / clubs) if clubs else float("nan"),
        "team_leaders_scored": int(clubs),
        "team_leader_ties": int(ties),
    }


def declared_winner_round(votes_by_round: pd.DataFrame,
                          rounds_remaining: Optional[pd.Series] = None) -> Optional[int]:
    """The round after which the leader can no longer be caught.

    ``votes_by_round`` is players down the rows and rounds across the columns,
    holding the votes polled in each. The count is revealed round by round, so
    after round R everyone still has at most three votes per round left to come.
    The winner is declared at the first R where the leader's total exceeds every
    other player's best possible finish.

    Returns ``None`` when the season ends without that ever being true -- which
    is what a count that goes to the last round looks like.
    """
    if votes_by_round.empty:
        return None
    cumulative = votes_by_round.cumsum(axis=1)
    rounds = list(votes_by_round.columns)

    for position, round_label in enumerate(rounds):
        standing = cumulative[round_label]
        leader = standing.max()
        left = len(rounds) - position - 1
        if left == 0:
            # The final round: the leader is declared if nobody is level.
            return round_label if (standing == leader).sum() == 1 else None
        # Everyone else can still add three votes per remaining round.
        best_possible = standing + 3 * left
        rivals = best_possible.drop(standing.idxmax())
        if leader > rivals.max():
            return round_label
    return None


def evaluate_season(predictions: pd.DataFrame, top_n: int = 10) -> Dict:
    """All of the above in one dictionary, ready to write to metrics.json."""
    metrics: Dict = {}
    metrics.update(match_metrics(predictions))
    metrics.update(season_metrics(predictions, top_n=top_n))
    if "team" in predictions.columns:
        metrics.update(team_leader_metrics(predictions))
    if "score" in predictions.columns:
        try:
            metrics.update(plackett_luce_log_likelihood(predictions))
        except ValueError:
            pass
    return metrics


def backtest(
    df: pd.DataFrame,
    model: Optional[BaseVoteModel] = None,
    train_seasons: Optional[Iterable[int]] = None,
    test_seasons: Optional[Iterable[int]] = None,
) -> Dict:
    """Fit on ``train_seasons``, score on ``test_seasons``.

    Returns a dict with the fitted ``model``, the test-set ``predictions`` and
    the ``metrics``. The test seasons are never touched during fitting.
    """
    model = model or PlackettLuceModel()
    seasons = sorted(df["season"].unique())
    if train_seasons is None or test_seasons is None:
        *train_default, test_default = seasons
        train_seasons = train_seasons or train_default
        test_seasons = test_seasons or [test_default]

    train_set, test_set = set(train_seasons), set(test_seasons)
    overlap = train_set & test_set
    if overlap:
        raise ValueError(f"Train and test seasons overlap: {sorted(overlap)}")

    train_df = df[df["season"].isin(train_set)]
    test_df = df[df["season"].isin(test_set)]
    if train_df.empty or test_df.empty:
        raise ValueError("Train or test split is empty -- check the season lists.")

    model.fit(train_df)
    predictions = model.predict(test_df)
    return {
        "model": model,
        "predictions": predictions,
        "metrics": evaluate_season(predictions),
        "train_seasons": sorted(train_set),
        "test_seasons": sorted(test_set),
    }


def rolling_origin_cv(
    df: pd.DataFrame,
    model_factory: Callable[[], BaseVoteModel] = PlackettLuceModel,
    min_train_seasons: int = 4,
    test_seasons: Optional[Sequence[int]] = None,
    expanding: bool = True,
) -> pd.DataFrame:
    """Walk-forward cross-validation, one fold per season.

    Fold 1 trains on the first ``min_train_seasons`` and tests on the next.
    Fold 2 rolls forward a season, and so on. With ``expanding=True`` the
    training window grows; with ``False`` it slides at a fixed width.

    This is the cross-validation to trust for this problem -- it never lets the
    model see the future, which a shuffled K-fold quietly does.
    """
    seasons = sorted(int(s) for s in df["season"].unique())
    if len(seasons) <= min_train_seasons:
        raise ValueError(
            f"Need more than {min_train_seasons} seasons to cross-validate; got {len(seasons)}."
        )
    candidates = seasons[min_train_seasons:]
    if test_seasons is not None:
        candidates = [s for s in candidates if s in set(test_seasons)]

    rows: List[Dict] = []
    for season in candidates:
        position = seasons.index(season)
        train = seasons[:position] if expanding else seasons[position - min_train_seasons:position]
        result = backtest(df, model_factory(), train_seasons=train, test_seasons=[season])
        rows.append(
            {"fold_test_season": season, "n_train_seasons": len(train), **result["metrics"]}
        )
    return pd.DataFrame(rows)


def leave_one_season_out_cv(
    df: pd.DataFrame,
    model_factory: Callable[[], BaseVoteModel] = PlackettLuceModel,
) -> pd.DataFrame:
    """Hold out each season in turn, training on all the others.

    More folds than :func:`rolling_origin_cv`, so it is the steadier signal for
    tuning a hyperparameter like ``alpha``. It does leak the future into the
    past, though, so never quote it as the model's expected live accuracy --
    use walk-forward for that.
    """
    seasons = sorted(int(s) for s in df["season"].unique())
    rows: List[Dict] = []
    for season in seasons:
        train = [s for s in seasons if s != season]
        result = backtest(df, model_factory(), train_seasons=train, test_seasons=[season])
        rows.append({"fold_test_season": season, **result["metrics"]})
    return pd.DataFrame(rows)


def summarise_cv(folds: pd.DataFrame) -> pd.Series:
    """Average the numeric columns of a CV table into one row."""
    numeric = folds.select_dtypes(include=[np.number]).drop(columns=["fold_test_season"],
                                                            errors="ignore")
    return numeric.mean()
