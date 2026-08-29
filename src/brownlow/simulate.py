"""Monte Carlo simulation of a full Brownlow count.

Expected votes tell you who *should* poll most. They do not tell you the thing
everyone actually asks on count night: what are the odds this player wins?

Those are different questions. A player who is a steady 1.4 expected votes a
week can have a higher expected total but a lower chance of winning than a
volatile 3-or-nothing midfielder, because winning needs an outlier, not an
average. To answer it we simulate the whole season many times.

Each simulated match draws the 3-2-1 order from the fitted Plackett-Luce
probabilities, using the Gumbel top-k trick: adding independent Gumbel noise to
each player's log-probability and taking the top three is *exactly* equivalent
to drawing them one at a time without replacement.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Optional

import numpy as np
import pandas as pd

from brownlow.model import MatchIndex

#: Joins a player's name to their club to form a unique key. A control
#: character, so it cannot collide with anything in a real name.
_KEY_SEPARATOR = "\u241f"


def simulate_season(
    predictions: pd.DataFrame,
    n_simulations: int = 10_000,
    ineligible: Optional[Iterable[str]] = None,
    seed: Optional[int] = 0,
    player_column: str = "player",
) -> pd.DataFrame:
    """Simulate the season repeatedly and summarise each player's outcomes.

    Parameters
    ----------
    predictions:
        Output of :meth:`PlackettLuceModel.predict`; needs ``p_3_votes``
        (the within-match win probability) and ``match_id``.
    n_simulations:
        How many seasons to run. 10,000 is plenty for stable win probabilities.
    ineligible:
        Players who cannot win the medal (suspended during the season). They
        still poll votes -- the umpires do not know about the tribunal -- but
        they are excluded when deciding the winner of each simulated season.
    seed:
        Seed for reproducibility. Pass ``None`` for a different draw each run.

    Returns
    -------
    A leaderboard with mean/median simulated votes, a credible interval, and
    the probability of winning or finishing top 5.
    """
    required = {"p_3_votes", "match_id", player_column}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions is missing columns: {sorted(missing)}")

    df = predictions.sort_values("match_id", kind="stable").reset_index(drop=True)
    index = MatchIndex(df["match_id"].to_numpy())

    # Two different players can share a name -- the AFL has had two Bailey
    # Williamses at once -- so identity is the name *and* the club, never the
    # name alone. Pooling them would inflate one player's total and hand both
    # the same wrong win probability.
    identity = df[player_column].astype(str)
    has_team = "team" in df.columns
    if has_team:
        identity = identity + _KEY_SEPARATOR + df["team"].astype(str)
    player_codes, unique_keys = pd.factorize(identity, sort=True)

    if has_team:
        split = pd.Series(unique_keys).str.split(_KEY_SEPARATOR, n=1, expand=True)
        players = split[0].to_numpy()
        player_teams = split[1].to_numpy()
    else:
        players = np.asarray(unique_keys)
        player_teams = None

    totals = np.zeros((n_simulations, len(players)), dtype=np.float32)

    rng = np.random.default_rng(seed)
    log_p = np.log(np.maximum(df["p_3_votes"].to_numpy(dtype=float), 1e-300))
    sim_rows = np.arange(n_simulations)

    for start, size in zip(index.starts, index.sizes):
        block = slice(start, start + size)
        # Gumbel-top-k sampling reproduces sequential draws without replacement.
        noisy = log_p[block] + rng.gumbel(size=(n_simulations, size))
        top3 = np.argpartition(-noisy, kth=min(2, size - 1), axis=1)[:, :3]
        ordered = np.take_along_axis(top3, np.argsort(-np.take_along_axis(noisy, top3, 1), 1), 1)
        codes = player_codes[start:start + size]
        for position, vote_value in enumerate((3.0, 2.0, 1.0)):
            if position < ordered.shape[1]:
                totals[sim_rows, codes[ordered[:, position]]] += vote_value

    eligible = np.ones(len(players), dtype=bool)
    if ineligible:
        # Matched on name, so a shared name rules out both players. That errs
        # towards excluding someone who could win rather than crowning someone
        # who cannot, which is the safer way round.
        eligible &= ~np.isin(players, list(ineligible))

    contested = np.where(eligible, totals, -np.inf)
    best = contested.max(axis=1, keepdims=True)
    winners = contested == best
    # A tie is shared, so split the credit rather than giving it to whoever
    # happens to sort first.
    win_credit = winners / winners.sum(axis=1, keepdims=True)
    win_probability = win_credit.mean(axis=0)

    ranks = (-contested).argsort(axis=1).argsort(axis=1)
    top5_probability = (ranks < 5).mean(axis=0)

    summary = pd.DataFrame(
        {
            player_column: players,
            "mean_votes": totals.mean(axis=0),
            "median_votes": np.median(totals, axis=0),
            "p10_votes": np.percentile(totals, 10, axis=0),
            "p90_votes": np.percentile(totals, 90, axis=0),
            "win_probability": win_probability,
            "top5_probability": top5_probability,
            "eligible": eligible,
        }
    )

    if player_teams is not None:
        summary["team"] = player_teams

    summary = summary.sort_values(
        ["win_probability", "mean_votes"], ascending=False
    ).reset_index(drop=True)
    summary.insert(0, "rank", np.arange(1, len(summary) + 1))
    return summary
