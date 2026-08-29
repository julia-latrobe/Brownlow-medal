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

Two corrections stop the result claiming more certainty than it has earned. Both
were measured on held-out seasons; see :data:`DEFAULT_TEMPERATURE` and
:data:`DEFAULT_PLAYER_SHOCK`. Neither touches expected votes or the order of the
leaderboard, which come from the exact marginals and never from a simulation.
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

#: How sharp the fitted probabilities are taken to be. Below 1.0 they are
#: softened. The model is confident in a way six held-out seasons do not
#: support, and softening its scores by this much fit those seasons better than
#: leaving them alone.
DEFAULT_TEMPERATURE = 0.8

#: How uncertain we are about the players themselves, in score units.
#:
#: Without this, every simulated season uses exactly the same estimate of how
#: good each player is, and only the umpires' choice varies. That treats the
#: model's read on a player as perfect. It is not: Matt Rowell was projected at
#: about nineteen votes for 2025 and polled thirty-nine, which was not bad luck
#: with the dice but a misjudged player, and the simulation had no way to
#: entertain the possibility.
#:
#: So each simulated season nudges every player up or down, held constant across
#: his own matches. Checked against 240 held-out player-seasons, the intervals
#: without it were far too tight -- a stated 95% range held the truth 84% of the
#: time, and a 50% range held it 46%. At 0.5 those become 96% and 58%. Fitted
#: independently against the seasons' actual medallists, the best value was also
#: in the 0.4 to 0.7 range.
DEFAULT_PLAYER_SHOCK = 0.5


def simulate_season(
    predictions: pd.DataFrame,
    n_simulations: int = 10_000,
    ineligible: Optional[Iterable[str]] = None,
    seed: Optional[int] = 0,
    player_column: str = "player",
    temperature: float = DEFAULT_TEMPERATURE,
    player_shock: float = DEFAULT_PLAYER_SHOCK,
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
    temperature:
        Softens the fitted probabilities. See :data:`DEFAULT_TEMPERATURE`.
        ``1.0`` takes them exactly as the model gives them.
    player_shock:
        How much the model's read on each player might be wrong, in score
        units. See :data:`DEFAULT_PLAYER_SHOCK`. ``0.0`` assumes it is exactly
        right, which is what makes the ranges too narrow.

    Returns
    -------
    A leaderboard with mean/median simulated votes, a credible interval, and
    the probability of winning or finishing top 5.

    Notes
    -----
    Neither correction changes ``expected_votes`` or the leaderboard order --
    those come from the exact Plackett-Luce marginals in
    :meth:`~brownlow.model.PlackettLuceModel.predict`, which no simulation
    touches. What they change is how wide the ranges are and how confident the
    probabilities, which is where the model was overstating itself.
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
    # Raising every probability to a power and renormalising is exactly a
    # softmax temperature, so this needs nothing but the probabilities
    # themselves. The renormalising can be skipped: Gumbel-top-k is unchanged by
    # adding the same constant to every player in a match.
    log_p = float(temperature) * np.log(
        np.maximum(df["p_3_votes"].to_numpy(dtype=float), 1e-300))
    sim_rows = np.arange(n_simulations)

    # One draw per player per simulated season, held constant across his own
    # matches -- a player we have misjudged is misjudged all year, not
    # independently in each game.
    shocks = (
        rng.normal(scale=float(player_shock),
                   size=(n_simulations, len(players))).astype(np.float32)
        if player_shock > 0.0 else None
    )

    for start, size in zip(index.starts, index.sizes):
        block = slice(start, start + size)
        weights = log_p[block]
        if shocks is not None:
            weights = weights[None, :] + shocks[:, player_codes[block]]
        # Gumbel-top-k sampling reproduces sequential draws without replacement.
        noisy = weights + rng.gumbel(size=(n_simulations, size))
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
