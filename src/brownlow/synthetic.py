"""Generate fake-but-realistic seasons.

Useful in two places: the test suite (which must run in CI without downloading
14 MB of real data), and as a way to try the pipeline end to end before you have
the real thing on disk.

Because we generate the votes *from* a known set of weights, we also know the
right answer -- which makes it possible to test that the model actually recovers
the signal rather than just running without crashing.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import numpy as np
import pandas as pd

TEAMS = [
    "Adelaide", "Brisbane Lions", "Carlton", "Collingwood", "Essendon", "Fremantle",
    "Geelong", "Gold Coast", "Greater Western Sydney", "Hawthorn", "Melbourne",
    "North Melbourne", "Port Adelaide", "Richmond", "St Kilda", "Sydney",
    "West Coast", "Western Bulldogs",
]

#: The "truth" used to award votes in synthetic data. The model should recover
#: roughly this ordering of importance.
TRUE_WEIGHTS = {
    "disposals": 0.10,
    "goals": 0.35,
    "contested_possessions": 0.06,
    "clearances": 0.08,
    "marks": 0.03,
    "tackles": 0.03,
    "win": 0.60,
}


def make_synthetic_seasons(
    seasons: Sequence[int] = (2020, 2021, 2022),
    matches_per_season: int = 40,
    players_per_team: int = 22,
    signal_strength: float = 2.5,
    seed: Optional[int] = 7,
) -> pd.DataFrame:
    """Build a tidy player-match frame with a known vote-generating process.

    Votes are drawn with a Plackett-Luce process from ``TRUE_WEIGHTS``, so every
    match gets exactly one 3, one 2 and one 1 -- the same structure as the real
    competition.

    ``signal_strength`` scales the true utilities against the Gumbel noise. The
    default is tuned so a well-fitted model picks the 3-vote getter about half
    the time, which is roughly what the real competition allows.
    """
    rng = np.random.default_rng(seed)
    rows = []

    for season in seasons:
        # Give each player a persistent ability so season totals mean something.
        squads = {
            team: [f"{team.split()[0]} Player {i:02d}" for i in range(players_per_team + 4)]
            for team in TEAMS
        }
        ability = {
            player: rng.normal(0, 1)
            for team in TEAMS
            for player in squads[team]
        }

        for match_number in range(matches_per_season):
            home, away = rng.choice(len(TEAMS), size=2, replace=False)
            home_team, away_team = TEAMS[home], TEAMS[away]
            home_score = int(rng.normal(85, 22))
            away_score = int(rng.normal(85, 22))
            if home_score == away_score:
                home_score += 1

            match_rows = []
            for team, is_home in ((home_team, 1.0), (away_team, 0.0)):
                players = rng.choice(squads[team], size=players_per_team, replace=False)
                team_score = home_score if is_home else away_score
                opp_score = away_score if is_home else home_score
                for player in players:
                    skill = ability[player]
                    disposals = max(0, rng.normal(18 + 5 * skill, 5))
                    contested = max(0, rng.normal(0.4 * disposals, 2))
                    match_rows.append(
                        {
                            "season": season,
                            "round": str(match_number % 23 + 1),
                            "round_number": float(match_number % 23 + 1),
                            "is_final": False,
                            "match_id": f"{season}-M{match_number:03d}-{home_team}-v-{away_team}",
                            "player": player,
                            "team": team,
                            "opponent": away_team if is_home else home_team,
                            "is_home": is_home,
                            "team_score": team_score,
                            "opp_score": opp_score,
                            "margin": team_score - opp_score,
                            "win": 1.0 if team_score > opp_score else 0.0,
                            "disposals": round(disposals),
                            "kicks": round(disposals * 0.6),
                            "handballs": round(disposals * 0.4),
                            "marks": round(max(0, rng.normal(4 + skill, 2))),
                            "goals": round(max(0, rng.normal(0.8 + 0.5 * skill, 1))),
                            "behinds": round(max(0, rng.normal(0.6, 0.8))),
                            "hit_outs": 0.0,
                            "tackles": round(max(0, rng.normal(3.5, 2))),
                            "rebounds_50": round(max(0, rng.normal(2, 1.5))),
                            "inside_50s": round(max(0, rng.normal(3, 2))),
                            "clearances": round(max(0, rng.normal(2 + skill, 1.5))),
                            "clangers": round(max(0, rng.normal(2.5, 1.5))),
                            "frees_for": round(max(0, rng.normal(1, 1))),
                            "frees_against": round(max(0, rng.normal(1, 1))),
                            "contested_possessions": round(contested),
                            "uncontested_possessions": round(max(0, disposals - contested)),
                            "contested_marks": round(max(0, rng.normal(0.5, 0.8))),
                            "marks_inside_50": round(max(0, rng.normal(0.7, 1))),
                            "one_percenters": round(max(0, rng.normal(2, 2))),
                            "bounces": 0.0,
                            "goal_assists": round(max(0, rng.normal(0.5, 0.8))),
                            "time_on_ground": float(min(100, max(40, rng.normal(82, 10)))),
                            "votes": 0.0,
                        }
                    )

            # Award 3-2-1 by the same sequential draw the real model assumes.
            utility = np.array(
                [sum(w * row[stat] for stat, w in TRUE_WEIGHTS.items()) for row in match_rows]
            )
            utility = utility - utility.mean()
            if utility.std() > 0:
                utility = utility / utility.std() * signal_strength
            noisy = utility + rng.gumbel(size=len(utility))
            for vote, position in zip((3.0, 2.0, 1.0), np.argsort(-noisy)[:3]):
                match_rows[position]["votes"] = vote

            rows.extend(match_rows)

    df = pd.DataFrame(rows)
    return df.sort_values(["season", "match_id", "player"]).reset_index(drop=True)
