"""Loading and tidying AFL player-by-match data.

Data source and credit
----------------------
All player statistics and Brownlow vote counts come from **AFL Tables**
(https://afltables.com). We read them through the public data mirror maintained
by the **fitzRoy** project (https://github.com/jimmyday12/fitzRoy), which
scrapes AFL Tables with permission and republishes a tidy snapshot at
https://github.com/jimmyday12/fitzroy_data on a scheduled job.

No fitzRoy source code is used here -- this module is an independent Python
reader for the dataset that project publishes. Please cite both AFL Tables and
fitzRoy if you use this data, and read their terms before redistributing it.
"""

from __future__ import annotations

import os
import shutil
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

#: The fitzRoy project's AFL Tables snapshot (an R ``.rda`` file, ~14 MB).
FITZROY_DATA_URL = (
    "https://raw.githubusercontent.com/jimmyday12/fitzroy_data/main/"
    "data-raw/afl_tables_playerstats/afldata.rda"
)

#: Round codes AFL Tables uses for finals. No Brownlow votes are awarded in
#: finals, so these matches are excluded from training and prediction.
FINALS_ROUNDS = frozenset({"EF", "QF", "SF", "PF", "GF"})

#: Raw AFL Tables column -> tidy snake_case name.
COLUMN_MAP = {
    "Season": "season",
    "Round": "round",
    "Date": "date",
    "Venue": "venue",
    "Player": "player",
    "First.name": "first_name",
    "Surname": "surname",
    "ID": "player_id",
    "Playing.for": "team",
    "Home.team": "home_team",
    "Away.team": "away_team",
    "Home.score": "home_score",
    "Away.score": "away_score",
    "Home.Away": "home_away",
    "Brownlow.Votes": "votes",
    "Kicks": "kicks",
    "Handballs": "handballs",
    "Disposals": "disposals",
    "Marks": "marks",
    "Goals": "goals",
    "Behinds": "behinds",
    "Hit.Outs": "hit_outs",
    "Tackles": "tackles",
    "Rebounds": "rebounds_50",
    "Inside.50s": "inside_50s",
    "Clearances": "clearances",
    "Clangers": "clangers",
    "Frees.For": "frees_for",
    "Frees.Against": "frees_against",
    "Contested.Possessions": "contested_possessions",
    "Uncontested.Possessions": "uncontested_possessions",
    "Contested.Marks": "contested_marks",
    "Marks.Inside.50": "marks_inside_50",
    "One.Percenters": "one_percenters",
    "Bounces": "bounces",
    "Goal.Assists": "goal_assists",
    "Time.on.Ground": "time_on_ground",
    "Substitute": "substitute",
    "Age": "age",
    "Career.Games": "career_games",
}

#: The raw per-match statistics the feature builder starts from.
FEATURE_STAT_COLUMNS: Sequence[str] = (
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
    "bounces",
    "goal_assists",
    "time_on_ground",
)

#: Columns every downstream function assumes exist after loading.
REQUIRED_COLUMNS: Sequence[str] = (
    "season",
    "round",
    "match_id",
    "player",
    "team",
    "votes",
    *FEATURE_STAT_COLUMNS,
)

_NUMERIC_COLUMNS = (
    *FEATURE_STAT_COLUMNS,
    "votes",
    "home_score",
    "away_score",
    "age",
    "career_games",
    "season",
)


def default_cache_dir() -> Path:
    """Where downloaded raw data is kept (override with ``BROWNLOW_DATA_DIR``)."""
    env = os.environ.get("BROWNLOW_DATA_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "data" / "raw"


def download_afltables(cache_dir: Optional[Path] = None, force: bool = False) -> Path:
    """Download the AFL Tables snapshot, returning the local path.

    The file is cached, so this is a no-op on subsequent calls unless
    ``force=True``.
    """
    cache_dir = Path(cache_dir) if cache_dir else default_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / "afldata.rda"

    if target.exists() and not force:
        return target

    tmp = target.with_suffix(".rda.part")
    with urllib.request.urlopen(FITZROY_DATA_URL, timeout=120) as response:  # noqa: S310
        with open(tmp, "wb") as handle:
            shutil.copyfileobj(response, handle)
    tmp.replace(target)
    return target


def _read_rda(path: Path) -> pd.DataFrame:
    try:
        import pyreadr
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "Reading the AFL Tables .rda snapshot needs pyreadr. "
            "Install it with:  pip install 'brownlow[data]'"
        ) from exc

    result = pyreadr.read_r(str(path))
    if not result:
        raise ValueError(f"No data frames found in {path}")
    return result[next(iter(result))]


def tidy(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert a raw AFL Tables frame into this project's tidy schema.

    Adds the match-level context the model needs: a stable ``match_id``, the
    player's own team score and margin, and a win indicator.
    """
    keep = {src: dst for src, dst in COLUMN_MAP.items() if src in raw.columns}
    df = raw[list(keep)].rename(columns=keep).copy()

    for column in _NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    # The upstream mirror occasionally has a blank Player/ID for a real player
    # (a join gap on their side), but always carries first name and surname.
    # Rebuild the name rather than dropping the game.
    if {"first_name", "surname"}.issubset(df.columns):
        rebuilt = (
            df["first_name"].fillna("").astype(str).str.strip()
            + " "
            + df["surname"].fillna("").astype(str).str.strip()
        ).str.strip()
        player = df["player"].astype("object") if "player" in df.columns else pd.Series(
            [None] * len(df), index=df.index, dtype="object"
        )
        player = player.where(player.notna() & player.astype(str).str.strip().ne(""))
        df["player"] = player.fillna(rebuilt.replace("", pd.NA))
    df["player"] = df["player"].astype("object").fillna("Unknown player").astype(str)

    df["round"] = df["round"].astype(str).str.strip()
    df["is_final"] = df["round"].isin(FINALS_ROUNDS)
    df["round_number"] = pd.to_numeric(df["round"], errors="coerce")

    # One row per player per match; the match key must be unique per fixture.
    df["match_id"] = (
        df["season"].astype(str)
        + "-R"
        + df["round"]
        + "-"
        + df["home_team"].astype(str)
        + "-v-"
        + df["away_team"].astype(str)
    )

    is_home = df["home_away"].astype(str).str.lower().eq("home")
    df["is_home"] = is_home.astype(float)
    df["team_score"] = np.where(is_home, df["home_score"], df["away_score"])
    df["opp_score"] = np.where(is_home, df["away_score"], df["home_score"])
    df["opponent"] = np.where(is_home, df["away_team"], df["home_team"])
    df["margin"] = df["team_score"] - df["opp_score"]
    df["win"] = np.sign(df["margin"]).clip(lower=0).astype(float)
    df.loc[df["margin"] == 0, "win"] = 0.5  # a draw is half a win

    if "substitute" in df.columns:
        df["was_subbed"] = df["substitute"].astype(str).str.strip().ne("").astype(float)

    for column in FEATURE_STAT_COLUMNS:
        if column in df.columns:
            df[column] = df[column].fillna(0.0)

    return df


def parse_spec(spec, cast=int):
    """Parse a flexible selection spec into a set of values, or None for "all".

    Accepts anything a caller or a command line is likely to hand us::

        parse_spec(None)              -> None          (no filter)
        parse_spec(2023)              -> {2023}
        parse_spec("2015-2018")       -> {2015, 2016, 2017, 2018}
        parse_spec("2021,2023")       -> {2021, 2023}
        parse_spec("2015-2017,2023")  -> {2015, 2016, 2017, 2023}
        parse_spec(range(2015, 2018)) -> {2015, 2016, 2017}
        parse_spec(["EF", "GF"], str) -> {"EF", "GF"}
    """
    if spec is None:
        return None
    if isinstance(spec, str):
        values = set()
        for chunk in spec.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            if "-" in chunk[1:] and cast is int:
                start, _, end = chunk.partition("-")
                values.update(range(int(start), int(end) + 1))
            else:
                values.add(cast(chunk))
        return values or None
    if isinstance(spec, (int, np.integer)):
        return {cast(spec)}
    return {cast(v) for v in spec} or None


def resolve_source(
    path: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
    download: bool = True,
    refresh: bool = False,
) -> Path:
    """Work out which file to read, downloading or refreshing the cache if asked."""
    if path is not None:
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"No such data file: {source}")
        return source

    cache = Path(cache_dir) if cache_dir else default_cache_dir()
    source = cache / "afldata.rda"
    if refresh:
        return download_afltables(cache, force=True)
    if source.exists():
        return source
    if not download:
        raise FileNotFoundError(
            f"{source} not found and download=False. Run `brownlow fetch` first."
        )
    return download_afltables(cache)


def load_player_matches(
    seasons=None,
    rounds=None,
    teams=None,
    players=None,
    home_and_away_only: bool = True,
    require_complete_votes: bool = False,
    columns: Optional[Sequence[str]] = None,
    path: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
    download: bool = True,
    refresh: bool = False,
) -> pd.DataFrame:
    """Load tidy player-by-match rows, downloading the data if needed.

    Every filter is optional; leave one as ``None`` to keep everything. The
    ``seasons``, ``rounds``, ``teams`` and ``players`` arguments all go through
    :func:`parse_spec`, so they accept a single value, a list, a ``range``, or a
    string like ``"2015-2025"``.

    Parameters
    ----------
    seasons:
        Seasons to keep, e.g. ``range(2015, 2026)``, ``2023`` or ``"2015-2025"``.
    rounds:
        Home-and-away round numbers to keep, e.g. ``"1-23"``.
    teams:
        Keep only rows for these teams (AFL Tables names, e.g. ``"Geelong"``).
        Note this filters *players*, not whole matches, so a match will come
        back half-populated -- useful for reporting, not for training.
    players:
        Keep only these player names.
    home_and_away_only:
        Drop finals. Brownlow votes are only awarded in home-and-away matches,
        so finals rows would train the model on games where nobody polled.
        Keep this True unless you know why you want otherwise.
    require_complete_votes:
        Keep only matches with a complete 3-2-1 result. Use this for training
        so in-progress seasons and bad rows are excluded automatically.
    columns:
        Return only these columns (plus the identifiers the model needs).
    path:
        Read this ``.rda`` file instead of the cached download.
    download:
        If False, never hit the network -- raise if the cache is missing.
    refresh:
        Re-download even if a cached copy exists. Use during a live season.
    """
    df = tidy(_read_rda(resolve_source(path, cache_dir, download, refresh)))

    season_set = parse_spec(seasons)
    if season_set is not None:
        df = df[df["season"].isin(season_set)]
    if home_and_away_only:
        df = df[~df["is_final"]]
    round_set = parse_spec(rounds)
    if round_set is not None:
        df = df[df["round_number"].isin(round_set)]
    team_set = parse_spec(teams, str)
    if team_set is not None:
        df = df[df["team"].isin(team_set)]
    player_set = parse_spec(players, str)
    if player_set is not None:
        df = df[df["player"].isin(player_set)]

    if require_complete_votes:
        valid = validate_votes(df)
        df = df[df["match_id"].isin(valid.index[valid["is_valid"]])]

    if df.empty:
        raise ValueError(
            "No rows matched those filters. Check the season/round/team names."
        )

    if columns is not None:
        identifiers = ["season", "round", "round_number", "match_id", "player", "team"]
        wanted = list(dict.fromkeys([*identifiers, *columns]))
        df = df[[c for c in wanted if c in df.columns]]

    df = df.sort_values(["season", "round_number", "match_id", "player"])
    return df.reset_index(drop=True)


def load_csv(path: Path) -> pd.DataFrame:
    """Load player-match rows from a CSV already in this project's tidy schema.

    Useful if you have your own data: a different scrape, a private export, or
    the named teams for a round that has not been played yet.
    """
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns and c != "votes"]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    if "votes" not in df.columns:
        df["votes"] = np.nan
    return df


def validate_votes(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per match describing its vote structure.

    Every completed home-and-away match should total 6 votes, awarded as
    exactly one 3, one 2 and one 1. Anything else means the match is unplayed
    or the data is wrong -- either way the model should not train on it.
    """
    grouped = df.groupby("match_id")["votes"]
    summary = pd.DataFrame(
        {
            "total_votes": grouped.sum(min_count=1),
            "n_players": grouped.size(),
            "n_voted": grouped.apply(lambda s: (s > 0).sum()),
            "max_votes": grouped.max(),
        }
    )
    summary["is_valid"] = (
        summary["total_votes"].eq(6)
        & summary["n_voted"].eq(3)
        & summary["max_votes"].eq(3)
    )
    return summary
