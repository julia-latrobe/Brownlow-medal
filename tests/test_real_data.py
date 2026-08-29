"""Contract tests against the actual AFL Tables data.

Synthetic fixtures are well behaved by construction. Real data is not: two
players share a name, some rows arrive with no player id, Opening Round is
folded into round 1, and finals award no votes at all. Every defect this suite
was written after was found by running against real data, not against fixtures,
so these tests exist to close that gap.

They skip automatically when the download is absent, which is how CI runs them.
Locally, `brownlow fetch` turns them on.
"""

import numpy as np
import pytest

from brownlow.data import (
    FINALS_ROUNDS,
    default_cache_dir,
    load_player_matches,
    validate_votes,
)

# Self-contained rather than imported from conftest, so the skip works whatever
# import mode pytest is run under.
pytestmark = [
    pytest.mark.realdata,
    pytest.mark.skipif(
        not (default_cache_dir() / "afldata.rda").exists(),
        reason="needs the AFL Tables data; run `brownlow fetch` first",
    ),
]

# The seasons with the detailed statistics the model relies on.
FIRST_SEASON, LAST_COUNTED_SEASON = 2015, 2025


@pytest.fixture(scope="module")
def real_seasons():
    return load_player_matches(seasons=f"{FIRST_SEASON}-{LAST_COUNTED_SEASON}",
                               require_complete_votes=True)


@pytest.fixture(scope="module")
def everything_loaded():
    """Including finals and the season still being played."""
    return load_player_matches(seasons="2015-2026", home_and_away_only=False)


class TestVoteStructure:
    def test_every_match_awards_exactly_six_votes(self, real_seasons):
        totals = real_seasons.groupby("match_id")["votes"].sum()
        assert (totals == 6).all(), (
            f"{(totals != 6).sum()} matches did not award six votes")

    def test_every_match_awards_one_three_one_two_one_one(self, real_seasons):
        shape = real_seasons.groupby("match_id")["votes"].apply(
            lambda s: tuple(sorted(s[s > 0], reverse=True)))
        assert set(shape.unique()) == {(3.0, 2.0, 1.0)}

    def test_validate_votes_agrees(self, real_seasons):
        summary = validate_votes(real_seasons)
        assert summary["is_valid"].all()

    def test_season_totals_are_six_times_the_matches(self, real_seasons):
        for season, group in real_seasons.groupby("season"):
            matches = group["match_id"].nunique()
            assert group["votes"].sum() == 6 * matches, f"{season} does not balance"

    def test_finals_award_no_votes(self, everything_loaded):
        finals = everything_loaded[everything_loaded["round"].isin(FINALS_ROUNDS)]
        assert len(finals) > 0, "no finals found; the fixture is not testing anything"
        assert finals["votes"].fillna(0).sum() == 0

    def test_finals_are_excluded_by_default(self, real_seasons):
        assert not real_seasons["is_final"].any()


class TestMatchShape:
    def test_a_match_has_a_plausible_number_of_players(self, real_seasons):
        counts = real_seasons.groupby("match_id").size()
        assert counts.min() >= 40, f"a match had only {counts.min()} players"
        assert counts.max() <= 50, f"a match had {counts.max()} players"

    def test_every_match_has_exactly_two_teams(self, real_seasons):
        teams = real_seasons.groupby("match_id")["team"].nunique()
        assert (teams == 2).all()

    def test_a_player_appears_once_per_match(self, real_seasons):
        duplicated = real_seasons.duplicated(subset=["match_id", "player", "team"])
        assert not duplicated.any()

    def test_each_season_has_the_expected_number_of_matches(self, real_seasons):
        counts = real_seasons.groupby("season")["match_id"].nunique()
        # 2020 was shortened by the pandemic; every other season is a full draw.
        for season, matches in counts.items():
            if season == 2020:
                assert 140 <= matches <= 165, f"{season}: {matches} matches"
            else:
                assert 190 <= matches <= 215, f"{season}: {matches} matches"

    def test_both_teams_are_named_in_the_match_id(self, real_seasons):
        sample = real_seasons.drop_duplicates("match_id").head(200)
        for _, row in sample.iterrows():
            assert row["team"] in row["match_id"] or row["opponent"] in row["match_id"]


class TestPlayers:
    def test_no_player_is_left_without_a_name(self, everything_loaded):
        """The upstream mirror blanks some names; the loader rebuilds them."""
        assert everything_loaded["player"].isna().sum() == 0
        assert (everything_loaded["player"].astype(str).str.strip() == "").sum() == 0

    def test_players_sharing_a_name_are_not_merged(self, real_seasons):
        """The AFL has had two Bailey Williamses at once, at different clubs."""
        by_name = real_seasons.groupby("player")["team"].nunique()
        shared = by_name[by_name > 1]
        assert len(shared) > 0, "expected at least one shared name in the real data"
        for name in shared.index:
            rows = real_seasons[real_seasons["player"] == name]
            per_team = rows.groupby("team")["match_id"].nunique()
            assert (per_team > 0).all()

    def test_nobody_plays_more_games_than_there_are_rounds(self, real_seasons):
        for season, group in real_seasons.groupby("season"):
            rounds = group["round"].nunique()
            games = group.groupby(["player", "team"]).size()
            assert games.max() <= rounds, (
                f"{season}: a player logged {games.max()} games in {rounds} rounds")


class TestRounds:
    def test_opening_round_is_folded_into_round_one(self, real_seasons):
        """AFL Tables numbers Opening Round as round 1, so 2024+ has a short one."""
        counts = real_seasons.groupby(["season", "round_number"])["match_id"].nunique()
        for season in (2024, 2025):
            if season not in counts.index.get_level_values(0):
                continue
            assert counts.loc[(season, 1.0)] < 7, (
                "round 1 of a post-2023 season should be the short Opening Round")

    def test_rounds_are_numbered_from_one_without_gaps(self, real_seasons):
        for _season, group in real_seasons.groupby("season"):
            rounds = sorted(group["round_number"].dropna().unique())
            assert rounds[0] == 1.0
            assert rounds == list(np.arange(1.0, len(rounds) + 1))

    def test_every_match_has_a_date_and_start_time(self, real_seasons):
        assert real_seasons["date"].isna().sum() == 0
        assert real_seasons["local_start_time"].isna().sum() == 0

    def test_start_times_are_valid_clock_readings(self, real_seasons):
        times = real_seasons["local_start_time"].dropna().astype(int)
        assert (times >= 0).all() and (times <= 2359).all()
        assert (times % 100 < 60).all(), "a start time had more than 59 minutes"

    def test_rounds_run_in_date_order(self, real_seasons):
        """Round numbers should broadly follow the calendar."""
        for season, group in real_seasons.groupby("season"):
            by_round = group.groupby("round_number")["date"].min().sort_index()
            assert by_round.is_monotonic_increasing, f"{season} rounds are out of order"


class TestStatistics:
    def test_counting_stats_are_never_negative(self, real_seasons):
        from brownlow.data import FEATURE_STAT_COLUMNS

        for column in FEATURE_STAT_COLUMNS:
            assert real_seasons[column].min() >= 0, f"{column} went negative"

    def test_disposals_are_kicks_plus_handballs(self, real_seasons):
        difference = (real_seasons["disposals"]
                      - real_seasons["kicks"] - real_seasons["handballs"]).abs()
        assert difference.max() == 0

    def test_scores_are_consistent_with_the_margin(self, real_seasons):
        rebuilt = real_seasons["team_score"] - real_seasons["opp_score"]
        assert (rebuilt == real_seasons["margin"]).all()

    def test_exactly_one_team_wins_each_match(self, real_seasons):
        for match_id, group in real_seasons.groupby("match_id"):
            wins = group.groupby("team")["win"].first()
            assert wins.sum() == pytest.approx(1.0), f"{match_id} has no single winner"

    def test_time_on_ground_is_a_percentage(self, real_seasons):
        values = real_seasons["time_on_ground"]
        assert values.min() >= 0
        assert values.max() <= 100


class TestVoteGetters:
    def test_vote_getters_are_better_than_average(self, real_seasons):
        """A sanity check on the data itself, not the model."""
        polled = real_seasons[real_seasons["votes"] > 0]
        assert polled["disposals"].mean() > real_seasons["disposals"].mean() * 1.4

    def test_most_votes_go_to_the_winning_team(self, real_seasons):
        polled = real_seasons[real_seasons["votes"] > 0]
        assert 0.70 < polled["win"].mean() < 0.90
        threes = real_seasons[real_seasons["votes"] == 3]
        assert threes["win"].mean() > 0.85


class TestUnplayedSeason:
    def test_the_current_season_has_no_votes_yet(self, everything_loaded):
        current = everything_loaded[everything_loaded["season"] == 2026]
        if current.empty:
            pytest.skip("2026 not present in this snapshot")
        assert current["votes"].fillna(0).sum() == 0

    def test_the_current_season_still_has_full_statistics(self, everything_loaded):
        current = everything_loaded[everything_loaded["season"] == 2026]
        if current.empty:
            pytest.skip("2026 not present in this snapshot")
        assert current["disposals"].sum() > 0
        assert current["player"].nunique() > 500

    def test_require_complete_votes_drops_the_uncounted_season(self):
        counted = load_player_matches(seasons="2025-2026", require_complete_votes=True)
        assert set(counted["season"].unique()) == {2025}
