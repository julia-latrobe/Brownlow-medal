"""Tests for loading and tidying the raw data."""

import numpy as np
import pandas as pd
import pytest

from brownlow.data import parse_spec, tidy, validate_votes


class TestParseSpec:
    """The filters accept whatever a caller or a command line hands them."""

    def test_none_means_no_filter(self):
        assert parse_spec(None) is None

    def test_single_value(self):
        assert parse_spec(2023) == {2023}

    def test_range_string(self):
        assert parse_spec("2015-2018") == {2015, 2016, 2017, 2018}

    def test_comma_list(self):
        assert parse_spec("2021,2023") == {2021, 2023}

    def test_mixed(self):
        assert parse_spec("2015-2017,2023") == {2015, 2016, 2017, 2023}

    def test_iterable(self):
        assert parse_spec(range(2015, 2018)) == {2015, 2016, 2017}

    def test_strings(self):
        assert parse_spec("Geelong,Carlton", str) == {"Geelong", "Carlton"}


@pytest.fixture
def raw_frame():
    """A minimal frame in the raw AFL Tables shape."""
    return pd.DataFrame(
        {
            "Season": [2023, 2023, 2023, 2023],
            "Round": ["1", "1", "GF", "GF"],
            "Player": ["A Player", None, "C Player", "D Player"],
            "First.name": ["A", "B", "C", "D"],
            "Surname": ["Player", "Player", "Player", "Player"],
            "Playing.for": ["Geelong", "Carlton", "Geelong", "Carlton"],
            "Home.team": ["Geelong"] * 4,
            "Away.team": ["Carlton"] * 4,
            "Home.score": [100, 100, 90, 90],
            "Away.score": [80, 80, 95, 95],
            "Home.Away": ["Home", "Away", "Home", "Away"],
            "Brownlow.Votes": [3, 0, None, None],
            "Disposals": [30, 20, 25, 22],
            "Goals": [2, 1, 0, 3],
        }
    )


class TestTidy:
    def test_renames_to_snake_case(self, raw_frame):
        out = tidy(raw_frame)
        assert {"season", "player", "team", "votes", "disposals"} <= set(out.columns)

    def test_rebuilds_missing_player_names(self, raw_frame):
        """The upstream mirror sometimes blanks Player but keeps first/surname."""
        out = tidy(raw_frame)
        assert out.loc[1, "player"] == "B Player"
        assert out["player"].isna().sum() == 0

    def test_flags_finals(self, raw_frame):
        out = tidy(raw_frame)
        assert out["is_final"].tolist() == [False, False, True, True]

    def test_margin_is_from_the_players_own_team(self, raw_frame):
        out = tidy(raw_frame)
        assert out.loc[0, "margin"] == 20   # Geelong won by 20
        assert out.loc[1, "margin"] == -20  # Carlton lost by 20

    def test_win_indicator(self, raw_frame):
        out = tidy(raw_frame)
        assert out.loc[0, "win"] == 1.0
        assert out.loc[1, "win"] == 0.0

    def test_draw_counts_as_half_a_win(self):
        frame = pd.DataFrame(
            {
                "Season": [2023, 2023], "Round": ["1", "1"],
                "Player": ["A", "B"], "Playing.for": ["Geelong", "Carlton"],
                "Home.team": ["Geelong"] * 2, "Away.team": ["Carlton"] * 2,
                "Home.score": [90, 90], "Away.score": [90, 90],
                "Home.Away": ["Home", "Away"], "Brownlow.Votes": [3, 2],
                "Disposals": [30, 20],
            }
        )
        assert tidy(frame)["win"].tolist() == [0.5, 0.5]

    def test_match_id_is_stable(self, raw_frame):
        out = tidy(raw_frame)
        assert out.loc[0, "match_id"] == out.loc[1, "match_id"]
        assert out.loc[0, "match_id"] != out.loc[2, "match_id"]


class TestValidateVotes:
    def test_accepts_a_proper_three_two_one(self, small_season):
        summary = validate_votes(small_season)
        assert summary["is_valid"].all()
        assert (summary["total_votes"] == 6).all()

    def test_rejects_an_incomplete_match(self, small_season):
        broken = small_season.copy()
        broken.loc[broken["votes"] == 3, "votes"] = 0
        assert not validate_votes(broken)["is_valid"].all()

    def test_rejects_an_unplayed_season(self, small_season):
        future = small_season.copy()
        future["votes"] = np.nan
        assert not validate_votes(future)["is_valid"].any()
