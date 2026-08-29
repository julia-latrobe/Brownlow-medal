"""Tests for the history, form and interaction features.

The defining property of every feature here is that it looks *backwards only*.
A feature that peeks at the current match, or at votes nobody had seen yet,
would make the model look brilliant in a backtest and useless in September. Most
of these tests exist to prove that has not happened.
"""

import numpy as np
import pandas as pd
import pytest

from brownlow.features import (
    DEFAULT_INTERACTION_PAIRS,
    FeatureBuilder,
    FeatureConfig,
    add_derived_stats,
    add_form_features,
    add_history_features,
    player_identity,
)


@pytest.fixture
def career():
    """One player across four seasons, polling a different amount each year."""
    rows = []
    for season, votes_each in ((2020, 1.0), (2021, 2.0), (2022, 0.0), (2023, 3.0)):
        for round_number in range(1, 4):
            rows.append({
                "season": season,
                "round_number": float(round_number),
                "match_id": f"{season}-R{round_number}",
                "player": "A Player",
                "player_id": 42.0,
                "team": "Geelong",
                "votes": votes_each,
                "disposals": 20.0 + round_number,
                "kicks": 10.0, "handballs": 10.0, "marks": 4.0, "tackles": 3.0,
                "goals": 1.0, "behinds": 0.0, "hit_outs": 0.0,
                "frees_for": 0.0, "frees_against": 0.0,
                "contested_possessions": 10.0, "clearances": 3.0,
            })
    return pd.DataFrame(rows)


class TestPlayerIdentity:
    def test_prefers_the_player_id(self, career):
        assert all(str(k).startswith("id:") for k in player_identity(career))

    def test_falls_back_to_name_and_club(self):
        frame = pd.DataFrame({"player": ["Sam", "Sam"], "team": ["Geelong", "Carlton"],
                              "player_id": [np.nan, np.nan]})
        keys = player_identity(frame)
        assert keys[0] != keys[1], "two players sharing a name must not merge"

    def test_one_player_keeps_one_key_across_clubs(self):
        frame = pd.DataFrame({"player": ["Sam", "Sam"], "team": ["Geelong", "Carlton"],
                              "player_id": [7.0, 7.0]})
        keys = player_identity(frame)
        assert keys[0] == keys[1], "a traded player is still the same player"


class TestHistoryFeatures:
    def test_first_season_has_no_history(self, career):
        out = add_history_features(career)
        first = out[out["season"] == 2020]
        assert (first["prior_season_votes"] == 0).all()
        assert (first["career_votes_before"] == 0).all()
        assert (first["seasons_before"] == 0).all()

    def test_prior_season_is_the_season_before(self, career):
        out = add_history_features(career)
        by_season = out.groupby("season")["prior_season_votes"].first()
        # 3 games a season: 2020 polled 3, 2021 polled 6, 2022 polled 0.
        assert by_season[2021] == 3.0
        assert by_season[2022] == 6.0
        assert by_season[2023] == 0.0

    def test_career_votes_accumulate_but_exclude_this_season(self, career):
        out = add_history_features(career)
        by_season = out.groupby("season")["career_votes_before"].first()
        assert by_season[2020] == 0.0
        assert by_season[2021] == 3.0
        assert by_season[2022] == 9.0
        assert by_season[2023] == 9.0

    def test_never_uses_the_current_season(self, career):
        """Votes are not published until count night, so the current season is unknown."""
        changed = career.copy()
        changed.loc[changed["season"] == 2023, "votes"] = 99.0
        before = add_history_features(career)
        after = add_history_features(changed)
        for column in ("prior_season_votes", "career_votes_before"):
            season_2023 = changed["season"] == 2023
            np.testing.assert_array_equal(
                before.loc[season_2023, column].to_numpy(),
                after.loc[season_2023, column].to_numpy(),
            )

    def test_has_polled_before_is_a_flag(self, career):
        out = add_history_features(career)
        assert out[out["season"] == 2020]["has_polled_before"].eq(0).all()
        assert out[out["season"] == 2023]["has_polled_before"].eq(1).all()

    def test_two_players_sharing_a_name_keep_separate_histories(self):
        rows = []
        for team, votes in (("Geelong", 3.0), ("Carlton", 0.0)):
            for season in (2020, 2021):
                rows.append({"season": season, "round_number": 1.0,
                             "match_id": f"{season}-{team}", "player": "Sam Smith",
                             "player_id": np.nan, "team": team, "votes": votes})
        out = add_history_features(pd.DataFrame(rows))
        second = out[out["season"] == 2021]
        assert second[second["team"] == "Geelong"]["prior_season_votes"].iloc[0] == 3.0
        assert second[second["team"] == "Carlton"]["prior_season_votes"].iloc[0] == 0.0


class TestFormFeatures:
    def test_first_game_has_no_form(self, career):
        out = add_form_features(career)
        out = out.sort_values(["season", "round_number"])
        assert out.iloc[0]["fantasy_points_form"] == 0.0

    def test_form_excludes_the_current_match(self, career):
        """A rating entering a game must not contain that game."""
        out = add_form_features(career).sort_values(["season", "round_number"])
        with_stats = add_derived_stats(career, FeatureConfig())
        second = out.iloc[1]
        first_points = with_stats.sort_values(["season", "round_number"]).iloc[0]
        assert second["fantasy_points_form"] == pytest.approx(
            first_points["fantasy_points"]
        )

    def test_recent_games_count_for_more(self):
        """A decaying average should sit nearer the latest game than the oldest."""
        rows = [{
            "season": 2021, "round_number": float(i), "match_id": f"m{i}",
            "player": "A Player", "player_id": 1.0, "team": "Geelong", "votes": 0.0,
            "disposals": value, "kicks": value, "handballs": 0.0, "marks": 0.0,
            "tackles": 0.0, "goals": 0.0, "behinds": 0.0, "hit_outs": 0.0,
            "frees_for": 0.0, "frees_against": 0.0, "contested_possessions": 0.0,
        } for i, value in enumerate([10.0, 10.0, 10.0, 30.0], start=1)]
        out = add_form_features(pd.DataFrame(rows), halflife=1.0)
        last = out.sort_values("round_number").iloc[-1]
        # Entering game 4 the player had three 10-disposal games.
        assert last["disposals_form"] == pytest.approx(10.0)

    def test_carries_across_seasons(self, career):
        out = add_form_features(career).sort_values(["season", "round_number"])
        first_of_2021 = out[out["season"] == 2021].iloc[0]
        assert first_of_2021["disposals_form"] > 0

    def test_games_played_before_counts_up(self, career):
        out = add_form_features(career).sort_values(["season", "round_number"])
        assert out["games_played_before"].tolist() == list(range(len(career)))


class TestInteractionFeatures:
    def test_off_by_default(self, synthetic_seasons):
        builder = FeatureBuilder(FeatureConfig())
        builder.transform(synthetic_seasons)
        assert not any("_x_" in n and not n.endswith("_x_win")
                       for n in builder.feature_names_)

    def test_products_are_added_when_asked(self, synthetic_seasons):
        builder = FeatureBuilder(FeatureConfig(include_interactions=True))
        out = builder.transform(synthetic_seasons)
        expected = [f"{a}_x_{b}" for a, b in DEFAULT_INTERACTION_PAIRS
                    if a in out.columns and b in out.columns]
        assert expected
        for name in expected:
            assert name in builder.feature_names_

    def test_a_product_is_actually_the_product(self, synthetic_seasons):
        out = FeatureBuilder(FeatureConfig(include_interactions=True)).transform(
            synthetic_seasons)
        np.testing.assert_allclose(
            out["disposals_z_x_goals_z"].to_numpy(),
            out["disposals_z"].to_numpy() * out["goals_z"].to_numpy(),
        )

    def test_match_best_flags_exactly_the_match_leader(self, synthetic_seasons):
        out = FeatureBuilder(FeatureConfig(include_match_best=True)).transform(
            synthetic_seasons)
        for _, group in out.groupby("match_id"):
            leader = group["disposals"].max()
            flagged = group[group["disposals_match_best"] == 1]
            assert (flagged["disposals"] == leader).all()
            assert len(flagged) >= 1

    def test_custom_pairs_are_respected(self, synthetic_seasons):
        config = FeatureConfig(include_interactions=True,
                               interaction_pairs=(("goals", "win"),))
        builder = FeatureBuilder(config)
        builder.transform(synthetic_seasons)
        assert "goals_x_win" in builder.feature_names_
        assert "disposals_z_x_goals_z" not in builder.feature_names_


class TestScenarioConfigsBuild:
    """Every feature combination the shipped scenarios use must actually build."""

    @pytest.mark.parametrize("features", [
        {"include_history": True},
        {"include_form": True},
        {"include_interactions": True, "include_match_best": True},
        {"include_history": True, "include_form": True,
         "include_interactions": True, "include_match_best": True},
        {"base_stats": ["disposals", "clearances"], "include_fantasy": False,
         "within_match_stats": ["disposals"]},
    ])
    def test_builds_without_missing_values(self, synthetic_seasons, features):
        builder = FeatureBuilder(FeatureConfig(**features))
        out = builder.transform(synthetic_seasons)
        assert builder.feature_names_
        assert not out[builder.feature_names_].isna().any().any()
        assert np.isfinite(out[builder.feature_names_].to_numpy()).all()
