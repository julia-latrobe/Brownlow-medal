"""Tests for feature engineering."""

import pandas as pd
import pytest

from brownlow.features import FeatureBuilder, FeatureConfig, add_derived_stats, build_features


@pytest.fixture
def two_matches():
    return pd.DataFrame(
        {
            "match_id": ["m1", "m1", "m1", "m2", "m2", "m2"],
            "team": ["A", "A", "B", "C", "C", "D"],
            "player": list("uvwxyz"),
            "disposals": [30.0, 20.0, 10.0, 15.0, 10.0, 5.0],
            "kicks": [20.0, 12.0, 6.0, 9.0, 6.0, 3.0],
            "handballs": [10.0, 8.0, 4.0, 6.0, 4.0, 2.0],
            "marks": [5.0, 3.0, 1.0, 4.0, 2.0, 1.0],
            "tackles": [4.0, 2.0, 6.0, 3.0, 1.0, 2.0],
            "goals": [3.0, 0.0, 1.0, 2.0, 0.0, 0.0],
            "behinds": [1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            "hit_outs": [0.0] * 6,
            "frees_for": [1.0] * 6,
            "frees_against": [0.0] * 6,
            "goal_assists": [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            "clangers": [2.0, 1.0, 0.0, 1.0, 1.0, 0.0],
            "contested_possessions": [12.0, 8.0, 4.0, 6.0, 4.0, 2.0],
            "win": [1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
            "margin": [20.0, 20.0, -20.0, 10.0, 10.0, -10.0],
            "is_home": [1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
        }
    )


class TestDerivedStats:
    def test_fantasy_points_follow_the_published_formula(self, two_matches):
        out = add_derived_stats(two_matches, FeatureConfig())
        expected = 20 * 3 + 10 * 2 + 5 * 3 + 4 * 4 + 1 * 1 - 0 * 3 + 3 * 6 + 1 * 1 + 0
        assert out.loc[0, "fantasy_points"] == expected

    def test_goal_accuracy_is_zero_when_there_were_no_shots(self, two_matches):
        out = add_derived_stats(two_matches, FeatureConfig())
        assert out.loc[5, "shots"] == 0
        assert out.loc[5, "goal_accuracy"] == 0.0


class TestWithinMatchFeatures:
    def test_zscore_is_computed_inside_the_match_only(self, two_matches):
        out = build_features(two_matches)
        # Match 1 disposals are 30/20/10 -> mean 20, sd 10.
        assert out.loc[0, "disposals_z"] == pytest.approx(1.0)
        assert out.loc[1, "disposals_z"] == pytest.approx(0.0)
        assert out.loc[2, "disposals_z"] == pytest.approx(-1.0)

    def test_zscores_do_not_leak_between_matches(self, two_matches):
        """The best player of a low-scoring game still tops that game."""
        out = build_features(two_matches)
        assert out.loc[3, "disposals_z"] == pytest.approx(1.0)

    def test_rank_is_a_percentile_within_the_match(self, two_matches):
        out = build_features(two_matches)
        assert out.loc[0, "disposals_rank"] == pytest.approx(1.0)
        assert out.loc[2, "disposals_rank"] == pytest.approx(1 / 3)

    def test_team_share_sums_to_one_per_team(self, two_matches):
        out = build_features(two_matches)
        team_a = out[(out["match_id"] == "m1") & (out["team"] == "A")]
        assert team_a["disposals_team_share"].sum() == pytest.approx(1.0)

    def test_win_interaction_is_zero_for_losers(self, two_matches):
        out = build_features(two_matches)
        assert out.loc[2, "disposals_x_win"] == 0.0
        assert out.loc[0, "disposals_x_win"] == 30.0


class TestFeatureBuilder:
    def test_feature_names_are_recorded_and_unique(self, two_matches):
        builder = FeatureBuilder()
        builder.transform(two_matches)
        names = builder.feature_names_
        assert names and len(names) == len(set(names))

    def test_config_can_switch_features_off(self, two_matches):
        config = FeatureConfig(within_match_stats=(), include_shares=False,
                               win_interactions=(), include_ranks=False)
        builder = FeatureBuilder(config)
        builder.transform(two_matches)
        assert not any(n.endswith(("_z", "_rank", "_team_share", "_x_win"))
                       for n in builder.feature_names_)

    def test_no_missing_values_survive(self, two_matches):
        builder = FeatureBuilder()
        out = builder.transform(two_matches)
        assert not out[builder.feature_names_].isna().any().any()

    def test_requires_a_match_id(self):
        with pytest.raises(ValueError, match="match_id"):
            FeatureBuilder().transform(pd.DataFrame({"disposals": [1.0]}))

    def test_is_deterministic(self, two_matches):
        a = build_features(two_matches)
        b = build_features(two_matches)
        pd.testing.assert_frame_equal(a, b)
