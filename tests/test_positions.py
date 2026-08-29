"""Tests for derived playing position, AFL round labels and the team-leader metric."""

import numpy as np
import pandas as pd
import pytest

from brownlow.data import add_afl_round_labels
from brownlow.evaluate import declared_winner_round, team_leader_metrics
from brownlow.features import FeatureBuilder, FeatureConfig
from brownlow.model import PlackettLuceModel
from brownlow.positions import POSITIONS, add_position_features, assign_positions


def _player(name, team, season, games=10, **stats):
    base = dict(hit_outs=0.0, clearances=1.0, contested_possessions=5.0, disposals=15.0,
                inside_50s=2.0, rebounds_50=1.0, goals=0.3, behinds=0.2,
                marks_inside_50=0.2, one_percenters=1.0, tackles=2.0, marks=4.0)
    base.update(stats)
    return pd.DataFrame([{
        "player": name, "team": team, "season": season,
        "match_id": f"{season}-{i}", "round_number": float(i + 1), **base
    } for i in range(games)])


@pytest.fixture
def archetypes():
    """One clear example of each role, built from the statistics that define it."""
    return pd.concat([
        _player("A Ruck", "Geelong", 2023, hit_outs=30.0, clearances=3.0, disposals=12.0),
        _player("A Forward", "Geelong", 2023, goals=2.5, marks_inside_50=2.0,
                behinds=1.5, rebounds_50=0.1, inside_50s=1.0, disposals=11.0),
        _player("A Defender", "Carlton", 2023, rebounds_50=5.0, one_percenters=6.0,
                inside_50s=0.3, goals=0.0, clearances=0.2, disposals=16.0),
        _player("A Midfielder", "Carlton", 2023, clearances=6.0,
                contested_possessions=13.0, disposals=28.0, inside_50s=5.0),
    ], ignore_index=True)


class TestPositionAssignment:
    def test_each_archetype_gets_its_position(self, archetypes):
        positions = assign_positions(archetypes, leave_one_out=False)
        got = archetypes.assign(position=positions).groupby("player")["position"].first()
        assert got["A Ruck"] == "RUC"
        assert got["A Forward"] == "FWD"
        assert got["A Defender"] == "DEF"
        assert got["A Midfielder"] == "MID"

    def test_only_the_four_positions_are_ever_produced(self, synthetic_seasons):
        positions = assign_positions(synthetic_seasons)
        assert set(positions.unique()) <= set(POSITIONS)

    def test_every_row_gets_a_position(self, synthetic_seasons):
        positions = assign_positions(synthetic_seasons)
        assert len(positions) == len(synthetic_seasons)
        assert positions.notna().all()

    def test_hit_outs_always_win(self):
        """Nothing else in football produces hit-outs, so they decide it."""
        ruck = _player("Tall", "Sydney", 2023, hit_outs=25.0, clearances=8.0,
                       contested_possessions=15.0, disposals=25.0)
        assert (assign_positions(ruck, leave_one_out=False) == "RUC").all()

    def test_a_single_big_game_does_not_change_a_players_position(self, archetypes):
        """Leave-one-out means the match being predicted cannot reclassify you."""
        frame = archetypes.copy()
        midfielder = frame["player"] == "A Midfielder"
        first = frame.index[midfielder][0]
        frame.loc[first, "goals"] = 6.0
        frame.loc[first, "marks_inside_50"] = 5.0
        positions = assign_positions(frame, leave_one_out=True)
        assert positions.loc[first] == "MID", "one forward-looking game flipped the role"

    def test_position_does_not_depend_on_who_else_is_in_the_frame(self, archetypes):
        """Scores are ratios, so classification is frame-independent."""
        everyone = assign_positions(archetypes, leave_one_out=False)
        alone = archetypes[archetypes["player"] == "A Defender"]
        assert (assign_positions(alone, leave_one_out=False) == "DEF").all()
        assert (everyone[archetypes["player"] == "A Defender"] == "DEF").all()

    def test_row_order_is_preserved(self, synthetic_seasons):
        positions = assign_positions(synthetic_seasons)
        assert positions.index.equals(synthetic_seasons.index)


class TestPositionFeatures:
    def test_dummies_are_mutually_exclusive(self, synthetic_seasons):
        out = add_position_features(synthetic_seasons)
        dummies = out[[f"is_{p.lower()}" for p in POSITIONS]]
        assert (dummies.sum(axis=1) == 1).all()

    def test_interactions_are_the_product(self, synthetic_seasons):
        out = add_position_features(synthetic_seasons, stats=["disposals"])
        np.testing.assert_allclose(
            out["disposals_x_mid"].to_numpy(),
            out["disposals"].to_numpy() * out["is_mid"].to_numpy())

    def test_the_model_builds_and_predicts_with_positions(self, synthetic_seasons):
        config = FeatureConfig(include_position=True)
        model = PlackettLuceModel(feature_config=config).fit(synthetic_seasons)
        predictions = model.predict(synthetic_seasons)
        totals = predictions.groupby("match_id")["expected_votes"].sum()
        np.testing.assert_allclose(totals.to_numpy(), 6.0, atol=1e-9)

    def test_one_dummy_is_dropped_to_avoid_collinearity(self, synthetic_seasons):
        builder = FeatureBuilder(FeatureConfig(include_position=True))
        builder.transform(synthetic_seasons)
        dummies = [n for n in builder.feature_names_
                   if n in {f"is_{p.lower()}" for p in POSITIONS}]
        assert len(dummies) == len(POSITIONS) - 1
        assert "is_mid" not in dummies, "the reference position should be the dropped one"


class TestAflRoundLabels:
    def _season(self, matches_in_round_one, rounds=6):
        rows = []
        for round_number in range(1, rounds + 1):
            count = matches_in_round_one if round_number == 1 else 9
            for match in range(count):
                rows.append({"season": 2024, "round": str(round_number),
                             "round_number": float(round_number), "is_final": False,
                             "match_id": f"2024-R{round_number}-{match}"})
        return pd.DataFrame(rows)

    def test_a_short_first_round_is_named_opening_round(self):
        out = add_afl_round_labels(self._season(matches_in_round_one=4))
        first = out[out["round_number"] == 1.0]
        assert (first["afl_round"] == "Opening Round").all()

    def test_later_rounds_shift_down_by_one(self):
        out = add_afl_round_labels(self._season(matches_in_round_one=4))
        assert (out.loc[out["round_number"] == 2.0, "afl_round"] == "1").all()
        assert (out.loc[out["round_number"] == 6.0, "afl_round"] == "5").all()

    def test_a_normal_season_is_left_alone(self):
        out = add_afl_round_labels(self._season(matches_in_round_one=9))
        assert (out.loc[out["round_number"] == 1.0, "afl_round"] == "1").all()
        assert (out.loc[out["round_number"] == 6.0, "afl_round"] == "6").all()

    def test_the_source_numbering_is_kept_for_ordering(self):
        out = add_afl_round_labels(self._season(matches_in_round_one=4))
        assert out["round_number"].max() == 6.0, "source numbering must not change"


class TestTeamLeaderMetric:
    def test_a_perfect_prediction_names_every_club_leader(self, small_season):
        model = PlackettLuceModel().fit(small_season)
        predictions = model.predict(small_season)
        predictions["expected_votes"] = predictions["votes"]  # pretend we were perfect
        metrics = team_leader_metrics(predictions)
        assert metrics["team_leader_accuracy"] == 1.0

    def test_it_scores_between_zero_and_one(self, fitted):
        metrics = team_leader_metrics(fitted.predictions)
        assert 0.0 <= metrics["team_leader_accuracy"] <= 1.0
        assert metrics["team_leaders_scored"] > 0

    def test_clubs_nobody_polled_for_are_not_scored(self, small_season):
        model = PlackettLuceModel().fit(small_season)
        predictions = model.predict(small_season)
        metrics = team_leader_metrics(predictions)
        clubs = predictions.groupby(["season", "team"])["votes"].max()
        assert metrics["team_leaders_scored"] == int((clubs > 0).sum())


class TestDeclaredWinnerRound:
    def test_a_runaway_leader_is_declared_early(self):
        votes = pd.DataFrame([[3, 3, 3, 3], [0, 0, 0, 0]],
                             index=["Leader", "Other"], columns=[1, 2, 3, 4])
        # After round 2 the leader has 6 and the rival can reach at most 6 -- not
        # yet decided. After round 3 the leader has 9 against a maximum of 3.
        assert declared_winner_round(votes) == 3

    def test_a_tight_count_goes_to_the_final_round(self):
        votes = pd.DataFrame([[3, 0, 3, 0], [0, 3, 0, 3]],
                             index=["A", "B"], columns=[1, 2, 3, 4])
        assert declared_winner_round(votes) is None, "a tie is never declared"

    def test_a_last_round_winner_is_declared_in_the_last_round(self):
        votes = pd.DataFrame([[3, 0, 3, 1], [0, 3, 0, 3]],
                             index=["A", "B"], columns=[1, 2, 3, 4])
        assert declared_winner_round(votes) == 4

    def test_an_empty_count_declares_nothing(self):
        assert declared_winner_round(pd.DataFrame()) is None
