"""Invariants: things that must be true of any prediction, from any model.

These are not tests of particular numbers. They are the arithmetic the
competition itself guarantees -- six votes a match, one 3, one 2, one 1, nobody
counted twice -- plus the structural promises the code makes about row order and
missing values.

Every check runs against every feature configuration. That is the point: the
defects that prompted this suite were all real logic errors sitting in
configurations no test happened to exercise, so covering configurations matters
as much as covering functions.
"""

import numpy as np
import pandas as pd
import pytest

from brownlow.features import FeatureBuilder
from brownlow.model import PlackettLuceModel


class TestVoteArithmetic:
    """The competition's own counting rules."""

    def test_exactly_six_votes_are_allocated_per_match(self, fitted_any_model):
        totals = fitted_any_model.predictions.groupby("match_id")["predicted_votes"].sum()
        assert (totals == 6.0).all(), "every match awards exactly six votes"

    def test_exactly_six_expected_votes_per_match(self, fitted):
        totals = fitted.predictions.groupby("match_id")["expected_votes"].sum()
        np.testing.assert_allclose(totals.to_numpy(), 6.0, atol=1e-9)

    def test_one_three_one_two_one_one_per_match(self, fitted_any_model):
        for match_id, group in fitted_any_model.predictions.groupby("match_id"):
            awarded = sorted(group.loc[group["predicted_votes"] > 0, "predicted_votes"],
                             reverse=True)
            assert awarded == [3.0, 2.0, 1.0], f"{match_id} did not award a clean 3-2-1"

    def test_allocation_only_uses_legal_values(self, fitted_any_model):
        values = set(fitted_any_model.predictions["predicted_votes"].unique())
        assert values <= {0.0, 1.0, 2.0, 3.0}

    def test_each_vote_probability_sums_to_one_per_match(self, fitted):
        for column in ("p_3_votes", "p_2_votes", "p_1_vote"):
            totals = fitted.predictions.groupby("match_id")[column].sum()
            np.testing.assert_allclose(totals.to_numpy(), 1.0, atol=1e-6)

    def test_probabilities_stay_between_zero_and_one(self, fitted):
        for column in ("p_3_votes", "p_2_votes", "p_1_vote", "p_any_votes"):
            values = fitted.predictions[column]
            assert values.min() >= -1e-9, f"{column} went negative"
            assert values.max() <= 1 + 1e-9, f"{column} exceeded 1"

    def test_expected_votes_are_the_weighted_probabilities(self, fitted):
        predictions = fitted.predictions
        rebuilt = (3 * predictions["p_3_votes"] + 2 * predictions["p_2_votes"]
                   + predictions["p_1_vote"])
        np.testing.assert_allclose(rebuilt.to_numpy(),
                                   predictions["expected_votes"].to_numpy(), atol=1e-12)

    def test_no_player_can_exceed_three_votes_in_a_match(self, fitted):
        assert fitted.predictions["expected_votes"].max() <= 3.0 + 1e-9


class TestCounts:
    """Nothing gained, nothing lost, nothing duplicated."""

    def test_every_input_row_gets_a_prediction(self, fitted):
        assert len(fitted.predictions) == len(fitted.frame)

    def test_match_count_is_unchanged(self, fitted):
        assert (fitted.predictions["match_id"].nunique()
                == fitted.frame["match_id"].nunique())

    def test_player_count_is_unchanged(self, fitted):
        before = fitted.frame.groupby(["player", "team"]).ngroups
        after = fitted.predictions.groupby(["player", "team"]).ngroups
        assert before == after

    def test_each_player_appears_once_per_match(self, fitted):
        duplicated = fitted.predictions.duplicated(subset=["match_id", "player", "team"])
        assert not duplicated.any(), "a player was predicted twice in one match"

    def test_games_per_player_match_the_input(self, fitted):
        before = fitted.frame.groupby(["player", "team"]).size()
        after = fitted.predictions.groupby(["player", "team"]).size()
        pd.testing.assert_series_equal(before.sort_index(), after.sort_index())

    def test_season_totals_sum_to_six_per_match(self, fitted):
        board = fitted.model.season_totals(fitted.predictions)
        n_matches = fitted.predictions["match_id"].nunique()
        assert board["predicted_votes"].sum() == pytest.approx(6 * n_matches)
        assert board["expected_votes"].sum() == pytest.approx(6 * n_matches)

    def test_season_totals_have_one_row_per_player_season_team(self, fitted):
        board = fitted.model.season_totals(fitted.predictions)
        keys = [c for c in ("season", "player", "team") if c in board.columns]
        assert not board.duplicated(subset=keys).any()

    def test_season_totals_games_match_appearances(self, fitted):
        board = fitted.model.season_totals(fitted.predictions)
        counted = (fitted.predictions.groupby(["season", "player", "team"]).size()
                   .rename("games").reset_index())
        merged = board.merge(counted, on=["season", "player", "team"],
                             suffixes=("_board", "_counted"))
        assert len(merged) == len(board)
        assert (merged["games_board"] == merged["games_counted"]).all()

    def test_a_player_who_played_every_round_is_counted_that_many_times(self, fitted):
        counts = fitted.predictions.groupby(["season", "player", "team"]).size()
        seasons = fitted.predictions.groupby("season")["round"].nunique()
        assert counts.max() <= seasons.max()


class TestNoMissingValues:
    """A NaN in the wrong place is how several of these bugs presented."""

    def test_no_missing_predictions(self, fitted):
        columns = ["predicted_votes", "expected_votes", "p_3_votes", "p_2_votes",
                   "p_1_vote", "score"]
        missing = fitted.predictions[columns].isna().sum()
        assert missing.sum() == 0, f"missing values: {missing[missing > 0].to_dict()}"

    def test_predictions_are_finite(self, fitted):
        columns = ["predicted_votes", "expected_votes", "score"]
        values = fitted.predictions[columns].to_numpy(dtype=float)
        assert np.isfinite(values).all()

    def test_no_missing_features(self, fitted):
        builder = FeatureBuilder(fitted.config)
        built = builder.transform(fitted.frame)
        block = built[builder.feature_names_]
        assert not block.isna().any().any()
        assert np.isfinite(block.to_numpy(dtype=float)).all()

    def test_scores_are_never_all_identical(self, fitted):
        """A constant score would mean the model learned nothing at all."""
        assert fitted.predictions["score"].std() > 1e-6


class TestRowOrder:
    """The bug class that cost the most: right numbers, wrong rows."""

    def test_predict_preserves_the_input_rows(self, fitted):
        before = fitted.frame[["match_id", "player", "team"]].sort_values(
            ["match_id", "player", "team"]).reset_index(drop=True)
        after = fitted.predictions[["match_id", "player", "team"]].sort_values(
            ["match_id", "player", "team"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(before, after)

    def test_predict_scores_returns_the_callers_order(self, fitted):
        scores = fitted.model.predict_scores(fitted.frame)
        assert len(scores) == len(fitted.frame)
        assert np.isfinite(scores).all(), "scores came back misaligned or missing"

        shuffled = fitted.frame.sample(frac=1.0, random_state=13)
        shuffled_scores = fitted.model.predict_scores(shuffled)
        reference = pd.Series(scores, index=fitted.frame.index)
        np.testing.assert_allclose(
            shuffled_scores, reference.reindex(shuffled.index).to_numpy(), atol=1e-9)

    def test_predict_scores_agrees_with_predict(self, fitted):
        scores = pd.Series(fitted.model.predict_scores(fitted.frame),
                           index=fitted.frame.index)
        np.testing.assert_allclose(
            fitted.predictions["score"].to_numpy(),
            scores.reindex(fitted.predictions.index).to_numpy(), atol=1e-9)

    def test_shuffling_the_input_does_not_change_a_players_result(self, fitted):
        shuffled = fitted.frame.sample(frac=1.0, random_state=29)
        again = fitted.model.predict(shuffled)
        keys = ["match_id", "player", "team"]
        merged = fitted.predictions[keys + ["expected_votes", "predicted_votes"]].merge(
            again[keys + ["expected_votes", "predicted_votes"]], on=keys,
            suffixes=("_a", "_b"))
        assert len(merged) == len(fitted.predictions)
        np.testing.assert_allclose(merged["expected_votes_a"], merged["expected_votes_b"],
                                   atol=1e-9)
        assert (merged["predicted_votes_a"] == merged["predicted_votes_b"]).all()

    def test_feature_building_preserves_row_order(self, fitted):
        built = FeatureBuilder(fitted.config).transform(fitted.frame)
        assert built["player"].tolist() == fitted.frame["player"].tolist()
        assert built.index.equals(fitted.frame.index)

    def test_feature_building_is_idempotent(self, fitted):
        """Transforming an already-transformed frame must not corrupt it."""
        builder = FeatureBuilder(fitted.config)
        once = builder.transform(fitted.frame)
        names = list(builder.feature_names_)
        twice = FeatureBuilder(fitted.config).transform(once)
        for name in names:
            assert name in twice.columns, f"{name} vanished on the second transform"
            np.testing.assert_allclose(once[name].to_numpy(dtype=float),
                                       twice[name].to_numpy(dtype=float), atol=1e-9)


class TestSharedNames:
    """Two players can have the same name. They are not the same player."""

    def test_shared_names_stay_separate_in_predictions(self, messy_season):
        model = PlackettLuceModel().fit(messy_season)
        predictions = model.predict(messy_season)
        shared = predictions[predictions["player"] == "Shared Name"]
        assert shared["team"].nunique() == 2

    def test_shared_names_get_separate_leaderboard_rows(self, messy_season):
        model = PlackettLuceModel().fit(messy_season)
        board = model.season_totals(model.predict(messy_season))
        rows = board[board["player"] == "Shared Name"]
        assert len(rows) == rows["team"].nunique() * rows["season"].nunique()

    def test_a_missing_player_id_does_not_merge_players(self, messy_season):
        """Some rows have no id upstream; falling back must not pool players."""
        from brownlow.features import player_identity

        keys = pd.Series(player_identity(messy_season))
        pairs = messy_season[["player", "team"]].assign(key=keys)
        for _, group in pairs.groupby("key"):
            assert group["player"].nunique() == 1, "one key covered two players"


class TestEdgeCases:
    def test_a_single_match_still_works(self, small_season):
        one = small_season[small_season["match_id"] == small_season["match_id"].iloc[0]]
        model = PlackettLuceModel().fit(small_season)
        predictions = model.predict(one)
        assert predictions["expected_votes"].sum() == pytest.approx(6.0)
        assert sorted(predictions["predicted_votes"].unique()) == [0.0, 1.0, 2.0, 3.0]

    def test_predicting_a_season_with_no_votes_yet(self, synthetic_seasons):
        """The real use: the season being projected has not been counted."""
        future = synthetic_seasons[synthetic_seasons["season"] == 2022].copy()
        future["votes"] = np.nan
        model = PlackettLuceModel().fit(
            synthetic_seasons[synthetic_seasons["season"] < 2022])
        predictions = model.predict(future)
        assert predictions["expected_votes"].notna().all()
        totals = predictions.groupby("match_id")["expected_votes"].sum()
        np.testing.assert_allclose(totals.to_numpy(), 6.0, atol=1e-9)

    def test_a_non_default_index_is_handled(self, small_season):
        """Callers do not always hand over a tidy RangeIndex."""
        odd = small_season.copy()
        odd.index = np.arange(1000, 1000 + len(odd))
        model = PlackettLuceModel().fit(small_season)
        predictions = model.predict(odd)
        assert len(predictions) == len(odd)
        assert predictions["expected_votes"].notna().all()
        scores = model.predict_scores(odd)
        assert np.isfinite(scores).all()
