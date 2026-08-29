"""Tests for the vote models.

The important ones here check the *maths*, not just that the code runs:
that the per-match probabilities are a genuine probability distribution, that
the closed-form expected votes agree with brute-force simulation, and that the
model recovers a signal we planted ourselves.
"""

import numpy as np
import pytest

from brownlow.model import (
    MatchIndex,
    PlackettLuceModel,
    WeightedLogisticModel,
    segment_softmax,
)
from brownlow.synthetic import TRUE_WEIGHTS


class TestMatchIndex:
    def test_finds_group_boundaries(self):
        index = MatchIndex(["a", "a", "b", "b", "b", "c"])
        assert index.n_matches == 3
        assert index.starts.tolist() == [0, 2, 5]
        assert index.sizes.tolist() == [2, 3, 1]

    def test_rejects_ungrouped_rows(self):
        with pytest.raises(ValueError, match="grouped by match"):
            MatchIndex(["a", "b", "a"])


class TestSegmentSoftmax:
    def test_probabilities_sum_to_one_within_each_match(self):
        index = MatchIndex(["a", "a", "a", "b", "b"])
        values = np.array([1.0, 2.0, 3.0, 0.5, -0.5])
        probabilities, _ = segment_softmax(values, index)
        assert probabilities[:3].sum() == pytest.approx(1.0)
        assert probabilities[3:].sum() == pytest.approx(1.0)

    def test_masked_rows_get_no_probability(self):
        index = MatchIndex(["a", "a", "a"])
        values = np.array([1.0, 2.0, 3.0])
        mask = np.array([True, False, True])
        probabilities, _ = segment_softmax(values, index, mask)
        assert probabilities[1] == 0.0
        assert probabilities.sum() == pytest.approx(1.0)

    def test_is_invariant_to_a_constant_shift(self):
        """Anything constant across a match must cancel -- weather, era, umpires."""
        index = MatchIndex(["a", "a", "a"])
        base, _ = segment_softmax(np.array([1.0, 2.0, 3.0]), index)
        shifted, _ = segment_softmax(np.array([101.0, 102.0, 103.0]), index)
        np.testing.assert_allclose(base, shifted)


class TestPlackettLuceModel:
    def test_fits_and_predicts(self, synthetic_seasons):
        model = PlackettLuceModel(alpha=1.0).fit(synthetic_seasons)
        assert model.optimisation_["success"]
        assert model.coefficients_.shape[0] == len(model.feature_names_)

    def test_expected_votes_total_six_per_match(self, synthetic_seasons):
        """Reality awards exactly 6 votes a match, and so must the model."""
        model = PlackettLuceModel().fit(synthetic_seasons)
        predictions = model.predict(synthetic_seasons)
        totals = predictions.groupby("match_id")["expected_votes"].sum()
        np.testing.assert_allclose(totals.to_numpy(), 6.0, atol=1e-6)

    def test_allocation_is_only_ever_3_2_1_or_0(self, synthetic_seasons):
        """The model's call for a game looks like an umpire's card."""
        model = PlackettLuceModel().fit(synthetic_seasons)
        predictions = model.predict(synthetic_seasons)
        assert set(predictions["predicted_votes"].unique()) <= {0.0, 1.0, 2.0, 3.0}

    def test_every_match_allocates_exactly_one_of_each(self, synthetic_seasons):
        model = PlackettLuceModel().fit(synthetic_seasons)
        predictions = model.predict(synthetic_seasons)
        for _, group in predictions.groupby("match_id"):
            awarded = sorted(group["predicted_votes"][group["predicted_votes"] > 0],
                             reverse=True)
            assert awarded == [3.0, 2.0, 1.0]

    def test_allocated_votes_also_total_six_per_match(self, synthetic_seasons):
        model = PlackettLuceModel().fit(synthetic_seasons)
        totals = model.predict(synthetic_seasons).groupby("match_id")[
            "predicted_votes"].sum()
        assert (totals == 6.0).all()

    def test_the_3_goes_to_the_highest_rated_player(self, synthetic_seasons):
        """The allocation must follow the model's own ordering."""
        model = PlackettLuceModel().fit(synthetic_seasons)
        predictions = model.predict(synthetic_seasons)
        for _, group in predictions.groupby("match_id"):
            ordered = group.sort_values("expected_votes", ascending=False)
            assert ordered["predicted_votes"].tolist()[:3] == [3.0, 2.0, 1.0]

    def test_allocation_and_expectation_are_different_answers(self, synthetic_seasons):
        """If they were the same column there would be no point having both."""
        model = PlackettLuceModel().fit(synthetic_seasons)
        predictions = model.predict(synthetic_seasons)
        assert not np.allclose(predictions["predicted_votes"],
                               predictions["expected_votes"])
        # The expectation spreads across the whole field; the allocation does not.
        assert (predictions["expected_votes"] > 0).sum() > (
            predictions["predicted_votes"] > 0).sum()

    def test_vote_probabilities_are_valid(self, synthetic_seasons):
        model = PlackettLuceModel().fit(synthetic_seasons)
        predictions = model.predict(synthetic_seasons)
        for column in ("p_3_votes", "p_2_votes", "p_1_vote"):
            assert (predictions[column] >= -1e-9).all()
            assert (predictions[column] <= 1 + 1e-9).all()
            # Exactly one player per match takes each vote value.
            totals = predictions.groupby("match_id")[column].sum()
            np.testing.assert_allclose(totals.to_numpy(), 1.0, atol=1e-6)

    def test_closed_form_marginals_match_simulation(self, small_season):
        """The 2- and 1-vote formulas are fiddly, so check them by brute force."""
        model = PlackettLuceModel().fit(small_season)
        predictions = model.predict(small_season).sort_values("match_id", kind="stable")

        one_match = predictions[predictions["match_id"] == predictions["match_id"].iloc[0]]
        probabilities = one_match["p_3_votes"].to_numpy()

        rng = np.random.default_rng(0)
        draws = 200_000
        noisy = np.log(probabilities) + rng.gumbel(size=(draws, len(probabilities)))
        order = np.argsort(-noisy, axis=1)[:, :3]

        for position, column in enumerate(("p_3_votes", "p_2_votes", "p_1_vote")):
            empirical = np.bincount(order[:, position], minlength=len(probabilities)) / draws
            np.testing.assert_allclose(
                empirical, one_match[column].to_numpy(), atol=0.01
            )

    def test_recovers_the_planted_signal(self, synthetic_seasons):
        """Votes were generated from known weights; the model should find them."""
        model = PlackettLuceModel(alpha=0.5).fit(synthetic_seasons)
        coefficients = dict(zip(model.feature_names_, model.coefficients_))
        # Goals and winning were weighted heavily; nothing was weighted negatively.
        assert coefficients["goals"] > 0
        assert coefficients["win"] > 0
        assert coefficients["disposals"] > 0
        # Goals carried more weight per unit than marks in TRUE_WEIGHTS.
        assert TRUE_WEIGHTS["goals"] > TRUE_WEIGHTS["marks"]
        assert coefficients["goals"] > coefficients["marks"]

    def test_beats_random_guessing(self, synthetic_seasons):
        train = synthetic_seasons[synthetic_seasons["season"] < 2022]
        test = synthetic_seasons[synthetic_seasons["season"] == 2022]
        model = PlackettLuceModel().fit(train)
        predictions = model.predict(test)
        top = predictions.sort_values(["match_id", "expected_votes"], ascending=[True, False])
        hits = top.groupby("match_id").head(1)["votes"].eq(3).mean()
        assert hits > 0.2  # random guessing among ~44 players is about 0.02

    def test_rejects_training_data_without_votes(self, synthetic_seasons):
        blank = synthetic_seasons.copy()
        blank["votes"] = np.nan
        with pytest.raises(ValueError, match="votes"):
            PlackettLuceModel().fit(blank)

    def test_rejects_incomplete_matches(self, small_season):
        broken = small_season.copy()
        broken.loc[broken["votes"] == 3, "votes"] = 0.0
        with pytest.raises(ValueError, match="exactly one"):
            PlackettLuceModel().fit(broken)

    def test_predicting_before_fitting_is_an_error(self, small_season):
        with pytest.raises(ValueError, match="not fitted"):
            PlackettLuceModel().predict(small_season)

    def test_save_and_load_round_trip(self, small_season, tmp_path):
        model = PlackettLuceModel().fit(small_season)
        before = model.predict(small_season)["expected_votes"].to_numpy()

        path = model.save(tmp_path / "model.json")
        reloaded = PlackettLuceModel.load(path)
        after = reloaded.predict(small_season)["expected_votes"].to_numpy()

        np.testing.assert_allclose(before, after)

    def test_coefficient_table_is_sorted_by_effect_size(self, small_season):
        table = PlackettLuceModel().fit(small_season).coefficient_table()
        assert table["abs_coefficient"].is_monotonic_decreasing


class TestWeightedLogisticModel:
    def test_assigns_exactly_three_two_one_per_match(self, small_season):
        model = WeightedLogisticModel().fit(small_season)
        predictions = model.predict(small_season)
        for _, group in predictions.groupby("match_id"):
            awarded = sorted(group["predicted_votes"][group["predicted_votes"] > 0],
                             reverse=True)
            assert awarded == [3.0, 2.0, 1.0]

    def test_totals_six_votes_per_match(self, small_season):
        model = WeightedLogisticModel().fit(small_season)
        totals = model.predict(small_season).groupby("match_id")["predicted_votes"].sum()
        assert (totals == 6.0).all()

    def test_save_and_load_round_trip(self, small_season, tmp_path):
        model = WeightedLogisticModel().fit(small_season)
        before = model.predict(small_season)["score"].to_numpy()
        reloaded = WeightedLogisticModel.load(model.save(tmp_path / "m.json"))
        np.testing.assert_allclose(before, reloaded.predict(small_season)["score"].to_numpy())
