"""Tests for the standing per-player adjustment.

The adjustment is the one part of the project that treats a named individual
differently from anyone else with the same statistics, so the things worth
pinning down are that it stays inside the rules every other model obeys, that
it cannot see a season it has not been shown, and that turning it off really
does turn it off.
"""

import numpy as np
import pandas as pd
import pytest

from brownlow.features import FeatureConfig
from brownlow.model import PlackettLuceModel
from brownlow.player_bias import PlayerAdjustedModel


@pytest.fixture(scope="module")
def fitted(synthetic_seasons):
    model = PlayerAdjustedModel(alpha=1.0, strength=0.5).fit(synthetic_seasons)
    return model, synthetic_seasons, model.predict(synthetic_seasons)


class TestItObeysTheSameRules:
    """Whatever the adjustment does, a match still awards exactly 3, 2 and 1."""

    def test_expected_votes_total_six_per_match(self, fitted):
        _, _, predictions = fitted
        totals = predictions.groupby("match_id")["expected_votes"].sum()
        np.testing.assert_allclose(totals.to_numpy(), 6.0, atol=1e-9)

    def test_allocation_is_a_clean_three_two_one(self, fitted):
        _, _, predictions = fitted
        for _, group in predictions.groupby("match_id"):
            awarded = sorted(group.loc[group["predicted_votes"] > 0, "predicted_votes"],
                             reverse=True)
            assert awarded == [3.0, 2.0, 1.0]

    def test_probabilities_sum_to_one_per_match(self, fitted):
        _, _, predictions = fitted
        for column in ("p_3_votes", "p_2_votes", "p_1_vote"):
            totals = predictions.groupby("match_id")[column].sum()
            np.testing.assert_allclose(totals.to_numpy(), 1.0, atol=1e-6)

    def test_no_row_is_gained_or_lost(self, fitted):
        _, seasons, predictions = fitted
        assert len(predictions) == len(seasons)

    def test_nothing_is_missing(self, fitted):
        _, _, predictions = fitted
        columns = ["score", "expected_votes", "predicted_votes", "player_adjustment"]
        assert predictions[columns].isna().sum().sum() == 0
        assert np.isfinite(predictions[columns].to_numpy(dtype=float)).all()

    def test_predict_scores_comes_back_in_the_callers_order(self, fitted):
        model, seasons, _ = fitted
        scores = pd.Series(model.predict_scores(seasons), index=seasons.index)
        shuffled = seasons.sample(frac=1.0, random_state=23)
        np.testing.assert_allclose(model.predict_scores(shuffled),
                                   scores.reindex(shuffled.index).to_numpy(), atol=1e-9)


class TestTheAdjustmentItself:
    def test_zero_strength_is_exactly_the_plain_model(self, synthetic_seasons):
        """The knob has to reach all the way to off, or it cannot be compared."""
        adjusted = PlayerAdjustedModel(alpha=1.0, strength=0.0).fit(synthetic_seasons)
        plain = PlackettLuceModel(alpha=1.0).fit(synthetic_seasons)
        a = adjusted.predict(synthetic_seasons).sort_values(["match_id", "player"])
        b = plain.predict(synthetic_seasons).sort_values(["match_id", "player"])
        np.testing.assert_allclose(a["expected_votes"].to_numpy(),
                                   b["expected_votes"].to_numpy(), atol=1e-9)
        assert (a["player_adjustment"] == 0.0).all()

    def test_a_stronger_setting_moves_predictions_further(self, synthetic_seasons):
        plain = PlackettLuceModel(alpha=1.0).fit(synthetic_seasons).predict(
            synthetic_seasons).sort_values(["match_id", "player"])["expected_votes"]
        distances = []
        for strength in (0.25, 1.0):
            model = PlayerAdjustedModel(alpha=1.0, strength=strength).fit(synthetic_seasons)
            moved = model.predict(synthetic_seasons).sort_values(
                ["match_id", "player"])["expected_votes"]
            distances.append(float(np.abs(moved.to_numpy() - plain.to_numpy()).sum()))
        assert distances[1] > distances[0]

    def test_adjustments_stay_within_the_ceiling(self, fitted):
        model, _, _ = fitted
        assert all(abs(v) <= model.max_offset for v in model.offsets_.values())

    def test_an_unknown_player_gets_no_adjustment(self, fitted):
        """A debutant has no history, so the model must not invent one for him."""
        model, seasons, _ = fitted
        stranger = seasons.head(20).copy()
        stranger["player"] = "Nobody In Particular"
        stranger["team"] = "Nowhere"
        if "player_id" in stranger.columns:
            stranger["player_id"] = np.nan
        assert (model._offsets_for(stranger) == 0.0).all()

    def test_players_seen_less_are_adjusted_less(self, synthetic_seasons):
        """Two seasons of evidence should not carry the weight of six."""
        cautious = PlayerAdjustedModel(alpha=1.0, strength=1.0, prior_seasons=10.0)
        eager = PlayerAdjustedModel(alpha=1.0, strength=1.0, prior_seasons=0.0)
        cautious.fit(synthetic_seasons)
        eager.fit(synthetic_seasons)
        total = sum(abs(v) for v in cautious.offsets_.values())
        assert total < sum(abs(v) for v in eager.offsets_.values())

    def test_the_table_names_who_moved_and_which_way(self, fitted):
        model, _, _ = fitted
        table = model.adjustment_table()
        assert {"player", "adjustment", "seasons_seen", "direction"} <= set(table.columns)
        if len(table) > 1:  # sorted by size of effect
            assert table["adjustment"].abs().is_monotonic_decreasing


class TestItCannotSeeTheFuture:
    def test_the_predicted_season_does_not_change_the_adjustments(self, synthetic_seasons):
        """Votes are not published until the count, so a season being predicted
        must not be able to feed its own answers back into the adjustment."""
        train = synthetic_seasons[synthetic_seasons["season"] <= 2021]
        model = PlayerAdjustedModel(alpha=1.0, strength=1.0).fit(train)
        before = dict(model.offsets_)

        later = synthetic_seasons[synthetic_seasons["season"] > 2021]
        if len(later):
            model.predict(later)
        assert model.offsets_ == before

    def test_shuffling_the_held_out_votes_changes_nothing(self, synthetic_seasons):
        train = synthetic_seasons[synthetic_seasons["season"] <= 2021]
        test = synthetic_seasons[synthetic_seasons["season"] > 2021].copy()
        if not len(test):
            pytest.skip("fixture has no later season to hold out")
        model = PlayerAdjustedModel(alpha=1.0, strength=1.0).fit(train)
        honest = model.predict(test)["expected_votes"].to_numpy()

        scrambled = test.copy()
        scrambled["votes"] = np.random.default_rng(0).permutation(
            scrambled["votes"].to_numpy())
        np.testing.assert_allclose(
            model.predict(scrambled)["expected_votes"].to_numpy(), honest, atol=1e-12)


class TestIntegration:
    def test_it_survives_a_save_and_load(self, fitted, tmp_path):
        model, seasons, predictions = fitted
        path = model.save(tmp_path / "adjusted.json")
        reloaded = PlayerAdjustedModel.load(path)
        assert reloaded.offsets_ == model.offsets_
        np.testing.assert_allclose(
            reloaded.predict(seasons)["expected_votes"].to_numpy(),
            predictions["expected_votes"].to_numpy(), atol=1e-9)

    def test_it_runs_as_an_experiment(self, synthetic_seasons, tmp_path):
        from brownlow.experiment import ExperimentConfig, run_experiment

        config = ExperimentConfig(name="pa", model="player_adjusted",
                                  train_seasons="2020-2021", test_seasons="2022",
                                  n_simulations=100,
                                  model_options={"strength": 0.5})
        results = run_experiment(config, synthetic_seasons, output_dir=tmp_path)
        assert 0.0 <= results["holdout_metrics"]["top3_recall"] <= 1.0
        for filename in ("model.json", "predictions.csv", "leaderboard.csv"):
            assert (tmp_path / "pa" / filename).exists()

    def test_the_shipped_scenario_config_is_valid(self):
        from pathlib import Path

        from brownlow.experiment import ExperimentConfig

        path = Path(__file__).resolve().parents[1] / "experiments" / "player-adjusted.json"
        config = ExperimentConfig.from_json(path)
        assert config.model == "player_adjusted"
        assert isinstance(config.build_model(), PlayerAdjustedModel)

    def test_unknown_model_options_are_rejected(self):
        from brownlow.experiment import ExperimentConfig

        config = ExperimentConfig(model="player_adjusted",
                                  model_options={"not_a_real_knob": 1})
        with pytest.raises(TypeError, match="not_a_real_knob"):
            config.build_model()

    def test_it_works_with_a_richer_feature_set(self, synthetic_seasons):
        config = FeatureConfig(include_interactions=True, include_match_best=True)
        model = PlayerAdjustedModel(alpha=1.0, feature_config=config).fit(synthetic_seasons)
        totals = model.predict(synthetic_seasons).groupby("match_id")["expected_votes"].sum()
        np.testing.assert_allclose(totals.to_numpy(), 6.0, atol=1e-9)
