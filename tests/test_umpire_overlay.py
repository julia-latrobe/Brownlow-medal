"""Tests for the range that shows the umpires' standing view instead of the votes.

Everywhere else the published range asks how the votes might fall if the read on
the players is right. This one asks what happens if the read on the *umpires* is
wrong -- whether a player who has been rewarded beyond his statistics for years
keeps being rewarded or stops. That makes it one-sided, which is the point, and
means the usual "the range sits evenly around the projection" rule must not be
applied to it.
"""

import numpy as np
import pytest

from brownlow.experiment import ExperimentConfig, run_experiment
from brownlow.features import FeatureConfig
from brownlow.player_bias import PlayerAdjustedModel


@pytest.fixture(scope="module")
def fitted(synthetic_seasons):
    model = PlayerAdjustedModel(alpha=1.0, strength=0.5).fit(synthetic_seasons)
    return model, synthetic_seasons


class TestPredictingAtOtherStrengths:
    def test_zero_strength_matches_the_plain_model(self, fitted):
        from brownlow.model import PlackettLuceModel

        model, seasons = fitted
        at_zero = model.predict_at_strength(seasons, 0.0)
        plain = PlackettLuceModel(alpha=1.0).fit(seasons).predict(seasons)
        np.testing.assert_allclose(
            at_zero.sort_values(["match_id", "player"])["expected_votes"].to_numpy(),
            plain.sort_values(["match_id", "player"])["expected_votes"].to_numpy(),
            atol=1e-9)

    def test_its_own_strength_reproduces_its_own_prediction(self, fitted):
        model, seasons = fitted
        np.testing.assert_allclose(
            model.predict_at_strength(seasons, model.strength)["expected_votes"].to_numpy(),
            model.predict(seasons)["expected_votes"].to_numpy(), atol=1e-9)

    def test_it_leaves_the_model_as_it_found_it(self, fitted):
        """Rescaling is temporary -- the fitted model must survive it intact."""
        model, seasons = fitted
        before = dict(model.offsets_)
        model.predict_at_strength(seasons, 1.0)
        assert model.offsets_ == before

    def test_it_restores_the_model_even_when_prediction_fails(self, fitted):
        import pandas as pd

        model, _ = fitted
        before = dict(model.offsets_)
        with pytest.raises((ValueError, KeyError)):
            model.predict_at_strength(pd.DataFrame({"nonsense": [1]}), 1.0)
        assert model.offsets_ == before

    def test_a_model_with_no_adjustment_cannot_be_rescaled(self, synthetic_seasons):
        """Scaling nothing up is not possible, and must say so rather than
        quietly returning the plain model."""
        model = PlayerAdjustedModel(alpha=1.0, strength=0.0).fit(synthetic_seasons)
        with pytest.raises(ValueError, match="strength"):
            model.predict_at_strength(synthetic_seasons, 1.0)


class TestTheBand:
    def test_it_brackets_the_projection(self, fitted):
        model, seasons = fitted
        band = model.bias_band(seasons)
        assert (band["bias_low"] <= band["bias_high"] + 1e-9).all()

    def test_the_movement_is_the_width(self, fitted):
        model, seasons = fitted
        band = model.bias_band(seasons)
        np.testing.assert_allclose(
            band["bias_votes"].abs().to_numpy(),
            (band["bias_high"] - band["bias_low"]).to_numpy(), atol=1e-9)

    def test_both_ends_still_award_six_votes_a_match(self, fitted):
        """Either end is a whole projection, not a fudge applied to one."""
        model, seasons = fitted
        matches = seasons["match_id"].nunique()
        band = model.bias_band(seasons)
        for column in ("expected_votes_on_statistics",
                       "expected_votes_full_umpire_bias"):
            assert band[column].sum() == pytest.approx(6.0 * matches, rel=1e-6)

    def test_players_the_umpires_read_plainly_barely_move(self, fitted):
        """The band must be narrow where there is no standing view to apply."""
        model, seasons = fitted
        band = model.bias_band(seasons)
        widths = (band["bias_high"] - band["bias_low"]).to_numpy()
        assert widths.min() < widths.max(), "every player moved by the same amount"


class TestTheScenario:
    def test_the_published_range_becomes_the_bias_range(self, synthetic_seasons,
                                                        tmp_path):
        seasons = [int(v) for v in sorted(synthetic_seasons["season"].unique())]
        config = ExperimentConfig(
            name="overlay", model="player_adjusted", train_seasons=seasons[0],
            test_seasons=seasons[1], predict_seasons=seasons[-1],
            n_simulations=200, interval="umpire_bias",
            model_options={"strength": 0.5})
        results = run_experiment(config, synthetic_seasons, output_dir=tmp_path)
        board = results["leaderboard"]
        np.testing.assert_allclose(board["p10_votes"].to_numpy(),
                                   board["bias_low"].to_numpy(), atol=1e-9)
        np.testing.assert_allclose(board["p90_votes"].to_numpy(),
                                   board["bias_high"].to_numpy(), atol=1e-9)

    def test_the_projections_are_the_usual_ones(self, synthetic_seasons, tmp_path):
        """Only the range changes. The scenario must project exactly what
        player-adjusted projects, or the two cannot be read side by side."""
        seasons = [int(v) for v in sorted(synthetic_seasons["season"].unique())]
        base = dict(model="player_adjusted", train_seasons=seasons[0],
                    test_seasons=seasons[1], predict_seasons=seasons[-1],
                    n_simulations=200, model_options={"strength": 0.5})
        usual = run_experiment(ExperimentConfig(name="usual", **base),
                               synthetic_seasons, output_dir=tmp_path)
        overlay = run_experiment(
            ExperimentConfig(name="overlay2", interval="umpire_bias", **base),
            synthetic_seasons, output_dir=tmp_path)
        a = usual["leaderboard"].sort_values(["player", "team"])
        b = overlay["leaderboard"].sort_values(["player", "team"])
        np.testing.assert_allclose(a["expected_votes"].to_numpy(),
                                   b["expected_votes"].to_numpy(), atol=1e-12)
        assert list(usual["leaderboard"]["player"]) == \
            list(overlay["leaderboard"]["player"])

    def test_a_model_that_cannot_measure_the_bias_is_refused(self, synthetic_seasons,
                                                             tmp_path):
        seasons = [int(v) for v in sorted(synthetic_seasons["season"].unique())]
        config = ExperimentConfig(
            name="wrong", model="plackett_luce", train_seasons=seasons[0],
            test_seasons=seasons[1], predict_seasons=seasons[-1],
            n_simulations=200, interval="umpire_bias")
        with pytest.raises(ValueError, match="player_adjusted"):
            run_experiment(config, synthetic_seasons, output_dir=tmp_path)

    def test_the_shipped_config_is_valid(self):
        from pathlib import Path

        path = Path(__file__).resolve().parents[1] / "experiments" / "umpire-overlay.json"
        config = ExperimentConfig.from_json(path)
        assert config.interval == "umpire_bias"
        assert config.model == "player_adjusted"
        assert isinstance(config.build_model(), PlayerAdjustedModel)

    def test_every_other_scenario_keeps_the_usual_range(self):
        from pathlib import Path

        root = Path(__file__).resolve().parents[1] / "experiments"
        for path in sorted(root.glob("*.json")):
            config = ExperimentConfig.from_json(path)
            expected = "umpire_bias" if path.stem == "umpire-overlay" else "simulation"
            assert config.interval == expected, f"{path.stem} has interval {config.interval}"


def test_it_works_with_the_real_feature_set(synthetic_seasons):
    config = FeatureConfig(include_interactions=True, include_match_best=True)
    model = PlayerAdjustedModel(alpha=1.0, strength=0.5,
                                feature_config=config).fit(synthetic_seasons)
    band = model.bias_band(synthetic_seasons)
    assert len(band) > 0
    assert band["bias_low"].notna().all() and band["bias_high"].notna().all()
