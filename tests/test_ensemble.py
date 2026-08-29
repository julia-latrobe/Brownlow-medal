"""Tests for the pooled rank + boosted model.

They skip when LightGBM is absent, which is how the base install runs. The whole
point of making it an optional extra is that everything else keeps working.
"""

import numpy as np
import pandas as pd
import pytest

from brownlow.ensemble import lightgbm_available
from brownlow.features import FeatureConfig

pytestmark = pytest.mark.skipif(
    not lightgbm_available(),
    reason="needs the optional boost extra: pip install 'brownlow[boost]'",
)


@pytest.fixture(scope="module")
def fitted_ensemble(request):
    from brownlow.ensemble import EnsembleModel

    seasons = request.getfixturevalue("synthetic_seasons")
    model = EnsembleModel(alpha=1.0, n_estimators=60, num_leaves=7).fit(seasons)
    return model, seasons, model.predict(seasons)


class TestEnsembleObeysTheSameRules:
    """Everything true of a single model must stay true of the pooled one."""

    def test_expected_votes_total_six_per_match(self, fitted_ensemble):
        _, _, predictions = fitted_ensemble
        totals = predictions.groupby("match_id")["expected_votes"].sum()
        np.testing.assert_allclose(totals.to_numpy(), 6.0, atol=1e-9)

    def test_allocation_is_a_clean_three_two_one(self, fitted_ensemble):
        _, _, predictions = fitted_ensemble
        for _, group in predictions.groupby("match_id"):
            awarded = sorted(group.loc[group["predicted_votes"] > 0, "predicted_votes"],
                             reverse=True)
            assert awarded == [3.0, 2.0, 1.0]

    def test_probabilities_sum_to_one_per_match(self, fitted_ensemble):
        _, _, predictions = fitted_ensemble
        for column in ("p_3_votes", "p_2_votes", "p_1_vote"):
            totals = predictions.groupby("match_id")[column].sum()
            np.testing.assert_allclose(totals.to_numpy(), 1.0, atol=1e-6)

    def test_no_row_is_gained_or_lost(self, fitted_ensemble):
        _, seasons, predictions = fitted_ensemble
        assert len(predictions) == len(seasons)

    def test_nothing_is_missing(self, fitted_ensemble):
        _, _, predictions = fitted_ensemble
        columns = ["score", "expected_votes", "predicted_votes", "p_3_votes"]
        assert predictions[columns].isna().sum().sum() == 0
        assert np.isfinite(predictions[columns].to_numpy(dtype=float)).all()

    def test_predict_scores_comes_back_in_the_callers_order(self, fitted_ensemble):
        model, seasons, _ = fitted_ensemble
        scores = model.predict_scores(seasons)
        assert np.isfinite(scores).all()
        shuffled = seasons.sample(frac=1.0, random_state=17)
        reference = pd.Series(scores, index=seasons.index)
        np.testing.assert_allclose(
            model.predict_scores(shuffled),
            reference.reindex(shuffled.index).to_numpy(), atol=1e-9)


class TestPooling:
    def test_it_differs_from_either_member(self, fitted_ensemble):
        """If pooling matched one member there would be no point to it."""
        from brownlow.model import PlackettLuceModel

        model, seasons, predictions = fitted_ensemble
        linear = PlackettLuceModel(alpha=1.0).fit(seasons).predict(seasons)
        merged = predictions[["match_id", "player", "team", "expected_votes"]].merge(
            linear[["match_id", "player", "team", "expected_votes"]],
            on=["match_id", "player", "team"], suffixes=("_pool", "_linear"))
        assert not np.allclose(merged["expected_votes_pool"],
                               merged["expected_votes_linear"], atol=1e-6)

    def test_weighting_entirely_to_the_linear_member_reproduces_it(self,
                                                                   synthetic_seasons):
        from brownlow.ensemble import EnsembleModel
        from brownlow.model import PlackettLuceModel

        pooled = EnsembleModel(alpha=1.0, n_estimators=40, weights=(1.0, 0.0))
        pooled.fit(synthetic_seasons)
        linear = PlackettLuceModel(alpha=1.0).fit(synthetic_seasons)

        a = pooled.predict(synthetic_seasons).sort_values(["match_id", "player"])
        b = linear.predict(synthetic_seasons).sort_values(["match_id", "player"])
        np.testing.assert_allclose(a["expected_votes"].to_numpy(),
                                   b["expected_votes"].to_numpy(), atol=1e-6)

    def test_weights_are_normalised(self):
        from brownlow.ensemble import EnsembleModel

        assert EnsembleModel(weights=(2.0, 2.0)).weights == (0.5, 0.5)
        assert EnsembleModel(weights=(3.0, 1.0)).weights == (0.75, 0.25)


class TestIntegration:
    def test_it_runs_as_an_experiment(self, synthetic_seasons, tmp_path):
        from brownlow.experiment import ExperimentConfig, run_experiment

        config = ExperimentConfig(name="ens", model="ensemble", train_seasons="2020-2021",
                                  test_seasons="2022", n_simulations=100)
        results = run_experiment(config, synthetic_seasons, output_dir=tmp_path)
        assert 0.0 <= results["holdout_metrics"]["top3_recall"] <= 1.0
        for filename in ("model.json", "predictions.csv", "leaderboard.csv"):
            assert (tmp_path / "ens" / filename).exists()

    def test_the_coefficient_table_shows_both_members(self, fitted_ensemble):
        model, _, _ = fitted_ensemble
        table = model.coefficient_table()
        assert {"feature", "coefficient", "booster_splits"} <= set(table.columns)
        assert table["booster_splits"].sum() > 0, "the booster used no features"

    def test_it_survives_a_save_and_load(self, fitted_ensemble, tmp_path):
        from brownlow.ensemble import EnsembleModel

        model, seasons, predictions = fitted_ensemble
        path = model.save(tmp_path / "ensemble.json")
        reloaded = EnsembleModel.load(path)
        again = reloaded.predict(seasons)
        np.testing.assert_allclose(predictions["expected_votes"].to_numpy(),
                                   again["expected_votes"].to_numpy(), atol=1e-9)

    def test_it_works_with_position_features(self, synthetic_seasons):
        from brownlow.ensemble import EnsembleModel

        config = FeatureConfig(include_position=True, include_interactions=True)
        model = EnsembleModel(alpha=1.0, n_estimators=40,
                              feature_config=config).fit(synthetic_seasons)
        predictions = model.predict(synthetic_seasons)
        totals = predictions.groupby("match_id")["expected_votes"].sum()
        np.testing.assert_allclose(totals.to_numpy(), 6.0, atol=1e-9)


def test_a_clear_error_when_the_extra_is_missing(monkeypatch):
    """Without LightGBM the message must say how to get it."""
    import builtins

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "lightgbm":
            raise ImportError("no lightgbm")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    from brownlow.ensemble import _require_lightgbm

    with pytest.raises(ImportError, match=r"brownlow\[boost\]"):
        _require_lightgbm()
