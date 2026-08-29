"""Tests for scoring and cross-validation."""

import numpy as np
import pytest

from brownlow.evaluate import (
    backtest,
    evaluate_season,
    leave_one_season_out_cv,
    match_metrics,
    plackett_luce_log_likelihood,
    rolling_origin_cv,
    season_metrics,
    summarise_cv,
)
from brownlow.model import PlackettLuceModel


@pytest.fixture
def perfect_predictions(small_season):
    """Predictions that exactly reproduce the real votes."""
    out = small_season.copy()
    out["predicted_votes"] = out["votes"]
    return out


class TestMatchMetrics:
    def test_perfect_predictions_score_perfectly(self, perfect_predictions):
        metrics = match_metrics(perfect_predictions)
        assert metrics["top1_accuracy"] == 1.0
        assert metrics["top3_recall"] == 1.0
        assert metrics["exact_order_accuracy"] == 1.0

    def test_right_players_wrong_order_still_scores_top3(self, small_season):
        out = small_season.copy()
        # Swap the 3 and the 1: same three players, wrong order.
        swapped = out["votes"].replace({3.0: 1.0, 1.0: 3.0})
        out["predicted_votes"] = swapped
        metrics = match_metrics(out)
        assert metrics["top3_recall"] == 1.0
        assert metrics["top1_accuracy"] == 0.0

    def test_requires_known_votes(self, small_season):
        blank = small_season.copy()
        blank["votes"] = np.nan
        blank["predicted_votes"] = 1.0
        with pytest.raises(ValueError, match="known votes"):
            match_metrics(blank)


class TestSeasonMetrics:
    def test_perfect_predictions_correlate_perfectly(self, perfect_predictions):
        metrics = season_metrics(perfect_predictions)
        assert metrics["spearman"] == pytest.approx(1.0)
        assert metrics["winner_correct"]
        assert metrics["season_total_mae"] == 0.0


class TestLogLikelihood:
    def test_a_fitted_model_beats_random_guessing(self, small_season):
        model = PlackettLuceModel().fit(small_season)
        result = plackett_luce_log_likelihood(model.predict(small_season))
        assert result["log_likelihood_per_match"] > result["random_baseline_per_match"]

    def test_needs_the_score_column(self, perfect_predictions):
        with pytest.raises(ValueError, match="score"):
            plackett_luce_log_likelihood(perfect_predictions)


class TestBacktest:
    def test_holds_the_test_seasons_out(self, synthetic_seasons):
        result = backtest(synthetic_seasons, PlackettLuceModel(),
                          train_seasons=[2020, 2021], test_seasons=[2022])
        assert result["train_seasons"] == [2020, 2021]
        assert set(result["predictions"]["season"]) == {2022}
        assert 0.0 <= result["metrics"]["top3_recall"] <= 1.0

    def test_refuses_overlapping_splits(self, synthetic_seasons):
        """Training on a season you then score against is not a holdout."""
        with pytest.raises(ValueError, match="overlap"):
            backtest(synthetic_seasons, train_seasons=[2020, 2021], test_seasons=[2021])

    def test_refuses_an_empty_split(self, synthetic_seasons):
        with pytest.raises(ValueError, match="empty"):
            backtest(synthetic_seasons, train_seasons=[1999], test_seasons=[2022])


class TestCrossValidation:
    def test_rolling_origin_never_trains_on_the_future(self, synthetic_seasons):
        folds = rolling_origin_cv(synthetic_seasons, min_train_seasons=1)
        assert list(folds["fold_test_season"]) == [2021, 2022]
        # The window expands: fold 1 trains on one season, fold 2 on two.
        assert list(folds["n_train_seasons"]) == [1, 2]

    def test_sliding_window_keeps_a_fixed_width(self, synthetic_seasons):
        folds = rolling_origin_cv(synthetic_seasons, min_train_seasons=1, expanding=False)
        assert set(folds["n_train_seasons"]) == {1}

    def test_needs_enough_seasons(self, small_season):
        with pytest.raises(ValueError, match="cross-validate"):
            rolling_origin_cv(small_season, min_train_seasons=4)

    def test_leave_one_season_out_covers_every_season(self, synthetic_seasons):
        folds = leave_one_season_out_cv(synthetic_seasons)
        assert list(folds["fold_test_season"]) == [2020, 2021, 2022]

    def test_summarise_averages_the_folds(self, synthetic_seasons):
        folds = rolling_origin_cv(synthetic_seasons, min_train_seasons=1)
        summary = summarise_cv(folds)
        assert summary["top3_recall"] == pytest.approx(folds["top3_recall"].mean())
        assert "fold_test_season" not in summary.index


class TestEvaluateSeason:
    def test_gathers_every_metric(self, small_season):
        model = PlackettLuceModel().fit(small_season)
        metrics = evaluate_season(model.predict(small_season))
        for key in ("top1_accuracy", "top3_recall", "spearman", "log_likelihood_per_match"):
            assert key in metrics
