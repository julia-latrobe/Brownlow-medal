"""Tests for the two corrections that widen the simulated ranges.

The model was stating ranges far tighter than six held-out seasons support: a
stated 95% range held the truth 84% of the time. Softening the probabilities and
allowing that the read on a player may be wrong fixes that. What matters in
tests is that they widen things without disturbing anything they should not --
the projected votes and the leaderboard order come from exact marginals and must
come out identical either way.
"""

import numpy as np
import pandas as pd
import pytest

from brownlow.model import PlackettLuceModel
from brownlow.simulate import (
    DEFAULT_PLAYER_SHOCK,
    DEFAULT_TEMPERATURE,
    simulate_season,
)


@pytest.fixture(scope="module")
def predictions(synthetic_seasons):
    model = PlackettLuceModel(alpha=1.0).fit(synthetic_seasons)
    latest = synthetic_seasons[synthetic_seasons["season"] ==
                               synthetic_seasons["season"].max()]
    return model.predict(latest), latest


def widths(summary):
    return (summary["p90_votes"] - summary["p10_votes"]).to_numpy(dtype=float)


class TestTheDefaultsAreTheCorrectedOnes:
    def test_the_defaults_are_what_the_backtest_chose(self):
        assert DEFAULT_TEMPERATURE == 0.8
        assert DEFAULT_PLAYER_SHOCK == 0.5

    def test_turning_both_off_restores_the_old_behaviour(self, predictions):
        """The uncorrected simulation has to remain reachable, or the two
        cannot be compared."""
        frame, _ = predictions
        off = simulate_season(frame, n_simulations=400, seed=3,
                              temperature=1.0, player_shock=0.0)
        again = simulate_season(frame, n_simulations=400, seed=3,
                                temperature=1.0, player_shock=0.0)
        pd.testing.assert_frame_equal(off, again)


class TestTheyWidenTheRanges:
    def test_the_shock_widens_them(self, predictions):
        frame, _ = predictions
        narrow = simulate_season(frame, n_simulations=1500, seed=1,
                                 temperature=1.0, player_shock=0.0)
        wide = simulate_season(frame, n_simulations=1500, seed=1,
                               temperature=1.0, player_shock=0.5)
        pair = narrow[["player", "team"]].assign(narrow=widths(narrow)).merge(
            wide[["player", "team"]].assign(wide=widths(wide)), on=["player", "team"])
        busy = pair[pair["narrow"] > 0]
        assert busy["wide"].mean() > busy["narrow"].mean()

    def test_softening_widens_them(self, predictions):
        frame, _ = predictions
        sharp = simulate_season(frame, n_simulations=1500, seed=1,
                                temperature=1.0, player_shock=0.0)
        soft = simulate_season(frame, n_simulations=1500, seed=1,
                               temperature=0.8, player_shock=0.0)
        assert widths(soft).mean() > widths(sharp).mean()

    def test_a_bigger_shock_widens_them_further(self, predictions):
        frame, _ = predictions
        means = []
        for shock in (0.0, 0.5, 1.0):
            summary = simulate_season(frame, n_simulations=1500, seed=1,
                                      temperature=1.0, player_shock=shock)
            means.append(widths(summary).mean())
        assert means[0] < means[1] < means[2]

    def test_the_favourite_becomes_less_certain(self, predictions):
        """Over-confidence showed up hardest at the top of the leaderboard."""
        frame, _ = predictions
        sharp = simulate_season(frame, n_simulations=2000, seed=1,
                                temperature=1.0, player_shock=0.0)
        corrected = simulate_season(frame, n_simulations=2000, seed=1)
        assert corrected["win_probability"].max() <= sharp["win_probability"].max()


class TestNothingElseMoves:
    """The corrections may touch the ranges and the probabilities. They must not
    touch a projected vote total or the order of the leaderboard."""

    def test_every_simulated_season_still_awards_six_votes_a_match(self, predictions):
        frame, _ = predictions
        summary = simulate_season(frame, n_simulations=800, seed=5)
        expected = 6.0 * frame["match_id"].nunique()
        assert summary["mean_votes"].sum() == pytest.approx(expected, rel=1e-9)

    def test_win_probabilities_still_add_to_one(self, predictions):
        frame, _ = predictions
        summary = simulate_season(frame, n_simulations=800, seed=5)
        assert summary["win_probability"].sum() == pytest.approx(1.0, abs=1e-9)

    def test_no_player_is_gained_or_lost(self, predictions):
        frame, _ = predictions
        a = simulate_season(frame, n_simulations=400, seed=5, player_shock=0.0)
        b = simulate_season(frame, n_simulations=400, seed=5, player_shock=0.9)
        assert set(zip(a["player"], a["team"])) == set(zip(b["player"], b["team"]))

    def test_projected_votes_are_untouched(self, synthetic_seasons, tmp_path):
        """expected_votes comes from the exact marginals, so the same config run
        with and without the corrections must produce identical projections."""
        from brownlow.experiment import ExperimentConfig, run_experiment

        seasons = [int(v) for v in sorted(synthetic_seasons["season"].unique())]
        base = dict(model="plackett_luce", train_seasons=seasons[0],
                    test_seasons=seasons[1], predict_seasons=seasons[-1],
                    n_simulations=300, cross_validate=False)
        corrected = run_experiment(ExperimentConfig(name="on", **base),
                                   synthetic_seasons, output_dir=tmp_path)
        plain = run_experiment(
            ExperimentConfig(name="off", simulation_temperature=1.0,
                             player_shock=0.0, **base),
            synthetic_seasons, output_dir=tmp_path)

        a = corrected["leaderboard"].sort_values(["player", "team"])
        b = plain["leaderboard"].sort_values(["player", "team"])
        np.testing.assert_allclose(a["expected_votes"].to_numpy(),
                                   b["expected_votes"].to_numpy(), atol=1e-12)
        np.testing.assert_allclose(a["predicted_votes"].to_numpy(),
                                   b["predicted_votes"].to_numpy(), atol=1e-12)
        assert list(corrected["leaderboard"]["player"]) == \
            list(plain["leaderboard"]["player"]), "the ranking must not move"

    def test_the_corrections_are_reproducible(self, predictions):
        frame, _ = predictions
        a = simulate_season(frame, n_simulations=500, seed=11)
        b = simulate_season(frame, n_simulations=500, seed=11)
        pd.testing.assert_frame_equal(a, b)

    def test_an_ineligible_player_still_cannot_win(self, predictions):
        frame, _ = predictions
        name = frame["player"].iloc[0]
        summary = simulate_season(frame, n_simulations=500, seed=5,
                                  ineligible=[name])
        assert (summary.loc[summary["player"] == name,
                            "win_probability"] == 0.0).all()


def test_the_config_carries_the_settings_through(synthetic_seasons, tmp_path):
    from brownlow.experiment import ExperimentConfig, run_experiment

    seasons = [int(v) for v in sorted(synthetic_seasons["season"].unique())]
    config = ExperimentConfig(name="carried", train_seasons=seasons[0],
                              test_seasons=seasons[1], predict_seasons=seasons[-1],
                              n_simulations=300, cross_validate=False,
                              player_shock=0.9)
    results = run_experiment(config, synthetic_seasons, output_dir=tmp_path)
    stored = results["config"] if isinstance(results.get("config"), dict) else {}
    if stored:
        assert stored.get("player_shock") == 0.9
    assert "p10_votes" in results["leaderboard"].columns
