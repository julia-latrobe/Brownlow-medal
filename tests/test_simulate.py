"""Tests for the season Monte Carlo."""

import numpy as np
import pytest

from brownlow.model import PlackettLuceModel
from brownlow.simulate import simulate_season


@pytest.fixture(scope="module")
def simulated(request):
    season = request.getfixturevalue("small_season")
    model = PlackettLuceModel().fit(season)
    predictions = model.predict(season)
    return predictions, simulate_season(predictions, n_simulations=2000, seed=1)


class TestSimulateSeason:
    def test_win_probabilities_form_a_distribution(self, simulated):
        _, summary = simulated
        assert summary["win_probability"].sum() == pytest.approx(1.0)
        assert (summary["win_probability"] >= 0).all()

    def test_simulated_votes_match_the_expected_votes(self, simulated):
        """Monte Carlo means should land on the closed-form expectation."""
        predictions, summary = simulated
        expected = predictions.groupby("player")["predicted_votes"].sum()
        merged = summary.set_index("player")["mean_votes"]
        common = expected.index.intersection(merged.index)
        np.testing.assert_allclose(
            expected.loc[common].to_numpy(), merged.loc[common].to_numpy(), atol=0.35
        )

    def test_total_votes_awarded_is_six_per_match(self, simulated):
        predictions, summary = simulated
        n_matches = predictions["match_id"].nunique()
        assert summary["mean_votes"].sum() == pytest.approx(6 * n_matches, rel=1e-6)

    def test_percentiles_bracket_the_median(self, simulated):
        _, summary = simulated
        assert (summary["p10_votes"] <= summary["median_votes"]).all()
        assert (summary["median_votes"] <= summary["p90_votes"]).all()

    def test_is_reproducible(self, small_season):
        model = PlackettLuceModel().fit(small_season)
        predictions = model.predict(small_season)
        a = simulate_season(predictions, n_simulations=500, seed=42)
        b = simulate_season(predictions, n_simulations=500, seed=42)
        np.testing.assert_allclose(a["win_probability"], b["win_probability"])

    def test_ineligible_players_cannot_win(self, small_season):
        """A suspended player still polls votes but cannot take the medal."""
        model = PlackettLuceModel().fit(small_season)
        predictions = model.predict(small_season)
        favourite = (
            predictions.groupby("player")["predicted_votes"].sum().idxmax()
        )
        summary = simulate_season(predictions, n_simulations=1000,
                                  ineligible=[favourite], seed=1)
        row = summary[summary["player"] == favourite].iloc[0]
        assert row["win_probability"] == 0.0
        assert not row["eligible"]
        assert row["mean_votes"] > 0  # they still poll
        assert summary["win_probability"].sum() == pytest.approx(1.0)

    def test_rejects_predictions_without_probabilities(self, small_season):
        with pytest.raises(ValueError, match="missing columns"):
            simulate_season(small_season)

    def test_players_sharing_a_name_are_kept_apart(self, small_season):
        """The AFL has had two Bailey Williamses at once, at different clubs.

        Keying on the name alone would pool their votes, inflating one total and
        giving both the same wrong win probability.
        """
        model = PlackettLuceModel().fit(small_season)
        predictions = model.predict(small_season)

        # Rename two players at different clubs to share a name.
        teams = predictions["team"].unique()[:2]
        renamed = predictions.copy()
        for team in teams:
            on_team = renamed["team"] == team
            victim = renamed.loc[on_team, "player"].iloc[0]
            renamed.loc[renamed["player"] == victim, "player"] = "Shared Name"

        summary = simulate_season(renamed, n_simulations=500, seed=3)
        rows = summary[summary["player"] == "Shared Name"]
        assert len(rows) == 2, "the two players should stay separate"
        assert set(rows["team"]) == set(teams)

        # Their combined total should still equal what the model expects of them.
        expected = renamed[renamed["player"] == "Shared Name"]["predicted_votes"].sum()
        assert rows["mean_votes"].sum() == pytest.approx(expected, abs=0.6)
        assert summary["win_probability"].sum() == pytest.approx(1.0)
