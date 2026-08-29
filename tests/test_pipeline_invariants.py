"""End-to-end consistency: the artefacts of a run must agree with each other.

A leaderboard is an aggregate of per-match predictions, a simulation is a draw
from the same probabilities, and the website is a rendering of both. Each is
computed by different code, so each is a chance for them to disagree. These
tests check they do not.
"""

import json

import numpy as np
import pandas as pd
import pytest

from brownlow.experiment import ExperimentConfig, run_experiment
from brownlow.report import collect_run
from brownlow.simulate import simulate_season


@pytest.fixture(scope="module")
def completed_run(tmp_path_factory):
    """A full run projecting an uncounted season, exactly as production does."""
    from brownlow.synthetic import make_synthetic_seasons

    frame = make_synthetic_seasons(seasons=(2020, 2021, 2022), matches_per_season=30,
                                   seed=5)
    frame.loc[frame["season"] == 2022, "votes"] = np.nan
    output = tmp_path_factory.mktemp("runs")
    config = ExperimentConfig(name="pipeline", train_seasons="2020", test_seasons="2021",
                              predict_seasons="2022", n_simulations=400, seed=0)
    results = run_experiment(config, frame, output_dir=output)
    return results, output / "pipeline", frame


class TestArtefacts:
    def test_every_expected_file_is_written(self, completed_run):
        _, run_dir, _ = completed_run
        for filename in ("model.json", "config.json", "coefficients.csv",
                         "metrics.json", "predictions.csv", "leaderboard.csv"):
            path = run_dir / filename
            assert path.exists(), f"{filename} was not written"
            assert path.stat().st_size > 0, f"{filename} is empty"

    def test_predictions_cover_every_match_of_the_projected_season(self, completed_run):
        _, run_dir, frame = completed_run
        predictions = pd.read_csv(run_dir / "predictions.csv")
        expected = frame[frame["season"] == 2022]
        assert len(predictions) == len(expected)
        assert predictions["match_id"].nunique() == expected["match_id"].nunique()

    def test_predictions_award_six_votes_per_match(self, completed_run):
        _, run_dir, _ = completed_run
        predictions = pd.read_csv(run_dir / "predictions.csv")
        for column in ("predicted_votes", "expected_votes"):
            totals = predictions.groupby("match_id")[column].sum()
            np.testing.assert_allclose(totals.to_numpy(), 6.0, atol=1e-6)

    def test_metrics_json_is_readable_and_complete(self, completed_run):
        _, run_dir, _ = completed_run
        payload = json.loads((run_dir / "metrics.json").read_text())
        assert payload["train_seasons"] == [2020]
        assert payload["test_seasons"] == [2021]
        assert payload["predict_seasons"] == [2022]
        assert 0.0 <= payload["holdout_metrics"]["top3_recall"] <= 1.0

    def test_coefficients_cover_every_feature(self, completed_run):
        results, run_dir, _ = completed_run
        coefficients = pd.read_csv(run_dir / "coefficients.csv")
        assert len(coefficients) == len(results["model"].feature_names_)
        assert coefficients["coefficient"].notna().all()


class TestLeaderboardAgreesWithPredictions:
    def test_totals_match_the_sum_of_the_matches(self, completed_run):
        _, run_dir, _ = completed_run
        predictions = pd.read_csv(run_dir / "predictions.csv")
        board = pd.read_csv(run_dir / "leaderboard.csv")

        summed = predictions.groupby(["player", "team"])[
            ["predicted_votes", "expected_votes"]].sum().reset_index()
        merged = board.merge(summed, on=["player", "team"], suffixes=("_board", "_summed"))
        assert len(merged) == len(board), "a leaderboard row had no matching predictions"
        for column in ("predicted_votes", "expected_votes"):
            np.testing.assert_allclose(merged[f"{column}_board"],
                                       merged[f"{column}_summed"], atol=1e-6)

    def test_games_match_the_number_of_matches_played(self, completed_run):
        _, run_dir, _ = completed_run
        predictions = pd.read_csv(run_dir / "predictions.csv")
        board = pd.read_csv(run_dir / "leaderboard.csv")
        counted = predictions.groupby(["player", "team"]).size().rename("counted")
        merged = board.join(counted, on=["player", "team"])
        assert (merged["games"] == merged["counted"]).all()

    def test_the_whole_leaderboard_balances(self, completed_run):
        _, run_dir, _ = completed_run
        predictions = pd.read_csv(run_dir / "predictions.csv")
        board = pd.read_csv(run_dir / "leaderboard.csv")
        matches = predictions["match_id"].nunique()
        assert board["predicted_votes"].sum() == pytest.approx(6 * matches)
        assert board["expected_votes"].sum() == pytest.approx(6 * matches, abs=1e-6)

    def test_every_player_appears_exactly_once(self, completed_run):
        _, run_dir, _ = completed_run
        board = pd.read_csv(run_dir / "leaderboard.csv")
        assert not board.duplicated(subset=["player", "team"]).any()

    def test_the_leaderboard_is_ordered(self, completed_run):
        _, run_dir, _ = completed_run
        board = pd.read_csv(run_dir / "leaderboard.csv")
        assert board["expected_votes"].is_monotonic_decreasing
        assert board["rank"].tolist() == list(range(1, len(board) + 1))


class TestSimulationAgreesWithTheModel:
    def test_win_probabilities_form_a_distribution(self, completed_run):
        results, _, _ = completed_run
        summary = results["simulation"]
        assert summary["win_probability"].sum() == pytest.approx(1.0)
        assert (summary["win_probability"] >= 0).all()
        assert (summary["win_probability"] <= 1).all()

    def test_simulated_totals_award_six_votes_a_match(self, completed_run):
        results, _, _ = completed_run
        matches = results["predictions"]["match_id"].nunique()
        assert results["simulation"]["mean_votes"].sum() == pytest.approx(
            6 * matches, rel=1e-6)

    def test_simulated_means_land_on_the_expected_votes(self, completed_run):
        """Monte Carlo and the closed form must describe the same model.

        With the confidence corrections off, that is: they widen the simulated
        distribution on purpose, so the corrected mean sits nearer the field
        than the closed form does. The projection published on the leaderboard
        is ``expected_votes``, which is the closed form and is unaffected.
        """
        results, _, _ = completed_run
        expected = results["predictions"].groupby(["player", "team"])[
            "expected_votes"].sum().rename("expected")
        uncorrected = simulate_season(results["predictions"], n_simulations=400,
                                      seed=0, temperature=1.0, player_shock=0.0)
        simulated = uncorrected.set_index(["player", "team"])["mean_votes"]
        joined = pd.concat([expected, simulated], axis=1).dropna()
        assert len(joined) > 0
        np.testing.assert_allclose(joined["expected"], joined["mean_votes"], atol=0.5)

    def test_percentiles_bracket_the_median(self, completed_run):
        results, _, _ = completed_run
        summary = results["simulation"]
        assert (summary["p10_votes"] <= summary["median_votes"]).all()
        assert (summary["median_votes"] <= summary["p90_votes"]).all()

    def test_simulated_votes_are_whole_numbers(self, completed_run):
        """Each simulated season awards real 3-2-1s, so every total is an integer.

        The median of an even number of them can land on a midpoint, which is
        numpy averaging the two middle values rather than anything awarding half
        a vote -- so what must hold is that twice the median is whole.

        Checked on the raw simulation. The published one slides each player's
        range onto his projection, which is a real number, so its percentiles
        are not whole either.
        """
        results, _, _ = completed_run
        raw = simulate_season(results["predictions"], n_simulations=400, seed=0,
                              recentre=False)
        medians = raw["median_votes"].to_numpy()
        np.testing.assert_allclose(2.0 * medians, np.round(2.0 * medians))

    def test_an_ineligible_player_cannot_win(self, completed_run):
        results, _, _ = completed_run
        favourite = results["leaderboard"].iloc[0]["player"]
        summary = simulate_season(results["predictions"], n_simulations=300,
                                  ineligible=[favourite], seed=1)
        rows = summary[summary["player"] == favourite]
        assert (rows["win_probability"] == 0).all()
        assert (rows["mean_votes"] > 0).all(), "they should still poll votes"
        assert summary["win_probability"].sum() == pytest.approx(1.0)


class TestReportPayload:
    def test_the_payload_covers_every_player(self, completed_run):
        _, run_dir, _ = completed_run
        board = pd.read_csv(run_dir / "leaderboard.csv")
        payload = collect_run(run_dir, detail_players=None)
        assert len(payload["players"]) == len(board)
        assert len(payload["games"]) == len(board)

    def test_per_match_rows_sum_to_the_season_total(self, completed_run):
        _, run_dir, _ = completed_run
        payload = collect_run(run_dir, detail_players=None)
        for player, games in zip(payload["players"], payload["games"]):
            expected = sum(g[3] or 0 for g in games)
            allocated = sum(g[8] or 0 for g in games)
            assert expected == pytest.approx(player["expected_votes"], abs=0.01)
            assert allocated == pytest.approx(player["predicted_votes"], abs=0.01)

    def test_every_round_is_represented(self, completed_run):
        _, run_dir, _ = completed_run
        predictions = pd.read_csv(run_dir / "predictions.csv")
        payload = collect_run(run_dir)
        assert len(payload["rounds"]) == predictions["round"].nunique()

    def test_rounds_award_six_votes_each_match(self, completed_run):
        _, run_dir, _ = completed_run
        payload = collect_run(run_dir)
        for round_payload in payload["rounds"]:
            for match in round_payload["matches"]:
                allocated = sum(entry["allocated"] for entry in match["top"])
                assert allocated == 6, (
                    f"round {round_payload['round']} {match['match_id']} "
                    f"allocated {allocated} votes in its top players")

    def test_every_match_appears_exactly_once_across_the_rounds(self, completed_run):
        _, run_dir, _ = completed_run
        predictions = pd.read_csv(run_dir / "predictions.csv")
        payload = collect_run(run_dir)
        seen = [m["match_id"] for r in payload["rounds"] for m in r["matches"]]
        assert len(seen) == len(set(seen)), "a match was listed in two rounds"
        assert set(seen) == set(predictions["match_id"].unique())
