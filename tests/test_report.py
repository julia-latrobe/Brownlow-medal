"""Tests for the generated results website.

These check the *data* the page is built from, which is where the bugs live.
Whether a bar is the right shade of blue is not something a test can usefully
say, but whether two players who share a name stay separate is.
"""

import json

import pandas as pd
import pytest

from brownlow.experiment import ExperimentConfig, run_experiment
from brownlow.report import (
    _collect_games,
    _collect_rounds,
    collect_run,
    collect_runs,
    render_site,
)


@pytest.fixture
def run_dir(synthetic_seasons, tmp_path):
    """A completed experiment run on disk, projecting an unscored season."""
    future = synthetic_seasons.copy()
    future.loc[future["season"] == 2022, "votes"] = float("nan")
    config = ExperimentConfig(
        name="site-run",
        train_seasons="2020-2021",
        test_seasons=None,
        predict_seasons="2022",
        n_simulations=200,
        annotations={"Geelong Player 01": "Omitted"},
    )
    run_experiment(config, future, output_dir=tmp_path)
    return tmp_path / "site-run"


class TestCollectGames:
    def test_one_entry_per_leaderboard_row(self, run_dir):
        payload = collect_run(run_dir)
        assert len(payload["games"]) == len(payload["players"])

    def test_every_match_is_represented(self, run_dir):
        payload = collect_run(run_dir)
        predictions = pd.read_csv(run_dir / "predictions.csv")
        assert sum(len(g) for g in payload["games"]) == len(predictions)

    def test_rows_are_compact_numeric_arrays(self, run_dir):
        """Per-match rows are the bulk of the page, so they stay as arrays."""
        payload = collect_run(run_dir)
        first = next(g for g in payload["games"] if g)
        assert len(first[0]) == 8  # round, opponent, home, expected, p3, p2, p1, actual

    def test_expected_votes_match_the_leaderboard_total(self, run_dir):
        payload = collect_run(run_dir)
        for player, games in zip(payload["players"], payload["games"]):
            total = sum(g[3] or 0 for g in games)
            assert total == pytest.approx(player["predicted_votes"], abs=0.01)

    def test_opponents_are_indexed_not_repeated(self, run_dir):
        payload = collect_run(run_dir)
        assert payload["opponents"]
        for games in payload["games"]:
            for game in games:
                assert 0 <= game[1] < len(payload["opponents"])

    def test_players_sharing_a_name_keep_separate_games(self, tmp_path):
        """The AFL has had two Bailey Williamses at once. They are not one player."""
        frame = pd.DataFrame({
            "round": [1, 1, 2, 2],
            "match_id": ["m1", "m1", "m2", "m2"],
            "player": ["Sam Smith", "Sam Smith", "Sam Smith", "Sam Smith"],
            "team": ["Geelong", "Carlton", "Geelong", "Carlton"],
            "opponent": ["Carlton", "Geelong", "Sydney", "Hawthorn"],
            "is_home": [1, 0, 1, 0],
            "predicted_votes": [2.0, 1.0, 2.5, 0.5],
            "p_3_votes": [0.5, 0.2, 0.6, 0.1],
            "p_2_votes": [0.3, 0.2, 0.2, 0.1],
            "p_1_vote": [0.1, 0.2, 0.1, 0.1],
            "votes": [None, None, None, None],
        })
        path = tmp_path / "predictions.csv"
        frame.to_csv(path, index=False)
        board = pd.DataFrame({"player": ["Sam Smith", "Sam Smith"],
                              "team": ["Geelong", "Carlton"]})

        games, _ = _collect_games(path, board)
        assert len(games) == 2
        assert [g[3] for g in games[0]] == [2.0, 2.5]   # the Geelong one
        assert [g[3] for g in games[1]] == [1.0, 0.5]   # the Carlton one


class TestCollectRounds:
    def test_one_entry_per_round(self, run_dir):
        payload = collect_run(run_dir)
        predictions = pd.read_csv(run_dir / "predictions.csv")
        assert len(payload["rounds"]) == predictions["round"].nunique()

    def test_rounds_are_in_order(self, run_dir):
        payload = collect_run(run_dir)
        numbers = [float(r["sort"]) for r in payload["rounds"]]
        assert numbers == sorted(numbers)

    def test_matches_are_in_the_order_they_were_played(self, run_dir):
        """A round view should read like the weekend did, not alphabetically."""
        payload = collect_run(run_dir)
        for round_payload in payload["rounds"]:
            keys = [(m["date"] or "", m["start"] or 0) for m in round_payload["matches"]]
            assert keys == sorted(keys)

    def test_each_match_names_both_teams(self, run_dir):
        payload = collect_run(run_dir)
        for round_payload in payload["rounds"]:
            for match in round_payload["matches"]:
                assert match["home"] and match["away"]
                assert match["home"] != match["away"]

    def test_top_players_are_ranked_for_the_three_two_one(self, run_dir):
        payload = collect_run(run_dir)
        for round_payload in payload["rounds"]:
            for match in round_payload["matches"]:
                expected = [entry["expected"] for entry in match["top"]]
                assert expected == sorted(expected, reverse=True)
                assert len(match["top"]) >= 3

    def test_survives_a_predictions_file_from_an_older_version(self, tmp_path):
        """Older runs have no is_home column; the round view must still build."""
        frame = pd.DataFrame({
            "round": [1, 1],
            "match_id": ["2026-R1-Geelong-v-Carlton"] * 2,
            "player": ["A Player", "B Player"],
            "team": ["Geelong", "Carlton"],
            "predicted_votes": [2.0, 1.0],
        })
        path = tmp_path / "old.csv"
        frame.to_csv(path, index=False)
        rounds = _collect_rounds(path)
        assert rounds[0]["matches"][0]["home"] == "Geelong"
        assert rounds[0]["matches"][0]["away"] == "Carlton"


class TestAnnotations:
    def test_annotations_reach_the_page(self, run_dir):
        payload = collect_run(run_dir)
        assert payload["annotations"] == {"Geelong Player 01": "Omitted"}

    def test_annotations_do_not_change_any_projection(self, synthetic_seasons, tmp_path):
        """A label beside a name is presentation, never an input to the model."""
        base = ExperimentConfig(name="plain", train_seasons="2020-2021",
                                test_seasons="2022", n_simulations=100)
        labelled = ExperimentConfig(name="labelled", train_seasons="2020-2021",
                                    test_seasons="2022", n_simulations=100,
                                    annotations={"Geelong Player 01": "Omitted"})
        a = run_experiment(base, synthetic_seasons, output_dir=tmp_path)
        b = run_experiment(labelled, synthetic_seasons, output_dir=tmp_path)
        assert a["holdout_metrics"] == b["holdout_metrics"]

    def test_no_annotations_by_default(self, synthetic_seasons, tmp_path):
        config = ExperimentConfig(name="bare", train_seasons="2020-2021",
                                  test_seasons="2022", n_simulations=100)
        run_experiment(config, synthetic_seasons, output_dir=tmp_path)
        assert collect_run(tmp_path / "bare")["annotations"] == {}


class TestRenderSite:
    def test_page_contains_every_view(self, run_dir, tmp_path):
        page = render_site(output_root=run_dir.parent, docs_path=tmp_path / "index.html")
        html = page.read_text()
        for element in ("season-view", "player-view", "round-view", "team-view"):
            assert f'id="{element}"' in html

    def test_page_wires_up_the_navigation(self, run_dir, tmp_path):
        page = render_site(output_root=run_dir.parent, docs_path=tmp_path / "index.html")
        html = page.read_text()
        assert 'id="player-search"' in html
        assert 'id="round-select"' in html
        assert 'id="home-link"' in html
        assert "#player=" in html and "#team=" in html and "#round=" in html

    def test_page_is_self_contained(self, run_dir, tmp_path):
        """No CDN, no build step -- it has to work opened straight off disk."""
        page = render_site(output_root=run_dir.parent, docs_path=tmp_path / "index.html")
        html = page.read_text()
        assert "<script src=" not in html
        assert 'rel="stylesheet"' not in html

    def test_run_payload_is_valid_json(self, run_dir, tmp_path):
        page = render_site(output_root=run_dir.parent, docs_path=tmp_path / "index.html")
        html = page.read_text()
        start = html.index("const RUNS = ") + len("const RUNS = ")
        end = html.index(";</script>", start)
        runs = json.loads(html[start:end])
        assert runs and runs[0]["players"]

    def test_runs_are_listed_most_recent_first(self, synthetic_seasons, tmp_path):
        for name in ("older", "newer"):
            run_experiment(
                ExperimentConfig(name=name, train_seasons="2020-2021",
                                 test_seasons="2022", n_simulations=100),
                synthetic_seasons, output_dir=tmp_path,
            )
        assert [r["name"] for r in collect_runs(tmp_path)][0] == "newer"
