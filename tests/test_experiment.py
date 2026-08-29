"""End-to-end tests: an experiment run should produce every artefact."""

import json

import pytest

from brownlow.experiment import (
    ExperimentConfig,
    compare_experiments,
    run_experiment,
    tune_alpha,
)
from brownlow.report import collect_runs, render_site


@pytest.fixture
def config():
    return ExperimentConfig(
        name="test-run",
        model="plackett_luce",
        alpha=1.0,
        train_seasons="2020-2021",
        test_seasons="2022",
        predict_seasons=None,
        n_simulations=200,
        notes="A run used by the test suite.",
    )


class TestExperimentConfig:
    def test_survives_a_json_round_trip(self, config, tmp_path):
        path = config.to_json(tmp_path / "config.json")
        assert ExperimentConfig.from_json(path) == config

    def test_builds_the_named_model(self, config):
        from brownlow.model import PlackettLuceModel, WeightedLogisticModel

        assert isinstance(config.build_model(), PlackettLuceModel)
        config.model = "logistic"
        assert isinstance(config.build_model(), WeightedLogisticModel)

    def test_rejects_an_unknown_model(self, config):
        config.model = "random forest"
        with pytest.raises(ValueError, match="Unknown model"):
            config.build_model()

    def test_feature_overrides_are_applied(self, config):
        config.features = {"include_shares": False}
        assert config.build_model().feature_config.include_shares is False


class TestRunExperiment:
    def test_writes_the_standard_artefacts(self, config, synthetic_seasons, tmp_path):
        run_experiment(config, synthetic_seasons, output_dir=tmp_path)
        run_dir = tmp_path / "test-run"
        for filename in ("model.json", "config.json", "coefficients.csv",
                         "metrics.json", "predictions.csv", "leaderboard.csv"):
            assert (run_dir / filename).exists(), f"{filename} was not written"

    def test_metrics_file_describes_the_run(self, config, synthetic_seasons, tmp_path):
        run_experiment(config, synthetic_seasons, output_dir=tmp_path)
        payload = json.loads((tmp_path / "test-run" / "metrics.json").read_text())
        assert payload["train_seasons"] == [2020, 2021]
        assert payload["test_seasons"] == [2022]
        assert payload["config"]["notes"] == "A run used by the test suite."
        assert 0.0 <= payload["holdout_metrics"]["top3_recall"] <= 1.0

    def test_compares_against_the_baseline(self, config, synthetic_seasons, tmp_path):
        results = run_experiment(config, synthetic_seasons, output_dir=tmp_path)
        assert len(results["comparison"]) == 2

    def test_predicting_an_unscored_season_still_works(self, config, synthetic_seasons, tmp_path):
        """The real use: the season being predicted has no votes yet."""
        future = synthetic_seasons.copy()
        future.loc[future["season"] == 2022, "votes"] = float("nan")
        config.test_seasons = None
        config.predict_seasons = "2022"
        results = run_experiment(config, future, output_dir=tmp_path)
        board = results["leaderboard"]
        assert len(board) > 0
        assert "win_probability" in board.columns
        assert board["win_probability"].sum() == pytest.approx(1.0)

    def test_refuses_overlapping_train_and_test(self, config, synthetic_seasons, tmp_path):
        config.test_seasons = "2021"
        with pytest.raises(ValueError, match="overlap"):
            run_experiment(config, synthetic_seasons, output_dir=tmp_path)


class TestCompareExperiments:
    def test_lists_every_run(self, config, synthetic_seasons, tmp_path):
        run_experiment(config, synthetic_seasons, output_dir=tmp_path)
        config.name = "second-run"
        config.alpha = 5.0
        run_experiment(config, synthetic_seasons, output_dir=tmp_path)

        table = compare_experiments(tmp_path)
        assert set(table["experiment"]) == {"test-run", "second-run"}
        assert "top3_recall" in table.columns

    def test_empty_when_nothing_has_run(self, tmp_path):
        assert compare_experiments(tmp_path).empty


class TestTuneAlpha:
    def test_returns_a_row_per_alpha(self, synthetic_seasons):
        table = tune_alpha(synthetic_seasons, alphas=(0.5, 5.0), min_train_seasons=1)
        assert list(table["alpha"].sort_values()) == [0.5, 5.0]
        assert "top3_recall" in table.columns


class TestReport:
    def test_builds_a_page_from_the_runs_on_disk(self, config, synthetic_seasons, tmp_path):
        run_experiment(config, synthetic_seasons, output_dir=tmp_path)
        page = render_site(output_root=tmp_path, docs_path=tmp_path / "site" / "index.html")
        html = page.read_text()
        assert "<title>" in html
        assert "test-run" in html
        assert "const RUNS" in html

    def test_collects_the_run_payload(self, config, synthetic_seasons, tmp_path):
        run_experiment(config, synthetic_seasons, output_dir=tmp_path)
        runs = collect_runs(tmp_path)
        assert len(runs) == 1
        assert runs[0]["name"] == "test-run"
        assert runs[0]["players"]
        assert runs[0]["coefficients"]

    def test_errors_clearly_when_there_is_nothing_to_report(self, tmp_path):
        with pytest.raises(ValueError, match="No experiment runs"):
            render_site(output_root=tmp_path, docs_path=tmp_path / "index.html")
