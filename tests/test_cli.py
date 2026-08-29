"""Tests for the command line entry points."""

import pytest

from brownlow.cli import build_parser, main


class TestParser:
    def test_every_command_is_registered(self):
        parser = build_parser()
        commands = parser._subparsers._group_actions[0].choices
        assert {"fetch", "backtest", "predict", "cv", "tune", "run", "report", "compare"} <= set(
            commands
        )

    def test_backtest_defaults_hold_out_the_recent_seasons(self):
        args = build_parser().parse_args(["backtest"])
        assert args.train == "2015-2023"
        assert args.test == "2024-2025"

    def test_predict_defaults_to_the_current_season(self):
        args = build_parser().parse_args(["predict"])
        assert args.predict == "2026"
        assert args.model == "plackett_luce"

    def test_data_filters_are_available(self):
        args = build_parser().parse_args(
            ["backtest", "--seasons", "2015-2020", "--teams", "Geelong", "--rounds", "1-10"]
        )
        assert args.seasons == "2015-2020"
        assert args.teams == "Geelong"
        assert args.rounds == "1-10"

    def test_a_command_is_required(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args([])


class TestMain:
    def test_reports_a_missing_config_cleanly(self, capsys):
        """Errors should be a readable message, not a traceback."""
        code = main(["run", "does-not-exist.json"])
        assert code == 1

    def test_compare_with_no_runs_is_not_an_error(self, tmp_path, capsys):
        assert main(["compare", "--output-dir", str(tmp_path)]) == 0
        assert "No experiment runs found" in capsys.readouterr().out
