"""Command line interface.

The three-step workflow, as commands::

    brownlow fetch                                   # get the data
    brownlow backtest --train 2015-2023 --test 2024-2025   # 2. hold out and score
    brownlow predict  --train 2015-2025 --predict 2026     # 3. predict the unknown

Plus the tools for iterating on the model::

    brownlow cv --min-train 5          # walk-forward cross-validation
    brownlow tune                      # grid-search the regularisation
    brownlow run experiments/my.json   # run a saved experiment config
    brownlow compare                   # table of every run so far
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from brownlow.data import download_afltables, load_player_matches
from brownlow.evaluate import (
    backtest,
    leave_one_season_out_cv,
    rolling_origin_cv,
    summarise_cv,
)
from brownlow.experiment import (
    ExperimentConfig,
    compare_experiments,
    run_experiment,
    tune_alpha,
)
from brownlow.model import PlackettLuceModel, WeightedLogisticModel
from brownlow.report import render_site

MODELS = {"plackett_luce": PlackettLuceModel, "logistic": WeightedLogisticModel}


def _show(frame: pd.DataFrame, rows: int = 20, title: Optional[str] = None) -> None:
    if title:
        print(f"\n{title}")
        print("=" * len(title))
    with pd.option_context("display.width", 200, "display.max_columns", 40):
        print(frame.head(rows).to_string(index=False))


def _print_metrics(metrics: dict, title: str = "Holdout metrics") -> None:
    print(f"\n{title}")
    print("=" * len(title))
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key:<32} {value:>10.4f}")
        else:
            print(f"  {key:<32} {value:>10}")


def _add_data_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("data selection")
    group.add_argument("--seasons", help="Seasons to load, e.g. 2015-2026 or 2021,2023")
    group.add_argument("--rounds", help="Home-and-away rounds to keep, e.g. 1-23")
    group.add_argument("--teams", help="Comma-separated team names to keep")
    group.add_argument("--players", help="Comma-separated player names to keep")
    group.add_argument("--include-finals", action="store_true",
                       help="Keep finals (no Brownlow votes are awarded in them)")
    group.add_argument("--data-path", type=Path, help="Read this .rda file instead of the cache")
    group.add_argument("--cache-dir", type=Path, help="Where the downloaded data lives")
    group.add_argument("--refresh", action="store_true", help="Re-download even if cached")


def _load(args, seasons=None, require_complete_votes=False) -> pd.DataFrame:
    return load_player_matches(
        seasons=seasons if seasons is not None else getattr(args, "seasons", None),
        rounds=getattr(args, "rounds", None),
        teams=getattr(args, "teams", None),
        players=getattr(args, "players", None),
        home_and_away_only=not getattr(args, "include_finals", False),
        require_complete_votes=require_complete_votes,
        path=getattr(args, "data_path", None),
        cache_dir=getattr(args, "cache_dir", None),
        refresh=getattr(args, "refresh", False),
    )


# -- commands ---------------------------------------------------------
def cmd_fetch(args) -> int:
    path = download_afltables(args.cache_dir, force=args.refresh)
    size_mb = path.stat().st_size / 1e6
    print(f"AFL Tables data ready: {path} ({size_mb:.1f} MB)")
    print("Source: afltables.com, via the fitzRoy project's data mirror.")
    df = load_player_matches(path=path, seasons=None)
    seasons = sorted(df["season"].unique())
    print(f"{len(df):,} player-match rows across seasons {seasons[0]}-{seasons[-1]}.")
    return 0


def cmd_backtest(args) -> int:
    df = _load(args, require_complete_votes=True)
    model = MODELS[args.model](alpha=args.alpha)
    result = backtest(df, model, train_seasons=_spec(args.train), test_seasons=_spec(args.test))
    print(f"Trained on {result['train_seasons']}, tested on {result['test_seasons']}.")
    _print_metrics(result["metrics"], f"{args.model} holdout metrics")

    if args.compare:
        other = "logistic" if args.model == "plackett_luce" else "plackett_luce"
        alt = backtest(df, MODELS[other](alpha=args.alpha),
                       train_seasons=_spec(args.train), test_seasons=_spec(args.test))
        _print_metrics(alt["metrics"], f"{other} holdout metrics")

    _show(result["model"].coefficient_table(), args.top, "Largest effects")
    leaderboard = result["model"].season_totals(result["predictions"])
    _show(leaderboard, args.top, "Predicted leaderboard (holdout seasons)")
    return 0


def cmd_predict(args) -> int:
    config = ExperimentConfig(
        name=args.name,
        model=args.model,
        alpha=args.alpha,
        train_seasons=args.train,
        test_seasons=args.test,
        predict_seasons=args.predict,
        n_simulations=args.simulations,
        ineligible=args.ineligible.split(",") if args.ineligible else [],
    )
    df = _load(args, seasons=None)
    results = run_experiment(config, df, output_dir=args.output_dir)
    if "holdout_metrics" in results:
        _print_metrics(results["holdout_metrics"], "Holdout check before predicting")
    _show(results["leaderboard"], args.top, f"Projected {args.predict} Brownlow Medal")
    print(f"\nArtefacts written to {results['output_dir']}")
    if not args.no_report:
        page = render_site(docs_path=args.report_path, active_run=config.name)
        print(f"Report page written to {page}")
        print("Commit it and GitHub Pages will publish the updated results.")
    return 0


def cmd_cv(args) -> int:
    df = _load(args, require_complete_votes=True)

    def factory():
        return MODELS[args.model](alpha=args.alpha)

    folds = (
        rolling_origin_cv(df, factory, min_train_seasons=args.min_train, expanding=not args.sliding)
        if args.method == "rolling"
        else leave_one_season_out_cv(df, factory)
    )
    columns = ["fold_test_season", "top1_accuracy", "top3_recall", "exact_order_accuracy",
               "spearman", "top5_overlap", "winner_correct", "log_likelihood_per_match"]
    _show(folds[[c for c in columns if c in folds.columns]], len(folds),
          f"{args.method} cross-validation, model={args.model}")
    print("\nAverage across folds:")
    print(summarise_cv(folds).round(4).to_string())
    return 0


def cmd_tune(args) -> int:
    df = _load(args, require_complete_votes=True)
    alphas = [float(a) for a in args.alphas.split(",")]
    table = tune_alpha(df, alphas=alphas, model=args.model, method=args.method,
                       min_train_seasons=args.min_train)
    _show(table, len(table), "Regularisation search (higher top3_recall is better)")
    print(f"\nBest alpha: {table.iloc[0]['alpha']}")
    return 0


def cmd_run(args) -> int:
    config = ExperimentConfig.from_json(args.config)
    df = _load(args, seasons=None)
    results = run_experiment(config, df, output_dir=args.output_dir)
    if "holdout_metrics" in results:
        _print_metrics(results["holdout_metrics"], f"{config.name}: holdout metrics")
    if "leaderboard" in results:
        _show(results["leaderboard"], args.top, f"{config.name}: leaderboard")
    print(f"\nArtefacts written to {results['output_dir']}")
    if not args.no_report:
        page = render_site(docs_path=args.report_path, active_run=config.name)
        print(f"Report page written to {page}")
    return 0


def cmd_report(args) -> int:
    page = render_site(output_root=args.output_dir, docs_path=args.docs_path)
    print(f"Report page written to {page}")
    print("Commit it and GitHub Pages will publish the updated results.")
    return 0


def cmd_compare(args) -> int:
    table = compare_experiments(args.output_dir)
    if table.empty:
        print("No experiment runs found yet. Run `brownlow predict` or `brownlow run` first.")
        return 0
    _show(table, len(table), "Experiment comparison")
    return 0


def _spec(value):
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="brownlow",
        description="Predict AFL Brownlow Medal votes from player match statistics.",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch = subparsers.add_parser("fetch", help="Download the AFL Tables dataset")
    fetch.add_argument("--cache-dir", type=Path)
    fetch.add_argument("--refresh", action="store_true")
    fetch.set_defaults(func=cmd_fetch)

    common_model = dict(
        model=dict(choices=sorted(MODELS), default="plackett_luce"),
        alpha=dict(type=float, default=1.0),
    )

    back = subparsers.add_parser("backtest", help="Train on past seasons, score on held-out ones")
    back.add_argument("--train", default="2015-2023", help="Training seasons")
    back.add_argument("--test", default="2024-2025", help="Held-out seasons")
    back.add_argument("--model", **common_model["model"])
    back.add_argument("--alpha", **common_model["alpha"])
    back.add_argument("--compare", action="store_true", help="Also score the other model")
    back.add_argument("--top", type=int, default=20)
    _add_data_arguments(back)
    back.set_defaults(func=cmd_backtest)

    predict = subparsers.add_parser("predict", help="Predict a season whose votes are unknown")
    predict.add_argument("--train", default="2015-2025")
    predict.add_argument("--test", default=None, help="Optional holdout check before predicting")
    predict.add_argument("--predict", default="2026", help="Season to project")
    predict.add_argument("--model", **common_model["model"])
    predict.add_argument("--alpha", **common_model["alpha"])
    predict.add_argument("--simulations", type=int, default=10_000)
    predict.add_argument("--ineligible", default="", help="Comma-separated suspended players")
    predict.add_argument("--name", default="latest", help="Name for the run's output folder")
    predict.add_argument("--output-dir", type=Path)
    predict.add_argument("--top", type=int, default=25)
    predict.add_argument("--no-report", action="store_true", help="Skip building docs/index.html")
    predict.add_argument("--report-path", type=Path, help="Where to write the HTML report")
    _add_data_arguments(predict)
    predict.set_defaults(func=cmd_predict)

    cv = subparsers.add_parser("cv", help="Season-wise cross-validation")
    cv.add_argument("--method", choices=["rolling", "loso"], default="rolling")
    cv.add_argument("--min-train", type=int, default=4)
    cv.add_argument("--sliding", action="store_true",
                    help="Fixed-width training window instead of an expanding one")
    cv.add_argument("--model", **common_model["model"])
    cv.add_argument("--alpha", **common_model["alpha"])
    _add_data_arguments(cv)
    cv.set_defaults(func=cmd_cv)

    tune = subparsers.add_parser("tune", help="Grid-search the regularisation strength")
    tune.add_argument("--alphas", default="0.1,0.3,1,3,10,30")
    tune.add_argument("--method", choices=["rolling", "loso"], default="rolling")
    tune.add_argument("--min-train", type=int, default=4)
    tune.add_argument("--model", **common_model["model"])
    _add_data_arguments(tune)
    tune.set_defaults(func=cmd_tune)

    run = subparsers.add_parser("run", help="Run a saved experiment config")
    run.add_argument("config", type=Path)
    run.add_argument("--output-dir", type=Path)
    run.add_argument("--top", type=int, default=25)
    run.add_argument("--no-report", action="store_true")
    run.add_argument("--report-path", type=Path)
    _add_data_arguments(run)
    run.set_defaults(func=cmd_run)

    report = subparsers.add_parser("report", help="Rebuild docs/index.html from every run")
    report.add_argument("--output-dir", type=Path, help="Folder holding the experiment runs")
    report.add_argument("--docs-path", type=Path, help="Where to write index.html")
    report.set_defaults(func=cmd_report)

    compare = subparsers.add_parser("compare", help="Compare every experiment run so far")
    compare.add_argument("--output-dir", type=Path)
    compare.add_argument("--top", type=int, default=50)
    compare.set_defaults(func=cmd_compare)

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError, ImportError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
