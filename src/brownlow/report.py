"""Build the results website.

Every experiment run writes a folder under ``data/output/``. This module reads
all of them and produces a single self-contained ``docs/index.html``: no build
step, no JavaScript dependencies, no CDN. Point GitHub Pages at ``docs/`` and
each merge to ``main`` republishes the results, so the model's output lives in
the repository's history rather than only in someone's terminal.

The page lets you switch between runs and filter to a single team, so the charts
are drawn in the browser from data embedded in the file.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# Palette roles. Light and dark are each chosen for their own surface rather
# than flipped, and every pair used together is validated for colour-vision
# deficiency separation.
LIGHT = {
    "surface": "#fcfcfb", "panel": "#ffffff", "border": "#e4e3df",
    "text": "#0b0b0b", "muted": "#52514e", "faint": "#8a8985",
    "series-1": "#2a78d6", "series-2": "#eb6834",
    "positive": "#2a78d6", "negative": "#e34948",
    "neutral": "#f0efec", "grid": "#ecebe7", "medal": "#b8860b",
}
DARK = {
    "surface": "#1a1a19", "panel": "#232322", "border": "#383835",
    "text": "#ffffff", "muted": "#c3c2b7", "faint": "#8d8c85",
    "series-1": "#3987e5", "series-2": "#d95926",
    "positive": "#3987e5", "negative": "#e66767",
    "neutral": "#383835", "grid": "#2c2c2a", "medal": "#eda100",
}


def default_output_root() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "output"


def default_docs_path() -> Path:
    return Path(__file__).resolve().parents[2] / "docs" / "index.html"


def _clean(value: Any) -> Any:
    if isinstance(value, float) and (value != value):  # NaN
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


#: How many players keep full per-match detail on the page. See _collect_games.
DEFAULT_DETAIL_PLAYERS = 100


def collect_run(run_dir: Path, detail_players: Optional[int] = DEFAULT_DETAIL_PLAYERS
                ) -> Optional[Dict[str, Any]]:
    """Read one experiment folder into the payload the page needs."""
    metrics_file = run_dir / "metrics.json"
    leaderboard_file = run_dir / "leaderboard.csv"
    if not metrics_file.exists() or not leaderboard_file.exists():
        return None

    metrics = json.loads(metrics_file.read_text())
    board = pd.read_csv(leaderboard_file)

    columns = [
        c for c in (
            "player", "team", "predicted_votes", "expected_votes", "games", "actual_votes",
            "win_probability", "top5_probability", "mean_votes",
            "p10_votes", "p90_votes",
        ) if c in board.columns
    ]
    players = [
        {k: _clean(v) for k, v in row.items()}
        for row in board[columns].to_dict(orient="records")
    ]

    # Per-match rows power the individual player pages. They are the biggest
    # thing on the page by far -- roughly 9,500 rows a season -- so they are
    # stored as parallel arrays of numbers rather than repeating a key name on
    # every row. That keeps a full season under a few hundred KB.
    games: List[List[List[Any]]] = []
    opponents: List[str] = []
    rounds: List[Dict[str, Any]] = []
    predictions_file = run_dir / "predictions.csv"
    if predictions_file.exists():
        games, opponents = _collect_games(predictions_file, board, detail_players)
        rounds = _collect_rounds(predictions_file)

    coefficients: List[Dict[str, Any]] = []
    coefficient_file = run_dir / "coefficients.csv"
    if coefficient_file.exists():
        frame = pd.read_csv(coefficient_file).head(16)
        coefficients = [
            {"feature": r["feature"], "coefficient": float(r["coefficient"])}
            for _, r in frame.iterrows()
        ]

    config = metrics.get("config", {})
    predict_seasons = metrics.get("predict_seasons") or []
    test_seasons = metrics.get("test_seasons") or []
    label = ", ".join(str(s) for s in (predict_seasons or test_seasons)) or "--"

    return {
        "name": metrics.get("name", run_dir.name),
        "season_label": label,
        "scored": bool(predict_seasons) is False,
        "model": config.get("model"),
        "alpha": config.get("alpha"),
        "notes": config.get("notes", ""),
        "train_seasons": metrics.get("train_seasons"),
        "test_seasons": test_seasons,
        "predict_seasons": predict_seasons,
        "simulations": config.get("n_simulations"),
        "holdout": metrics.get("holdout_metrics") or {},
        "cv": metrics.get("cv_metrics") or {},
        "comparison": metrics.get("comparison") or {},
        "coefficients": coefficients,
        "players": players,
        "games": games,
        "opponents": opponents,
        "rounds": rounds,
        "detail_players": detail_players,
        # Presentational labels beside a player's name. They never touch the model.
        "annotations": config.get("annotations") or {},
    }


def _round_label(value: Any) -> Any:
    """Round labels are mostly numeric, but finals and Opening Round are words."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def _collect_games(predictions_file: Path, board: pd.DataFrame, limit: Optional[int] = None):
    """Build each player's round-by-round projection, in the leaderboard's order.

    Returns a list parallel to ``board``'s rows, each holding that player's
    matches as ``[round, opponent index, is home, expected votes, p3, p2, p1,
    actual votes, allocated votes]``. Players are identified by name *and* club,
    because two players can share a name.

    ``limit`` keeps per-match detail for only the leading players. These rows are
    most of the page's weight -- a season is roughly 9,500 of them per run -- and
    a player ranked 400th projected to poll a fraction of a vote is not someone
    anyone opens. Players past the cut still get their season summary; the page
    says so rather than showing an empty table.
    """
    frame = pd.read_csv(predictions_file)
    if "player" not in frame.columns:
        return [], []

    opponent_names = sorted(frame["opponent"].dropna().astype(str).unique()) \
        if "opponent" in frame.columns else []
    opponent_index = {name: i for i, name in enumerate(opponent_names)}

    has_team = "team" in frame.columns and "team" in board.columns
    keys = frame["player"].astype(str)
    if has_team:
        keys = keys + "|" + frame["team"].astype(str)
    frame = frame.assign(_key=keys)

    sort_columns = [c for c in ("round", "match_id") if c in frame.columns]
    if sort_columns:
        frame = frame.sort_values(["_key", *sort_columns], key=lambda c: (
            pd.to_numeric(c, errors="coerce").fillna(1e9) if c.name == "round" else c
        ))

    def cell(row, column, digits):
        value = row.get(column)
        if value is None or (isinstance(value, float) and value != value):
            return None
        return round(float(value), digits)

    grouped = {}
    for key, group in frame.groupby("_key", sort=False):
        rows = []
        for record in group.to_dict(orient="records"):
            rows.append([
                _round_label(record.get("afl_round", record.get("round"))),
                opponent_index.get(str(record.get("opponent")), -1),
                1 if record.get("is_home") else 0,
                cell(record, "expected_votes", 4),
                cell(record, "p_3_votes", 4),
                cell(record, "p_2_votes", 4),
                cell(record, "p_1_vote", 4),
                cell(record, "votes", 0),
                cell(record, "predicted_votes", 0),
            ])
        grouped[key] = rows

    board_keys = board["player"].astype(str)
    if has_team:
        board_keys = board_keys + "|" + board["team"].astype(str)
    keys = list(board_keys)
    return (
        [grouped.get(key, []) if limit is None or i < limit else []
         for i, key in enumerate(keys)],
        opponent_names,
    )


def _collect_rounds(predictions_file: Path, top_n: int = 5):
    """Group the projection by round, then by match, in the order they were played.

    For each match we keep the players most likely to poll, so the round view can
    show a projected 3-2-1 alongside the runners-up.
    """
    frame = pd.read_csv(predictions_file)
    needed = {"round", "match_id", "player", "predicted_votes"}
    if not needed <= set(frame.columns):
        return []
    rank_column = "expected_votes" if "expected_votes" in frame.columns else "predicted_votes"

    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["_round_sort"] = pd.to_numeric(frame["round"], errors="coerce").fillna(9999)
    # Show the AFL's own round label. Its numbering differs from the source's
    # from 2024 on, and a page people follow during the count has to match the
    # numbers being read out.
    if "afl_round" not in frame.columns:
        frame["afl_round"] = frame["round"].astype(str)

    rounds = []
    for (round_sort, round_label), round_rows in frame.groupby(
        ["_round_sort", "afl_round"], sort=True
    ):
        matches = []
        for match_id, match_rows in round_rows.groupby("match_id", sort=False):
            first = match_rows.iloc[0]
            # A predictions.csv written by an older version may have no
            # is_home column, in which case fall back to the match_id, which
            # always reads "<season>-R<round>-<home>-v-<away>".
            if "is_home" in match_rows.columns:
                home_flag = match_rows["is_home"] == 1
                home_team = match_rows.loc[home_flag, "team"]
                away_team = match_rows.loc[~home_flag, "team"]
                home_name = str(home_team.iloc[0]) if len(home_team) else ""
                away_name = str(away_team.iloc[0]) if len(away_team) else ""
            else:
                home_name, _, away_name = str(match_id).partition("-v-")
                home_name = home_name.split("-", 2)[-1]
            # Rank on the continuous expectation. The 3-2-1 allocation ties
            # everyone outside the top three on zero, so the runners-up would
            # come back in arbitrary order.
            ranked = match_rows.sort_values(rank_column, ascending=False).head(top_n)

            date_value = first.get("date")
            start = first.get("local_start_time")
            matches.append({
                "match_id": str(match_id),
                "home": home_name,
                "away": away_name,
                "date": None if pd.isna(date_value) else pd.Timestamp(date_value).strftime("%Y-%m-%d"),
                "start": None if pd.isna(start) else int(start),
                "venue": None if pd.isna(first.get("venue")) else str(first.get("venue")),
                "top": [
                    {
                        "player": str(row["player"]),
                        "team": str(row.get("team", "")),
                        "expected": round(float(row.get("expected_votes",
                                                        row["predicted_votes"])), 3),
                        "allocated": int(row.get("predicted_votes", 0) or 0),
                        "p3": round(float(row.get("p_3_votes", 0) or 0), 4),
                        "actual": (
                            None if pd.isna(row.get("votes")) else int(row.get("votes"))
                        ),
                    }
                    for _, row in ranked.iterrows()
                ],
            })

        # Order matches the way the round was actually played.
        matches.sort(key=lambda m: (m["date"] or "", m["start"] if m["start"] is not None else 0))
        rounds.append({
            "round": str(round_label),
            "sort": float(round_sort),
            "matches": matches,
        })

    rounds.sort(key=lambda r: r["sort"])
    return rounds


def collect_runs(output_root: Optional[Path] = None,
                 detail_players: Optional[int] = DEFAULT_DETAIL_PLAYERS
                 ) -> List[Dict[str, Any]]:
    root = Path(output_root) if output_root else default_output_root()
    runs = []
    directories = [d for d in (root.iterdir() if root.exists() else []) if d.is_dir()]
    # Most recently run first, so the dropdown opens on what you just did
    # rather than on whichever folder happens to sort first.
    directories.sort(key=lambda d: (d / "metrics.json").stat().st_mtime
                     if (d / "metrics.json").exists() else 0, reverse=True)
    for run_dir in directories:
        payload = collect_run(run_dir, detail_players)
        if payload:
            runs.append(payload)
    return rank_runs(runs)


def run_score(run: Dict[str, Any]):
    """How good a run is, for ordering. Higher is better.

    Uses cross-validated top-3 recall where the run recorded it, and the single
    held-out season otherwise. The distinction matters: scenarios here differ by
    fractions of a percentage point, and one season is a noisy enough measure to
    crown the wrong one.

    Top-3 recall is the metric because getting the right three players in the
    wrong order is nearly right. Log-likelihood breaks ties, rewarding a model
    for being correctly confident rather than merely correct.
    """
    cv = run.get("cv") or {}
    holdout = run.get("holdout") or {}
    source = cv if cv.get("top3_recall") is not None else holdout
    recall = source.get("top3_recall")
    likelihood = source.get("log_likelihood_per_match")
    return (
        recall if recall is not None else -1.0,
        likelihood if likelihood is not None else -1e9,
    )


def rank_runs(runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Order runs best first, and mark which one leads and on what basis."""
    ordered = sorted(runs, key=run_score, reverse=True)
    for position, run in enumerate(ordered):
        run["is_best"] = position == 0 and run_score(run)[0] >= 0
        run["ranked_on"] = (
            "cross-validation" if (run.get("cv") or {}).get("top3_recall") is not None
            else "the held-out season"
        )
    return ordered


def _logo() -> str:
    """An original mark for the site: a football and a medal.

    Deliberately not the AFL or Brownlow Medal marks -- both are registered
    trademarks, and using them would imply an official association this project
    does not have. Swap this for licensed artwork if you have permission.
    """
    return (
        '<svg class="logo" viewBox="0 0 48 40" aria-hidden="true" focusable="false">'
        '<ellipse cx="21" cy="20" rx="19" ry="12" fill="none" '
        'stroke="var(--series-1)" stroke-width="2.5"/>'
        '<line x1="9" y1="20" x2="33" y2="20" stroke="var(--series-1)" stroke-width="2"/>'
        '<line x1="15" y1="16" x2="15" y2="24" stroke="var(--series-1)" stroke-width="1.6"/>'
        '<line x1="21" y1="16" x2="21" y2="24" stroke="var(--series-1)" stroke-width="1.6"/>'
        '<line x1="27" y1="16" x2="27" y2="24" stroke="var(--series-1)" stroke-width="1.6"/>'
        '<circle cx="37" cy="26" r="9" fill="var(--surface)"/>'
        '<circle cx="37" cy="26" r="7.5" fill="none" stroke="var(--medal)" stroke-width="2.5"/>'
        '<circle cx="37" cy="26" r="3" fill="var(--medal)"/>'
        "</svg>"
    )


def _favicon() -> str:
    """Inline SVG favicon so the page stays a single self-contained file."""
    svg = (
        "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'>"
        "<rect width='32' height='32' rx='7' fill='%232a78d6'/>"
        "<circle cx='16' cy='16' r='9' fill='none' stroke='%23eda100' stroke-width='3'/>"
        "<circle cx='16' cy='16' r='3.5' fill='%23eda100'/></svg>"
    )
    return "data:image/svg+xml," + svg.replace("#", "%23").replace('"', "'")


def _variables(colours: Dict[str, str]) -> str:
    return "\n".join(f"  --{key}: {value};" for key, value in colours.items())


def _css() -> str:
    return f""":root {{
  color-scheme: light;
{_variables(LIGHT)}
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    color-scheme: dark;
{_variables(DARK)}
  }}
}}
:root[data-theme="dark"] {{
  color-scheme: dark;
{_variables(DARK)}
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0; background: var(--surface); color: var(--text);
  font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}}
.wrap {{ max-width: 960px; margin: 0 auto; padding: 38px 20px 72px; }}
header {{ border-bottom: 1px solid var(--border); padding-bottom: 20px; }}
h1 {{ font-size: 30px; margin: 0; letter-spacing: -0.02em; }}
/* The title doubles as the way home, the way a site logo normally does. */
.brand {{ display: flex; align-items: center; gap: 13px; margin-bottom: 7px;
          text-decoration: none; color: inherit; width: fit-content; }}
.brand:hover h1 {{ color: var(--series-1); }}
.logo {{ width: 50px; height: 42px; flex: none; }}
h2 {{ font-size: 19px; margin: 38px 0 4px; letter-spacing: -0.01em; }}
.sub {{ color: var(--muted); margin: 0; font-size: 14px; }}
.note {{ color: var(--muted); font-size: 13.5px; margin: 4px 0 14px; max-width: 70ch; }}
.controls {{
  display: flex; flex-wrap: wrap; gap: 14px; align-items: flex-end;
  margin: 20px 0 4px; padding: 14px 16px;
  background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
}}
.control {{ display: flex; flex-direction: column; gap: 5px; }}
.control label {{ font-size: 11.5px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--faint); }}
select {{
  font: inherit; font-size: 14px; padding: 6px 10px; min-width: 190px;
  background: var(--surface); color: var(--text);
  border: 1px solid var(--border); border-radius: 7px;
}}
/* Sits under the controls as its own line: the notes can run long, and reading
   a paragraph ragged-left in a narrow right-hand column is unpleasant. */
.run-meta {{ font-size: 13px; color: var(--muted); line-height: 1.5;
             margin: 10px 2px 0; max-width: 84ch; }}
.run-meta .pill {{ margin-right: 6px; }}
.tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(178px, 1fr)); gap: 12px; margin: 16px 0 8px; }}
.tile {{ background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 13px 15px; }}
.tile-label {{ font-size: 11.5px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--faint); }}
.tile-value {{ font-size: 24px; font-weight: 600; margin: 4px 0 2px; letter-spacing: -0.02em; }}
.tile-note {{ font-size: 12.5px; color: var(--muted); }}
.panel {{ background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 14px 12px; overflow-x: auto; }}
.chart {{ width: 100%; height: auto; display: block; min-width: 540px; }}
.axis-label {{ fill: var(--text); font-size: 12.5px; }}
.axis-label.mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11.5px; }}
.axis-sublabel {{ fill: var(--faint); font-size: 11px; }}
.value-label {{ fill: var(--muted); font-size: 12px; font-variant-numeric: tabular-nums; }}
.value-label.small {{ font-size: 10.5px; }}
.range-line, .range-cap {{ stroke: var(--text); stroke-width: 2; opacity: 0.62; }}
.zero-line {{ stroke: var(--border); stroke-width: 1; }}
.grid {{ stroke: var(--grid); stroke-width: 1; }}
.row-hit {{ fill: transparent; }}
.mark:hover .row-hit {{ fill: var(--neutral); opacity: 0.6; }}
.legend {{ display: flex; gap: 18px; padding: 2px 0 10px 48px; font-size: 13px; color: var(--muted); }}
.legend-item {{ display: inline-flex; align-items: center; gap: 7px; }}
.swatch {{ width: 11px; height: 11px; border-radius: 3px; display: inline-block; }}
.table-wrap {{ overflow-x: auto; border: 1px solid var(--border); border-radius: 10px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13.5px; background: var(--panel); }}
th, td {{ text-align: left; padding: 7px 12px; border-bottom: 1px solid var(--border); white-space: nowrap; }}
th {{ font-size: 11.5px; text-transform: uppercase; letter-spacing: 0.05em; color: var(--faint); font-weight: 600; }}
td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
tbody tr:last-child td {{ border-bottom: none; }}
tbody tr:hover td {{ background: var(--neutral); }}
footer {{ margin-top: 46px; padding-top: 18px; border-top: 1px solid var(--border); color: var(--muted); font-size: 13px; }}
a {{ color: var(--series-1); }}
code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12.5px; }}
#tip {{
  position: fixed; pointer-events: none; opacity: 0; transition: opacity .1s;
  background: var(--text); color: var(--surface); font-size: 12.5px;
  padding: 5px 9px; border-radius: 6px; max-width: 320px; z-index: 10;
}}
.empty {{ color: var(--muted); font-size: 13.5px; padding: 10px 4px; }}

/* Player links: readable as text, obviously clickable on hover. */
.nav-link {{ color: inherit; text-decoration: none; cursor: pointer; border-bottom: 1px solid transparent; }}
.nav-link:hover {{ color: var(--series-1); border-bottom-color: var(--series-1); }}
text.nav-link:hover {{ fill: var(--series-1); }}
input[type="search"], input[type="text"] {{
  font: inherit; font-size: 14px; padding: 6px 10px; min-width: 220px;
  background: var(--surface); color: var(--text);
  border: 1px solid var(--border); border-radius: 7px;
}}
.back-link {{
  display: inline-flex; align-items: center; gap: 6px; margin: 4px 0 14px;
  font-size: 14px; color: var(--series-1); text-decoration: none; cursor: pointer;
}}
.back-link:hover {{ text-decoration: underline; }}
.player-header {{ display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap; margin-bottom: 2px; }}
.player-header h2 {{ margin: 0; font-size: 26px; }}
.player-team {{ color: var(--muted); font-size: 15px; }}
.pill {{
  display: inline-block; padding: 2px 9px; border-radius: 999px; font-size: 12px;
  background: var(--neutral); color: var(--muted);
}}
.hidden {{ display: none; }}
"""


def _script() -> str:
    # Plain string (no f-string) so JavaScript braces stay untouched.
    return r"""
const $ = (id) => document.getElementById(id);
const tip = $('tip');
const fmt = (v, d = 1) => (v === null || v === undefined || isNaN(v)) ? '--' : v.toFixed(d);
const esc = (s) => String(s).replace(/[&<>"]/g, c => (
  { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

function barPath(x, y, w, h, r) {
  r = Math.max(0, Math.min(r === undefined ? 4 : r, w));
  if (w <= 0) return '';
  return `M${x},${y} H${x + w - r} Q${x + w},${y} ${x + w},${y + r} V${y + h - r}
          Q${x + w},${y + h} ${x + w - r},${y + h} H${x} Z`;
}

function hbar(rows, opts) {
  const o = Object.assign({ labelWidth: 190, rowHeight: 30, colour: 'var(--series-1)',
                            decimals: 1, suffix: '', width: 720 }, opts || {});
  if (!rows.length) return "<p class='empty'>Nothing to show for this selection.</p>";
  const h = o.rowHeight * rows.length + 12;
  const plotLeft = o.labelWidth, plotWidth = o.width - plotLeft - 66;
  const scale = Math.max(...rows.map(r => Math.max(r.value, r.high || 0))) || 1;
  let s = `<svg viewBox="0 0 ${o.width} ${h}" class="chart" role="img"
             preserveAspectRatio="xMinYMin meet">`;
  rows.forEach((r, i) => {
    const y = i * o.rowHeight + 6, bh = o.rowHeight - 12;
    const bw = Math.max(0, r.value / scale * plotWidth);
    s += `<g class="mark" data-tip="${esc(r.tip)}">`;
    s += `<rect x="0" y="${y - 3}" width="${o.width}" height="${o.rowHeight - 2}" class="row-hit"/>`;
    const labelText = `<text x="${plotLeft - 10}" y="${y + bh / 2 + (r.sub ? 0 : 4)}"
             text-anchor="end" class="axis-label${r.href ? ' nav-link' : ''}"
             >${esc(r.label)}</text>`;
    s += r.href ? `<a href="${r.href}">${labelText}</a>` : labelText;
    if (r.sub) s += `<text x="${plotLeft - 10}" y="${y + bh / 2 + 12}" text-anchor="end"
             class="axis-sublabel">${esc(r.sub)}</text>`;
    s += `<path d="${barPath(plotLeft, y, bw, bh)}" fill="${o.colour}"/>`;
    if (r.low !== undefined && r.high !== undefined && r.low !== null && r.high !== null) {
      const x1 = plotLeft + r.low / scale * plotWidth, x2 = plotLeft + r.high / scale * plotWidth;
      const mid = y + bh / 2;
      s += `<line x1="${x1}" y1="${mid}" x2="${x2}" y2="${mid}" class="range-line"/>`;
      s += `<line x1="${x1}" y1="${mid - 4}" x2="${x1}" y2="${mid + 4}" class="range-cap"/>`;
      s += `<line x1="${x2}" y1="${mid - 4}" x2="${x2}" y2="${mid + 4}" class="range-cap"/>`;
    }
    const labelX = plotLeft + Math.max(bw, (r.high || 0) / scale * plotWidth) + 9;
    s += `<text x="${labelX}" y="${y + bh / 2 + 4}" class="value-label">
            ${fmt(r.value, o.decimals)}${o.suffix}</text></g>`;
  });
  return s + '</svg>';
}

function diverging(rows) {
  if (!rows.length) return "<p class='empty'>No coefficients recorded.</p>";
  const width = 720, rowHeight = 26, labelWidth = 236;
  const h = rowHeight * rows.length + 24;
  const plotWidth = width - labelWidth - 70, zero = labelWidth + plotWidth / 2, half = plotWidth / 2;
  const scale = Math.max(...rows.map(r => Math.abs(r.coefficient))) || 1;
  let s = `<svg viewBox="0 0 ${width} ${h}" class="chart" role="img"
             preserveAspectRatio="xMinYMin meet">`;
  s += `<line x1="${zero}" y1="4" x2="${zero}" y2="${h - 18}" class="zero-line"/>`;
  rows.forEach((r, i) => {
    const y = i * rowHeight + 6, bh = rowHeight - 11;
    const len = Math.abs(r.coefficient) / scale * half, pos = r.coefficient >= 0;
    const colour = pos ? 'var(--positive)' : 'var(--negative)';
    const x = pos ? zero : zero - len;
    const dir = pos ? 'increases' : 'decreases';
    s += `<g class="mark" data-tip="${esc(r.feature)}: ${r.coefficient.toFixed(3)} — ${dir} a player's vote chance">`;
    s += `<rect x="0" y="${y - 3}" width="${width}" height="${rowHeight - 2}" class="row-hit"/>`;
    s += `<text x="${labelWidth - 12}" y="${y + bh / 2 + 4}" text-anchor="end"
            class="axis-label mono">${esc(r.feature)}</text>`;
    s += pos
      ? `<path d="${barPath(zero, y, len, bh)}" fill="${colour}"/>`
      : `<path d="M${zero},${y} H${x + 4} Q${x},${y} ${x},${y + 4} V${y + bh - 4}
           Q${x},${y + bh} ${x + 4},${y + bh} H${zero} Z" fill="${colour}"/>`;
    const tx = pos ? zero + len + 8 : zero - len - 8;
    s += `<text x="${tx}" y="${y + bh / 2 + 4}" text-anchor="${pos ? 'start' : 'end'}"
            class="value-label">${r.coefficient > 0 ? '+' : ''}${r.coefficient.toFixed(2)}</text></g>`;
  });
  s += `<text x="${zero}" y="${h - 4}" text-anchor="middle" class="axis-sublabel">0 (no effect)</text>`;
  return s + '</svg>';
}

function groupedBars(categories, series) {
  const names = Object.keys(series);
  if (!names.length) return '';
  const width = 720, height = 250, left = 46, bottom = 52, top = 18;
  const plotW = width - left - 16, plotH = height - bottom - top;
  const colours = ['var(--series-1)', 'var(--series-2)'];
  let scale = Math.max(...names.flatMap(n => series[n])) || 1;
  scale *= 1.15;
  const groupW = plotW / categories.length;
  const barW = Math.min(48, (groupW - 20) / names.length);
  let s = `<svg viewBox="0 0 ${width} ${height}" class="chart" role="img"
             preserveAspectRatio="xMinYMin meet">`;
  [0, 0.25, 0.5, 0.75, 1].forEach(f => {
    const y = top + plotH * (1 - f);
    s += `<line x1="${left}" y1="${y}" x2="${width - 16}" y2="${y}" class="grid"/>`;
    s += `<text x="${left - 8}" y="${y + 4}" text-anchor="end" class="axis-sublabel">
            ${(f * scale).toFixed(2)}</text>`;
  });
  categories.forEach((cat, c) => {
    const centre = left + groupW * (c + 0.5);
    const offset = centre - (barW * names.length) / 2;
    names.forEach((name, i) => {
      const v = series[name][c], bh = v / scale * plotH;
      const x = offset + i * barW, y = top + plotH - bh;
      s += `<g class="mark" data-tip="${esc(name)} — ${esc(cat)}: ${v.toFixed(3)}">`;
      s += `<path d="${barPath(x + 1, y, barW - 2, bh)}" fill="${colours[i % 2]}"/>`;
      s += `<text x="${x + barW / 2}" y="${y - 6}" text-anchor="middle"
              class="value-label small">${v.toFixed(2)}</text></g>`;
    });
    s += `<text x="${centre}" y="${height - 26}" text-anchor="middle"
            class="axis-label">${esc(cat)}</text>`;
  });
  s += '</svg>';
  const legend = names.map((n, i) =>
    `<span class="legend-item"><span class="swatch" style="background:${colours[i % 2]}"></span>
     ${esc(n)}</span>`).join('');
  return `<div class="legend">${legend}</div>${s}`;
}

function tile(label, value, note) {
  return `<div class="tile"><div class="tile-label">${esc(label)}</div>
    <div class="tile-value">${esc(value)}</div>
    <div class="tile-note">${esc(note || '')}</div></div>`;
}

function attachTips(root) {
  root.querySelectorAll('.mark').forEach(mark => {
    mark.addEventListener('mousemove', e => {
      tip.textContent = mark.dataset.tip;
      tip.style.opacity = '1';
      tip.style.left = Math.min(e.clientX + 14, window.innerWidth - tip.offsetWidth - 12) + 'px';
      tip.style.top = (e.clientY + 16) + 'px';
    });
    mark.addEventListener('mouseleave', () => { tip.style.opacity = '0'; });
  });
}

function currentRun() { return RUNS.find(r => r.name === $('run-select').value) || RUNS[0]; }

// Two players can share a name, so identity is always name + club.
function playerKey(p) { return p.team ? `${p.player}|${p.team}` : p.player; }

// A short label beside a name, e.g. "Omitted". Presentational only -- these
// never touch the model, the projection or the simulation.
function annotationFor(run, name) { return (run.annotations || {})[name] || ''; }

function annotationTag(run, name) {
  const label = annotationFor(run, name);
  return label ? ` <span class="pill">${esc(label)}</span>` : '';
}

function playerHref(p) { return '#player=' + encodeURIComponent(playerKey(p)); }
function teamHref(team) { return '#team=' + encodeURIComponent(team); }
function roundHref(round) { return '#round=' + encodeURIComponent(round); }

function teamLink(team) {
  return team ? `<a class="nav-link" href="${teamHref(team)}">${esc(team)}</a>` : '--';
}
function roundLink(round) {
  return `<a class="nav-link" href="${roundHref(round)}">${esc(round)}</a>`;
}

function playerLink(run, p, text) {
  return `<a class="nav-link" href="${playerHref(p)}">${esc(text || p.player)}</a>`
    + annotationTag(run, p.player);
}

function populateTeams(run) {
  const select = $('team-select');
  const previous = select.value;
  const teams = [...new Set(run.players.map(p => p.team).filter(Boolean))].sort();
  select.innerHTML = '<option value="">All teams</option>' +
    teams.map(t => `<option value="${esc(t)}">${esc(t)}</option>`).join('');
  if (teams.includes(previous)) select.value = previous;
}

function populateRounds(run) {
  const select = $('round-select');
  const previous = select.value;
  const rounds = run.rounds || [];
  select.innerHTML = '<option value="">Whole season</option>' +
    rounds.map(r => `<option value="${esc(r.round)}">Round ${esc(r.round)}</option>`).join('');
  if (rounds.some(r => r.round === previous)) select.value = previous;
}

// The search list follows the team filter, so choosing a club narrows the
// names you can type -- which is the whole point of having both.
function populatePlayerSearch(run) {
  const team = $('team-select').value;
  const pool = team ? run.players.filter(p => p.team === team) : run.players;
  $('player-options').innerHTML = [...pool]
    .sort((a, b) => a.player.localeCompare(b.player))
    .map(p => {
      const label = team ? p.player : `${p.player} (${p.team || '--'})`;
      return `<option value="${esc(label)}"></option>`;
    }).join('');
}

// Accepts what the datalist puts in the box, and also a plain typed name.
function findPlayerByLabel(run, text) {
  const wanted = (text || '').trim().toLowerCase();
  if (!wanted) return null;
  const withTeam = run.players.find(
    p => `${p.player} (${p.team || '--'})`.toLowerCase() === wanted);
  if (withTeam) return withTeam;
  const exact = run.players.filter(p => p.player.toLowerCase() === wanted);
  if (exact.length) return exact[0];
  const partial = run.players.filter(p => p.player.toLowerCase().includes(wanted));
  return partial.length === 1 ? partial[0] : null;
}

function render() {
  const run = currentRun();
  const team = $('team-select').value;
  const players = team ? run.players.filter(p => p.team === team) : run.players;
  const ranked = [...players].sort((a, b) => b.expected_votes - a.expected_votes);
  const hasWin = run.players.some(p => p.win_probability !== undefined && p.win_probability !== null);

  const seasons = run.train_seasons || [];
  const trained = seasons.length > 2
    ? `${seasons[0]}\u2013${seasons[seasons.length - 1]}` : seasons.join(', ');
  const cv = run.cv || {};
  const best = run.is_best
    ? `<span class="pill">best of ${RUNS.length}, by ${esc(run.ranked_on || 'score')}</span>` : '';
  const folds = cv.n_folds
    ? ` · cross-validated over ${cv.n_folds} seasons: ` +
      `${(cv.top3_recall * 100).toFixed(1)}% top-3 recall` : '';
  $('run-meta').innerHTML = best +
    `Trained on ${esc(trained)}` +
    (run.model ? ` · ${esc(run.model)} (alpha ${esc(run.alpha)})` : '') + folds +
    (run.notes ? `<br>${esc(run.notes)}` : '');

  // Headline tiles
  const leader = ranked[0];
  let tiles = '';
  if (leader) {
    tiles += tile(team ? `${team} leader` : 'Projected winner', leader.player,
                  team ? '' : (leader.team || ''));
    if (hasWin && !team) {
      tiles += tile('Win probability', (leader.win_probability * 100).toFixed(0) + '%',
                    `from ${(run.simulations || 0).toLocaleString()} simulated seasons`);
    }
    tiles += tile('Projected votes', fmt(leader.expected_votes),
                  'expected total across the season');
  }
  if (run.holdout && run.holdout.top3_recall !== undefined) {
    tiles += tile('Holdout top-3 recall', (run.holdout.top3_recall * 100).toFixed(1) + '%',
                  `tested on ${(run.test_seasons || []).join(', ') || 'held-out seasons'}`);
  }
  $('tiles').innerHTML = tiles;

  // Leaderboard
  $('leaderboard-title').textContent = team
    ? `${team}: projected votes, ${run.season_label}`
    : `Projected leaderboard, ${run.season_label}`;
  $('leaderboard').innerHTML = hbar(ranked.slice(0, 20).map(p => ({
    label: p.player, sub: team ? null : p.team, value: p.expected_votes,
    href: playerHref(p),
    low: p.p10_votes, high: p.p90_votes,
    tip: `${p.player} (${p.team || '--'}): ${fmt(p.expected_votes)} expected votes` +
         (p.p10_votes !== null && p.p10_votes !== undefined
            ? ` · likely ${fmt(p.p10_votes, 0)}–${fmt(p.p90_votes, 0)}` : '') +
         (p.win_probability ? ` · ${(p.win_probability * 100).toFixed(1)}% to win` : '')
  })), { labelWidth: team ? 170 : 190 });

  // Win probability
  const winSection = $('win-section');
  if (hasWin) {
    const contenders = [...players]
      .filter(p => p.win_probability > 0.0005)
      .sort((a, b) => b.win_probability - a.win_probability).slice(0, 12);
    if (contenders.length) {
      winSection.style.display = '';
      $('win-chart').innerHTML = hbar(contenders.map(p => ({
        label: p.player, sub: team ? null : p.team, value: p.win_probability * 100,
        href: playerHref(p),
        tip: `${p.player}: ${(p.win_probability * 100).toFixed(1)}% chance of winning the medal`
      })), { suffix: '%', decimals: 1 });
    } else {
      winSection.style.display = 'none';
    }
  } else {
    winSection.style.display = 'none';
  }

  // Team totals -- always the whole competition, so the bars stay comparable.
  const totals = {};
  run.players.forEach(p => {
    if (!p.team) return;
    totals[p.team] = (totals[p.team] || 0) + p.expected_votes;
  });
  const teamRows = Object.entries(totals).sort((a, b) => b[1] - a[1]).map(([name, value]) => ({
    label: name, value,
    tip: `${name}: ${fmt(value)} projected votes across the squad`
  }));
  $('team-chart').innerHTML = hbar(teamRows, { labelWidth: 200, rowHeight: 26 });
  $('team-table').innerHTML = teamTable(run, totals);

  // Model quality
  const comparison = run.comparison || {};
  const names = Object.keys(comparison);
  if (names.length >= 2) {
    $('quality').innerHTML = groupedBars(
      ['Top-1 accuracy', 'Top-3 recall', 'Leaderboard correlation'],
      Object.fromEntries(names.map(n => [n, [
        comparison[n].top1_accuracy || 0,
        comparison[n].top3_recall || 0,
        comparison[n].spearman || 0]])));
  } else if (run.holdout && run.holdout.top3_recall !== undefined) {
    const h = run.holdout;
    $('quality').innerHTML = '<div class="tiles">' +
      tile('Top-1 accuracy', (h.top1_accuracy * 100).toFixed(1) + '%', 'right 3-vote getter') +
      tile('Top-3 recall', (h.top3_recall * 100).toFixed(1) + '%', 'vote-getters in our top 3') +
      tile('Spearman', fmt(h.spearman, 3), 'season leaderboard correlation') +
      tile('Winner called', h.winner_correct ? 'Yes' : 'No',
           'predicted ' + (h.predicted_winner || '--')) + '</div>';
  } else {
    $('quality').innerHTML = "<p class='empty'>No holdout season was scored for this run.</p>";
  }

  $('coefficients').innerHTML = diverging(run.coefficients || []);
  $('player-table').innerHTML = playerTable(run, ranked, hasWin);
  attachTips(document);
}

function teamTable(run, totals) {
  const actual = {};
  let hasActual = false;
  run.players.forEach(p => {
    if (!p.team) return;
    if (p.actual_votes !== null && p.actual_votes !== undefined) {
      actual[p.team] = (actual[p.team] || 0) + p.actual_votes;
      hasActual = true;
    }
  });
  const counts = {};
  run.players.forEach(p => { if (p.team) counts[p.team] = (counts[p.team] || 0) + 1; });
  const rows = Object.entries(totals).sort((a, b) => b[1] - a[1]).map(([team, value], i) => {
    const best = run.players.filter(p => p.team === team)
      .sort((a, b) => b.expected_votes - a.expected_votes)[0];
    return `<tr><td class="num">${i + 1}</td><td>${teamLink(team)}</td>
      <td class="num">${fmt(value)}</td>
      ${hasActual ? `<td class="num">${fmt(actual[team] || 0)}</td>` : ''}
      <td>${best ? playerLink(run, best) : '--'}</td>
      <td class="num">${fmt(best ? best.expected_votes : 0)}</td>
      <td class="num">${counts[team] || 0}</td></tr>`;
  }).join('');
  return `<div class="table-wrap"><table><thead><tr><th>#</th><th>Team</th>
    <th>Projected votes</th>${hasActual ? '<th>Actual</th>' : ''}
    <th>Best player</th><th>Their votes</th><th>Players used</th>
    </tr></thead><tbody>${rows}</tbody></table></div>`;
}

function playerTable(run, ranked, hasWin) {
  const hasActual = ranked.some(p => p.actual_votes !== null && p.actual_votes !== undefined);
  // Matches the per-match detail cap, so every player listed here has a full
  // round-by-round page behind their name.
  const rows = ranked.slice(0, run.detail_players || 50).map((p, i) => `<tr>
    <td class="num">${i + 1}</td><td>${playerLink(run, p)}</td><td>${esc(p.team || '--')}</td>
    <td class="num"><strong>${p.predicted_votes === null || p.predicted_votes === undefined
        ? '--' : fmt(p.predicted_votes, 0)}</strong></td>
    <td class="num">${fmt(p.expected_votes)}</td>
    <td class="num">${p.games === undefined || p.games === null ? '--' : p.games}</td>
    ${hasWin ? `<td class="num">${p.win_probability === null || p.win_probability === undefined
        ? '--' : (p.win_probability * 100).toFixed(1) + '%'}</td>` : ''}
    ${hasWin ? `<td class="num">${fmt(p.p10_votes, 0)}–${fmt(p.p90_votes, 0)}</td>` : ''}
    ${hasActual ? `<td class="num">${fmt(p.actual_votes, 0)}</td>` : ''}
  </tr>`).join('');
  return `<div class="table-wrap"><table><thead><tr>
    <th>#</th><th>Player</th><th>Team</th>
    <th title="Sum of the 3-2-1 the model awards, game by game">Predicted</th>
    <th title="Sum of 3&times;P(3) + 2&times;P(2) + P(1)">Expected</th>
    <th>Games</th>
    ${hasWin ? '<th>Win prob.</th><th>Likely range</th>' : ''}
    ${hasActual ? '<th>Actual</th>' : ''}
    </tr></thead><tbody>${rows}</tbody></table></div>`;
}

/* ---------------- Individual player pages ---------------- */

function startTimeLabel(start) {
  if (start === null || start === undefined) return '';
  const hours = Math.floor(start / 100), minutes = start % 100;
  const suffix = hours >= 12 ? 'pm' : 'am';
  const hour12 = ((hours + 11) % 12) + 1;
  return `${hour12}:${String(minutes).padStart(2, '0')}${suffix}`;
}

// Expected votes round by round, with the actual result beside it when the
// season has been counted. Both are votes on the same scale, so they share one
// axis rather than being forced onto two.
function roundBars(games, opponents, showActual) {
  if (!games.length) return "<p class='empty'>No matches recorded for this player.</p>";
  const width = 720, height = 240, left = 40, bottom = 46, top = 16;
  const plotW = width - left - 14, plotH = height - bottom - top;
  const maxValue = Math.max(3, ...games.map(g => Math.max(g[3] || 0, showActual ? (g[7] || 0) : 0)));
  const scale = maxValue * 1.1;
  const slot = plotW / games.length;
  const series = showActual ? 2 : 1;
  const barW = Math.max(3, Math.min(22, (slot - 6) / series));

  let s = `<svg viewBox="0 0 ${width} ${height}" class="chart" role="img"
             preserveAspectRatio="xMinYMin meet">`;
  [0, 1, 2, 3].filter(v => v <= scale).forEach(v => {
    const y = top + plotH * (1 - v / scale);
    s += `<line x1="${left}" y1="${y}" x2="${width - 14}" y2="${y}" class="grid"/>`;
    s += `<text x="${left - 7}" y="${y + 4}" text-anchor="end" class="axis-sublabel">${v}</text>`;
  });
  games.forEach((g, i) => {
    const centre = left + slot * (i + 0.5);
    const offset = centre - (barW * series) / 2;
    const opponent = opponents[g[1]] || 'unknown';
    const venue = g[2] ? 'vs' : 'at';
    const values = showActual ? [g[3] || 0, g[7] || 0] : [g[3] || 0];
    const colours = ['var(--series-1)', 'var(--series-2)'];
    values.forEach((v, k) => {
      const bh = (v / scale) * plotH;
      const x = offset + k * barW, y = top + plotH - bh;
      const what = k === 0 ? 'projected' : 'actual';
      s += `<g class="mark" data-tip="Round ${g[0]} ${venue} ${opponent}: ${v.toFixed(2)} ${what} votes">`;
      s += `<path d="${barPath(x + 1, y, barW - 2, bh)}" fill="${colours[k]}"/></g>`;
    });
    if (games.length <= 26) {
      s += `<text x="${centre}" y="${height - 28}" text-anchor="middle"
              class="axis-sublabel">${esc(g[0])}</text>`;
    }
  });
  s += `<text x="${left + plotW / 2}" y="${height - 8}" text-anchor="middle"
          class="axis-sublabel">Round</text></svg>`;
  const legend = showActual
    ? `<div class="legend"><span class="legend-item"><span class="swatch"
         style="background:var(--series-1)"></span>Projected</span>
       <span class="legend-item"><span class="swatch"
         style="background:var(--series-2)"></span>Actual</span></div>`
    : '';
  return legend + s;
}

function renderPlayer(key) {
  const run = currentRun();
  const position = run.players.findIndex(p => playerKey(p) === key);
  const view = $('player-view');
  if (position < 0) {
    view.innerHTML = `<a class="back-link" href="#">&larr; Back to the season</a>
      <p class="empty">No player matching that link in this run.</p>`;
    return null;
  }
  const p = run.players[position];
  const games = (run.games || [])[position] || [];
  const showActual = games.some(g => g[7] !== null && g[7] !== undefined);
  const ranked = [...run.players].sort((a, b) => b.expected_votes - a.expected_votes);
  const rank = ranked.findIndex(x => playerKey(x) === key) + 1;
  const best = games.length
    ? games.reduce((a, b) => ((b[3] || 0) > (a[3] || 0) ? b : a)) : null;
  const opponents = run.opponents || [];

  const allocated = games.reduce((total, g) => total + (g[8] || 0), 0);
  let tiles = tile('Expected votes', fmt(p.expected_votes),
                   `${p.games || games.length} games`);
  if (p.predicted_votes !== null && p.predicted_votes !== undefined) {
    tiles += tile('Predicted votes', fmt(p.predicted_votes, 0),
                  'games where the model names them');
  }
  if (p.win_probability !== null && p.win_probability !== undefined) {
    tiles += tile('Chance of winning', (p.win_probability * 100).toFixed(1) + '%',
                  'across simulated seasons');
  }
  if (p.p10_votes !== null && p.p10_votes !== undefined) {
    tiles += tile('Likely range', `${fmt(p.p10_votes, 0)}–${fmt(p.p90_votes, 0)}`,
                  '10th to 90th percentile');
  }
  tiles += tile('Season rank', '#' + rank, 'by projected votes');
  if (best) {
    tiles += tile('Best game', `Round ${best[0]}`,
                  `${best[2] ? 'vs' : 'at'} ${opponents[best[1]] || '--'} · ` +
                  `${fmt(best[3], 2)} votes`);
  }
  if (p.actual_votes !== null && p.actual_votes !== undefined) {
    tiles += tile('Actual votes', fmt(p.actual_votes, 0), 'what they really polled');
  }

  const rows = games.map(g => `<tr>
    <td class="num">${roundLink(g[0])}</td>
    <td>${g[2] ? 'vs' : 'at'} ${teamLink(opponents[g[1]])}</td>
    <td>${g[2] ? 'Home' : 'Away'}</td>
    <td class="num">${((g[4] || 0) * 100).toFixed(1)}%</td>
    <td class="num">${((g[5] || 0) * 100).toFixed(1)}%</td>
    <td class="num">${((g[6] || 0) * 100).toFixed(1)}%</td>
    <td class="num"><strong>${g[8] === null || g[8] === undefined ? '--' : g[8]}</strong></td>
    <td class="num">${fmt(g[3], 2)}</td>
    ${showActual ? `<td class="num">${g[7] === null || g[7] === undefined ? '--' : g[7]}</td>` : ''}
  </tr>`).join('');

  view.innerHTML = `
    <a class="back-link" href="#">&larr; Back to the season</a>
    <div class="player-header">
      <h2>${esc(p.player)}</h2>
      <span class="player-team">${teamLink(p.team)}</span>
      ${annotationFor(run, p.player) ? `<span class="pill">${esc(annotationFor(run, p.player))}</span>` : ''}
    </div>
    <p class="note">Projected votes for ${esc(run.season_label)}, match by match.</p>
    <div class="tiles">${tiles}</div>
    ${games.length ? `
    <h2>Round by round</h2>
    <p class="note">How many votes this player is projected to poll in each match.
       Three is the most any single game can award.</p>
    <div class="panel">${roundBars(games, opponents, showActual)}</div>
    <h2>Every match</h2>
    <p class="note"><strong>Predicted</strong> is the model's call for that game
       &mdash; 3, 2, 1 or 0, given to the three players it rates highest.
       <strong>Expected</strong> is ${'3'} &times; P(3) + 2 &times; P(2) + P(1), which
       is not a whole number but is the better estimate of a season total,
       because it keeps the games a player nearly polled in. Both add to exactly
       6 across every match.</p>
    <div class="table-wrap"><table><thead><tr>
      <th>Round</th><th>Opponent</th><th>H/A</th>
      <th>P(3)</th><th>P(2)</th><th>P(1)</th><th>Predicted</th><th>Expected</th>
      ${showActual ? '<th>Actual</th>' : ''}
    </tr></thead><tbody>${rows}</tbody></table></div>` : `
    <p class="note">Match-by-match detail is kept for the top
       ${run.detail_players || 100} players, to hold the page size down. This
       player's season totals are above; the
       <a class="nav-link" href="${roundHref((run.rounds && run.rounds.length)
         ? run.rounds[0].round : '1')}">round pages</a> show every match.</p>`}
    <p><a class="back-link" href="#">&larr; Back to the season</a></p>`;
  attachTips(view);
  return p;
}

/* ---------------- Round pages ---------------- */

function renderRound(roundLabel) {
  const run = currentRun();
  const view = $('round-view');
  const round = (run.rounds || []).find(r => r.round === roundLabel);
  if (!round) {
    view.innerHTML = `<a class="back-link" href="#">&larr; Back to the season</a>
      <p class="empty">No matches recorded for that round in this run.</p>`;
    return;
  }
  const byKey = {};
  run.players.forEach(p => { byKey[playerKey(p)] = p; });
  const voteLabels = ['3 votes', '2 votes', '1 vote'];

  const cards = round.matches.map(match => {
    const when = [match.date, startTimeLabel(match.start)].filter(Boolean).join(' · ');
    const rows = match.top.map((entry, i) => {
      const linked = byKey[`${entry.player}|${entry.team}`];
      const name = linked ? playerLink(run, linked) : esc(entry.player) + annotationTag(run, entry.player);
      const predicted = i < 3
        ? `<span class="pill">${voteLabels[i]}</span>`
        : '<span class="axis-sublabel">next best</span>';
      return `<tr>
        <td>${predicted}</td>
        <td>${name}</td>
        <td>${teamLink(entry.team)}</td>
        <td class="num"><strong>${entry.allocated || 0}</strong></td>
        <td class="num">${fmt(entry.expected, 2)}</td>
        <td class="num">${(entry.p3 * 100).toFixed(1)}%</td>
        ${entry.actual === null || entry.actual === undefined
          ? '' : `<td class="num">${entry.actual}</td>`}
      </tr>`;
    }).join('');
    const hasActual = match.top.some(e => e.actual !== null && e.actual !== undefined);
    return `
      <h2>${teamLink(match.home)} v ${teamLink(match.away)}</h2>
      <p class="note">${esc(when)}${match.venue ? ' · ' + esc(match.venue) : ''}</p>
      <div class="table-wrap"><table><thead><tr>
        <th>Projected</th><th>Player</th><th>Team</th>
        <th>Predicted</th><th>Expected</th><th>Chance of the 3</th>
        ${hasActual ? '<th>Actual</th>' : ''}
      </tr></thead><tbody>${rows}</tbody></table></div>`;
  }).join('');

  view.innerHTML = `
    <a class="back-link" href="#">&larr; Back to the season</a>
    <div class="player-header"><h2>Round ${esc(round.round)}</h2></div>
    <p class="note">Every match in the round, in the order it was played, with the
       players most likely to poll. The top three are the projected 3-2-1.</p>
    ${cards}
    <p><a class="back-link" href="#">&larr; Back to the season</a></p>`;
  attachTips(view);
}

/* ---------------- Team pages ---------------- */

function renderTeam(teamName) {
  const run = currentRun();
  const view = $('team-view');
  const squad = run.players
    .map((p, i) => ({ player: p, games: (run.games || [])[i] || [] }))
    .filter(entry => entry.player.team === teamName)
    .sort((a, b) => b.player.expected_votes - a.player.expected_votes);

  if (!squad.length) {
    view.innerHTML = `<a class="back-link" href="#">&larr; Back to the season</a>
      <p class="empty">No players for that team in this run.</p>`;
    return;
  }

  const totals = {};
  run.players.forEach(p => {
    if (p.team) totals[p.team] = (totals[p.team] || 0) + p.expected_votes;
  });
  const order = Object.entries(totals).sort((a, b) => b[1] - a[1]);
  const teamRank = order.findIndex(([name]) => name === teamName) + 1;
  const opponents = run.opponents || [];

  let tiles = tile('Projected votes', fmt(totals[teamName]), 'across the whole squad');
  tiles += tile('Competition rank', '#' + teamRank, `of ${order.length} clubs`);
  tiles += tile('Leading player', squad[0].player.player, fmt(squad[0].player.expected_votes) + ' votes');
  tiles += tile('Players used', String(squad.length), 'polled at least a share');

  // The club's fixtures, taken from the round data so they stay in playing order.
  const fixtures = [];
  (run.rounds || []).forEach(round => {
    round.matches.forEach(match => {
      if (match.home !== teamName && match.away !== teamName) return;
      const isHome = match.home === teamName;
      // Who from this club is most likely to poll in this match?
      let best = null;
      squad.forEach(entry => {
        entry.games.forEach(g => {
          if (String(g[0]) !== String(round.round)) return;
          if (!best || (g[3] || 0) > (best.expected || 0)) {
            best = { name: entry.player, expected: g[3] || 0 };
          }
        });
      });
      fixtures.push({ round: round.round, opponent: isHome ? match.away : match.home,
                      isHome, date: match.date, start: match.start, best });
    });
  });

  const fixtureRows = fixtures.map(f => `<tr>
    <td class="num">${roundLink(f.round)}</td>
    <td>${f.isHome ? 'vs' : 'at'} ${teamLink(f.opponent)}</td>
    <td>${esc([f.date, startTimeLabel(f.start)].filter(Boolean).join(' · '))}</td>
    <td>${f.best ? playerLink(run, f.best.name) : '--'}</td>
    <td class="num">${f.best ? fmt(f.best.expected, 2) : '--'}</td>
  </tr>`).join('');

  const hasWin = squad.some(e => e.player.win_probability !== null
                                 && e.player.win_probability !== undefined);
  const squadRows = squad.map((entry, i) => {
    const p = entry.player;
    return `<tr>
      <td class="num">${i + 1}</td>
      <td>${playerLink(run, p)}</td>
      <td class="num"><strong>${p.predicted_votes === null || p.predicted_votes === undefined
          ? '--' : fmt(p.predicted_votes, 0)}</strong></td>
      <td class="num">${fmt(p.expected_votes)}</td>
      <td class="num">${p.games === undefined || p.games === null ? '--' : p.games}</td>
      ${hasWin ? `<td class="num">${p.win_probability === null || p.win_probability === undefined
          ? '--' : (p.win_probability * 100).toFixed(1) + '%'}</td>` : ''}
      ${p.actual_votes === null || p.actual_votes === undefined
        ? '' : `<td class="num">${fmt(p.actual_votes, 0)}</td>`}
    </tr>`;
  }).join('');
  const hasActual = squad.some(e => e.player.actual_votes !== null
                                    && e.player.actual_votes !== undefined);

  view.innerHTML = `
    <a class="back-link" href="#">&larr; Back to the season</a>
    <div class="player-header"><h2>${esc(teamName)}</h2>
      <span class="player-team">${esc(run.season_label)}</span></div>
    <div class="tiles">${tiles}</div>

    <h2>Projected votes by player</h2>
    <p class="note">Every player at the club who is projected to poll, most votes first.</p>
    <div class="panel">${hbar(squad.slice(0, 20).map(e => ({
      label: e.player.player, value: e.player.expected_votes,
      href: playerHref(e.player), low: e.player.p10_votes, high: e.player.p90_votes,
      tip: `${e.player.player}: ${fmt(e.player.expected_votes)} projected votes`
    })), { labelWidth: 170 })}</div>

    <h2>Fixtures</h2>
    <p class="note">Every match this club played, in order, with the player from this
       club most likely to poll in it.</p>
    <div class="table-wrap"><table><thead><tr>
      <th>Round</th><th>Opponent</th><th>When</th>
      <th>Most likely to poll</th><th>Expected votes</th>
    </tr></thead><tbody>${fixtureRows}</tbody></table></div>

    <h2>Full squad</h2>
    <div class="table-wrap"><table><thead><tr>
      <th>#</th><th>Player</th><th>Predicted</th><th>Expected</th><th>Games</th>
      ${hasWin ? '<th>Win prob.</th>' : ''}${hasActual ? '<th>Actual</th>' : ''}
    </tr></thead><tbody>${squadRows}</tbody></table></div>
    <p><a class="back-link" href="#">&larr; Back to the season</a></p>`;
  attachTips(view);
}

/* ---------------- Routing between the views ---------------- */

function showOnly(id) {
  ['season-view', 'player-view', 'round-view', 'team-view'].forEach(name => {
    $(name).classList.toggle('hidden', name !== id);
  });
}

// Keep the controls telling the truth about where you are. Landing on a
// Collingwood player while the team box still reads "Sydney" is just confusing,
// and the round box does not apply to a player or team page at all.
function syncControls({ team, round }) {
  $('round-select').value = round || '';
  if (team !== undefined && $('team-select').value !== team) {
    const exists = [...$('team-select').options].some(o => o.value === team);
    if (exists) {
      $('team-select').value = team;
      populatePlayerSearch(currentRun());
    }
  }
}

function route() {
  const hash = window.location.hash || '';
  if (hash.startsWith('#player=')) {
    showOnly('player-view');
    const player = renderPlayer(decodeURIComponent(hash.slice('#player='.length)));
    syncControls({ team: player ? player.team : undefined, round: '' });
  } else if (hash.startsWith('#round=')) {
    const label = decodeURIComponent(hash.slice('#round='.length));
    showOnly('round-view');
    renderRound(label);
    syncControls({ round: label });
  } else if (hash.startsWith('#team=')) {
    const name = decodeURIComponent(hash.slice('#team='.length));
    showOnly('team-view');
    renderTeam(name);
    syncControls({ team: name, round: '' });
  } else {
    syncControls({ round: '' });
    showOnly('season-view');
    render();
  }
  window.scrollTo({ top: 0 });
}

$('run-select').addEventListener('change', () => {
  const run = currentRun();
  populateTeams(run);
  populateRounds(run);
  populatePlayerSearch(run);
  route();
});
$('team-select').addEventListener('change', () => {
  populatePlayerSearch(currentRun());
  if (!window.location.hash || window.location.hash === '#') render();
});
$('round-select').addEventListener('change', () => {
  const value = $('round-select').value;
  window.location.hash = value ? '#round=' + encodeURIComponent(value) : '';
  if (!value) route();
});
$('player-search').addEventListener('change', event => {
  const match = findPlayerByLabel(currentRun(), event.target.value);
  if (match) {
    window.location.hash = playerHref(match);
    event.target.value = '';
  }
});
window.addEventListener('hashchange', route);

populateTeams(currentRun());
populateRounds(currentRun());
populatePlayerSearch(currentRun());
route();
"""


def render_site(
    output_root: Optional[Path] = None,
    docs_path: Optional[Path] = None,
    title: str = "Brownlow Medal Projections",
    active_run: Optional[str] = None,
    detail_players: Optional[int] = DEFAULT_DETAIL_PLAYERS,
) -> Path:
    """Build ``docs/index.html`` from every experiment run on disk.

    Runs are ordered by how well they scored on their held-out season, so the
    page opens on the best one. ``active_run`` is accepted for callers that pass
    it but no longer changes the order -- the best model leads regardless of
    which one was run most recently.

    ``detail_players`` caps how many players keep round-by-round detail. Lower it
    if the page grows uncomfortably large as scenarios are added; pass ``None``
    to keep every player's matches.
    """
    runs = collect_runs(output_root, detail_players)
    docs_path = Path(docs_path) if docs_path else default_docs_path()
    docs_path.parent.mkdir(parents=True, exist_ok=True)

    if not runs:
        raise ValueError(
            "No experiment runs found. Run `brownlow predict` or `brownlow run` first."
        )

    # collect_runs already returns them best first (see rank_runs).

    generated = datetime.now(timezone.utc).strftime("%d %B %Y, %H:%M UTC")
    # The label carries the score, so the ordering explains itself.
    def option_label(run: Dict[str, Any]) -> str:
        cv = run.get("cv") or {}
        recall = cv.get("top3_recall")
        if recall is None:
            recall = (run.get("holdout") or {}).get("top3_recall")
        parts = [run["name"]]
        if recall is not None:
            parts.append(f"{recall * 100:.1f}%")
        return " -- ".join(parts)

    options = "".join(
        f'<option value="{r["name"]}">{option_label(r)}</option>' for r in runs
    )

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="icon" href="{_favicon()}">
<title>{title}</title>
<style>{_css()}</style>
</head>
<body>
<div id="tip" role="status" aria-live="polite"></div>
<div class="wrap">
<header>
  <a class="brand" href="#" id="home-link">{_logo()}<h1>{title}</h1></a>
  <p class="sub">Predicted 3-2-1 umpire votes from AFL player match statistics.
     Generated {generated}.</p>
</header>

<div class="controls">
  <div class="control">
    <label for="run-select" title="Ordered best first">Model</label>
    <select id="run-select">{options}</select>
  </div>
  <div class="control">
    <label for="team-select">Team</label>
    <select id="team-select"><option value="">All teams</option></select>
  </div>
  <div class="control">
    <label for="player-search">Find a player</label>
    <input id="player-search" type="search" list="player-options"
           placeholder="Start typing a name..." autocomplete="off">
    <datalist id="player-options"></datalist>
  </div>
  <div class="control">
    <label for="round-select">Round</label>
    <select id="round-select"><option value="">Whole season</option></select>
  </div>
</div>
<div class="run-meta" id="run-meta"></div>

<div id="season-view">
<div class="tiles" id="tiles"></div>

<h2 id="leaderboard-title">Projected leaderboard</h2>
<p class="note">Expected votes for each player across the season. The pale line through each bar
   spans the 10th to 90th percentile of simulated outcomes &mdash; how far the count could
   reasonably swing either way.</p>
<div class="panel" id="leaderboard"></div>

<div id="win-section">
  <h2>Chance of winning the medal</h2>
  <p class="note">Not the same ranking as expected votes. Winning needs a high total, so a player
     whose good games are very good can be likelier to win than a steadier player with a higher
     average.</p>
  <div class="panel" id="win-chart"></div>
</div>

<h2>Projected votes by team</h2>
<p class="note">Total projected votes across each club's whole squad &mdash; a rough read on which
   teams the umpires rewarded most. Always shown for the full competition so the bars stay
   comparable; use the team filter above to drill into one club's players.</p>
<div class="panel" id="team-chart"></div>
<div id="team-table" style="margin-top:14px"></div>

<h2>How accurate is this?</h2>
<p class="note">Measured on seasons the model never saw while training.
   <strong>Top-1</strong> is how often we picked the right 3-vote getter;
   <strong>top-3 recall</strong> is the share of all vote-getters we had in our top three;
   <strong>leaderboard correlation</strong> is how well the projected season table matches the
   real one. Higher is better throughout.</p>
<div class="panel" id="quality"></div>

<h2>What the model learned</h2>
<p class="note">The biggest drivers of a vote, on standardised inputs so they are directly
   comparable. A bar to the right means more of that thing makes votes more likely. Names ending
   in <code>_z</code> are measured against the other players in the same match rather than against
   the league.</p>
<div class="panel" id="coefficients"></div>

<h2>Full projected leaderboard</h2>
<div id="player-table"></div>
</div><!-- /season-view -->

<div id="player-view" class="hidden"></div>
<div id="round-view" class="hidden"></div>
<div id="team-view" class="hidden"></div>

<footer>
  <p>Built by the
     <a href="https://github.com/julia-latrobe/Brownlow-medal">Brownlow-medal</a> model.
     Player statistics and historical votes from <a href="https://afltables.com">AFL Tables</a>,
     accessed via the <a href="https://github.com/jimmyday12/fitzRoy">fitzRoy</a> project's
     public data mirror.</p>
  <p>This page is generated by <code>brownlow report</code> and committed to the repository, so
     the numbers above always correspond to a specific commit.</p>
</footer>
</div>
<script>const RUNS = {json.dumps(runs, default=str)};</script>
<script>{_script()}</script>
</body>
</html>
"""
    docs_path.write_text(document, encoding="utf-8")
    return docs_path


# Kept so older calls still work: build the site after a run.
def render_report(results, output_path=None, title="Brownlow Medal Projections", **kwargs):
    return render_site(docs_path=output_path, title=title,
                       active_run=(results or {}).get("config", {}).get("name"))
