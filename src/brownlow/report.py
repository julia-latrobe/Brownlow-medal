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


def collect_run(run_dir: Path) -> Optional[Dict[str, Any]]:
    """Read one experiment folder into the payload the page needs."""
    metrics_file = run_dir / "metrics.json"
    leaderboard_file = run_dir / "leaderboard.csv"
    if not metrics_file.exists() or not leaderboard_file.exists():
        return None

    metrics = json.loads(metrics_file.read_text())
    board = pd.read_csv(leaderboard_file)

    columns = [
        c for c in (
            "player", "team", "predicted_votes", "games", "actual_votes",
            "win_probability", "top5_probability", "mean_votes",
            "p10_votes", "p90_votes",
        ) if c in board.columns
    ]
    players = [
        {k: _clean(v) for k, v in row.items()}
        for row in board[columns].to_dict(orient="records")
    ]

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
        "comparison": metrics.get("comparison") or {},
        "coefficients": coefficients,
        "players": players,
    }


def collect_runs(output_root: Optional[Path] = None) -> List[Dict[str, Any]]:
    root = Path(output_root) if output_root else default_output_root()
    runs = []
    directories = [d for d in (root.iterdir() if root.exists() else []) if d.is_dir()]
    # Most recently run first, so the dropdown opens on what you just did
    # rather than on whichever folder happens to sort first.
    directories.sort(key=lambda d: (d / "metrics.json").stat().st_mtime
                     if (d / "metrics.json").exists() else 0, reverse=True)
    for run_dir in directories:
        payload = collect_run(run_dir)
        if payload:
            runs.append(payload)
    return runs


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
.brand {{ display: flex; align-items: center; gap: 13px; margin-bottom: 7px; }}
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
.run-meta {{ font-size: 12.5px; color: var(--muted); margin-left: auto; text-align: right; max-width: 42ch; line-height: 1.45; }}
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
    s += `<text x="${plotLeft - 10}" y="${y + bh / 2 + (r.sub ? 0 : 4)}" text-anchor="end"
             class="axis-label">${esc(r.label)}</text>`;
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

function populateTeams(run) {
  const select = $('team-select');
  const previous = select.value;
  const teams = [...new Set(run.players.map(p => p.team).filter(Boolean))].sort();
  select.innerHTML = '<option value="">All teams</option>' +
    teams.map(t => `<option value="${esc(t)}">${esc(t)}</option>`).join('');
  if (teams.includes(previous)) select.value = previous;
}

function render() {
  const run = currentRun();
  const team = $('team-select').value;
  const players = team ? run.players.filter(p => p.team === team) : run.players;
  const ranked = [...players].sort((a, b) => b.predicted_votes - a.predicted_votes);
  const hasWin = run.players.some(p => p.win_probability !== undefined && p.win_probability !== null);

  const seasons = run.train_seasons || [];
  const trained = seasons.length > 2
    ? `${seasons[0]}\u2013${seasons[seasons.length - 1]}` : seasons.join(', ');
  $('run-meta').innerHTML =
    `Trained on ${esc(trained)}` +
    (run.model ? ` · ${esc(run.model)} (alpha ${esc(run.alpha)})` : '') +
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
    tiles += tile('Projected votes', fmt(leader.predicted_votes),
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
    label: p.player, sub: team ? null : p.team, value: p.predicted_votes,
    low: p.p10_votes, high: p.p90_votes,
    tip: `${p.player} (${p.team || '--'}): ${fmt(p.predicted_votes)} expected votes` +
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
    totals[p.team] = (totals[p.team] || 0) + p.predicted_votes;
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
  $('player-table').innerHTML = playerTable(ranked, hasWin);
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
      .sort((a, b) => b.predicted_votes - a.predicted_votes)[0];
    return `<tr><td class="num">${i + 1}</td><td>${esc(team)}</td>
      <td class="num">${fmt(value)}</td>
      ${hasActual ? `<td class="num">${fmt(actual[team] || 0)}</td>` : ''}
      <td>${esc(best ? best.player : '--')}</td>
      <td class="num">${fmt(best ? best.predicted_votes : 0)}</td>
      <td class="num">${counts[team] || 0}</td></tr>`;
  }).join('');
  return `<div class="table-wrap"><table><thead><tr><th>#</th><th>Team</th>
    <th>Projected votes</th>${hasActual ? '<th>Actual</th>' : ''}
    <th>Best player</th><th>Their votes</th><th>Players used</th>
    </tr></thead><tbody>${rows}</tbody></table></div>`;
}

function playerTable(ranked, hasWin) {
  const hasActual = ranked.some(p => p.actual_votes !== null && p.actual_votes !== undefined);
  const rows = ranked.slice(0, 60).map((p, i) => `<tr>
    <td class="num">${i + 1}</td><td>${esc(p.player)}</td><td>${esc(p.team || '--')}</td>
    <td class="num">${fmt(p.predicted_votes)}</td>
    <td class="num">${p.games === undefined || p.games === null ? '--' : p.games}</td>
    ${hasWin ? `<td class="num">${p.win_probability === null || p.win_probability === undefined
        ? '--' : (p.win_probability * 100).toFixed(1) + '%'}</td>` : ''}
    ${hasWin ? `<td class="num">${fmt(p.p10_votes, 0)}–${fmt(p.p90_votes, 0)}</td>` : ''}
    ${hasActual ? `<td class="num">${fmt(p.actual_votes, 0)}</td>` : ''}
  </tr>`).join('');
  return `<div class="table-wrap"><table><thead><tr>
    <th>#</th><th>Player</th><th>Team</th><th>Projected votes</th><th>Games</th>
    ${hasWin ? '<th>Win prob.</th><th>Likely range</th>' : ''}
    ${hasActual ? '<th>Actual</th>' : ''}
    </tr></thead><tbody>${rows}</tbody></table></div>`;
}

$('run-select').addEventListener('change', () => { populateTeams(currentRun()); render(); });
$('team-select').addEventListener('change', render);
populateTeams(currentRun());
render();
"""


def render_site(
    output_root: Optional[Path] = None,
    docs_path: Optional[Path] = None,
    title: str = "Brownlow Medal Projections",
    active_run: Optional[str] = None,
) -> Path:
    """Build ``docs/index.html`` from every experiment run on disk."""
    runs = collect_runs(output_root)
    docs_path = Path(docs_path) if docs_path else default_docs_path()
    docs_path.parent.mkdir(parents=True, exist_ok=True)

    if not runs:
        raise ValueError(
            "No experiment runs found. Run `brownlow predict` or `brownlow run` first."
        )
    if active_run:
        runs.sort(key=lambda r: r["name"] != active_run)

    generated = datetime.now(timezone.utc).strftime("%d %B %Y, %H:%M UTC")
    options = "".join(
        f'<option value="{r["name"]}">{r["name"]}'
        f'{" -- " + r["season_label"] if r["season_label"] != "--" else ""}</option>'
        for r in runs
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
  <div class="brand">{_logo()}<h1>{title}</h1></div>
  <p class="sub">Predicted 3-2-1 umpire votes from AFL player match statistics.
     Generated {generated}.</p>
</header>

<div class="controls">
  <div class="control">
    <label for="run-select">Model run</label>
    <select id="run-select">{options}</select>
  </div>
  <div class="control">
    <label for="team-select">Team</label>
    <select id="team-select"><option value="">All teams</option></select>
  </div>
  <div class="run-meta" id="run-meta"></div>
</div>

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
