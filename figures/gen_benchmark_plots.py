#!/usr/bin/env python3
"""
Generate benchmark bar charts for MAC README figures.
Outputs hover_results.png, gsm8k_results.png, hotpotqa_results.png to /tmp/mac_plots/.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# ── colours ───────────────────────────────────────────────────────────
C_BASELINE = '#b7e4c7'
C_BEST     = '#1e7b34'
C_TITLE    = '#1a6b2f'
C_DELTA    = '#1a6b2f'
C_SEP      = '#cccccc'

BAR_W     = 0.35
GROUP_GAP = 1.0

# ── data — groups ordered by max delta descending (best gain on left) ─
BENCHMARKS = {
    'hover': {
        'title':      'HoVer — Fact Verification',
        'ylabel':     'Accuracy (%)',
        'ylim':       (0, 105),
        'legend_loc': 'upper right',
        'groups': [
            # max +63%
            ('gpt-4o-mini', 'gpt-5.2',  [('adapt', 25, 88), ('custom', 88, 88)]),
            # max +26%
            ('Qwen3-8B',   'gpt-5.2',  [('adapt', 69, 81), ('custom', 62, 88)]),
            # max +6%
            ('Qwen3-8B',   'Qwen3-8B', [('adapt', 75, 75), ('custom', 75, 81)]),
        ],
    },
    'gsm8k': {
        'title':      'GSM8K — Math Reasoning',
        'ylabel':     'Accuracy (%)',
        'ylim':       (0, 115),
        'legend_loc': 'lower right',
        'groups': [
            # max +6% (2 improvements)
            ('Qwen3-8B',   'gpt-5.2',  [('adapt',  94, 100), ('custom', 94, 100)]),
            # max +6% (1 improvement)
            ('Qwen3-8B',   'Qwen3-8B', [('adapt', 100, 100), ('custom', 94, 100)]),
            # max 0%
            ('gpt-4o-mini', 'gpt-5.2', [('adapt', 100, 100), ('custom', 94,  94)]),
        ],
    },
    'hotpotqa': {
        'title':      'HotpotQA — Multi-Hop QA',
        'ylabel':     'Accuracy (%)',
        'ylim':       (0, 55),
        'legend_loc': 'upper right',
        'groups': [
            # max +14%
            ('Qwen3-8B',   'Qwen3-8B', [('adapt', 27, 27), ('custom', 22, 36)]),
            # max +9%
            ('Qwen3-8B',   'gpt-5.2',  [('adapt', 29, 38), ('custom', 29, 36)]),
            # max +9%
            ('gpt-4o-mini', 'gpt-5.2', [('adapt', 25, 34), ('custom', 26, 32)]),
        ],
    },
}


def make_plot(name: str, cfg: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    # ── build positions ───────────────────────────────────────────────
    pairs      = []   # (x, baseline, best, style)
    group_info = []   # (group_x_center, worker, mac)
    sep_xs     = []

    x = 0.0
    for g_idx, (worker, mac, rows) in enumerate(cfg['groups']):
        grp_xs = []
        for style, baseline, best in rows:
            pairs.append((x, baseline, best, style))
            grp_xs.append(x)
            x += 1.0
        group_info.append((float(np.mean(grp_xs)), worker, mac))
        if g_idx < len(cfg['groups']) - 1:
            sep_xs.append(x - 0.5 + GROUP_GAP / 2)
            x += GROUP_GAP

    xs        = [p[0] for p in pairs]
    baselines = [p[1] for p in pairs]
    bests     = [p[2] for p in pairs]
    styles    = [p[3] for p in pairs]

    # ── bars ──────────────────────────────────────────────────────────
    ax.bar([x - BAR_W / 2 for x in xs], baselines, BAR_W,
           color=C_BASELINE, zorder=3)
    ax.bar([x + BAR_W / 2 for x in xs], bests, BAR_W,
           color=C_BEST, zorder=3)

    # ── delta labels ──────────────────────────────────────────────────
    ylo, yhi = cfg['ylim']
    pad = (yhi - ylo) * 0.018
    for x_pos, baseline, best, _ in pairs:
        delta = best - baseline
        if delta != 0:
            sign = '+' if delta > 0 else ''
            ax.text(x_pos + BAR_W / 2, best + pad,
                    f'{sign}{delta}%',
                    ha='center', va='bottom',
                    color=C_DELTA, fontweight='bold', fontsize=10)

    # ── x tick labels: adapt / custom ─────────────────────────────────
    ax.set_xticks(xs)
    ax.set_xticklabels(styles, fontsize=10)
    ax.tick_params(axis='x', length=0, pad=6)

    # ── group separator lines ─────────────────────────────────────────
    for sx in sep_xs:
        ax.axvline(sx, color=C_SEP, linewidth=1.2, linestyle='--', zorder=2)

    # ── group labels — tight, just below tick labels ───────────────────
    # Uses xaxis transform: x in data coords, y in axes fraction
    trans = ax.get_xaxis_transform()
    for gc_x, worker, mac in group_info:
        ax.text(gc_x, -0.14, f'Worker: {worker}  |  MAC: {mac}',
                transform=trans,
                ha='center', va='top',
                fontsize=8.5, color='#444444',
                clip_on=False)

    # ── styling ───────────────────────────────────────────────────────
    ax.set_title(cfg['title'], fontsize=14, fontweight='bold',
                 color=C_TITLE, pad=10)
    ax.set_ylabel(cfg['ylabel'], fontsize=11)
    ax.set_ylim(ylo, yhi)
    ax.yaxis.grid(True, alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.legend(
        handles=[
            mpatches.Patch(color=C_BASELINE, label='Baseline'),
            mpatches.Patch(color=C_BEST,     label='Best'),
        ],
        loc=cfg['legend_loc'], framealpha=0.9, fontsize=10,
    )

    plt.subplots_adjust(bottom=0.20)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {out_path}')
    plt.close()


if __name__ == '__main__':
    out_dir = Path('/tmp/mac_plots')
    out_dir.mkdir(exist_ok=True)
    for name, cfg in BENCHMARKS.items():
        make_plot(name, cfg, out_dir / f'{name}_results.png')
    print('Done.')
