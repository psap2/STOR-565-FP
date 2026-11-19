"""Turn the training logs into the report's numbers and figures.

Reads  <run_dir>/logs/{arena,losses,selfplay}.csv
Writes <out_dir>/fig1_arena_winrate.png, fig2_losses.png, arena_summary.json

Definitions (fixed before looking at the results):
  decisive win rate = new_wins / (new_wins + prev_wins)   (draws ignored, exactly the promotion rule)
  'reached threshold' = decisive win rate >= updateThreshold (0.6)
  'collapsed'         = decisive win rate <= 0.2 (the new network lost at least 4 of every 5 decisive games)
"""
import argparse
import json
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_dir', default='runs/main')
    ap.add_argument('--out_dir', default='results')
    ap.add_argument('--threshold', type=float, default=0.6)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    logs = os.path.join(a.run_dir, 'logs')

    arena = pd.read_csv(os.path.join(logs, 'arena.csv'))
    arena['decisive'] = arena.new_wins + arena.prev_wins
    arena['wr'] = arena.new_wins / arena.decisive.replace(0, np.nan)
    ci = np.array([wilson(k, n) for k, n in zip(arena.new_wins, arena.decisive)])

    summary = dict(
        iterations=int(len(arena)),
        reached_threshold=int((arena.wr >= a.threshold).sum()),
        accepted=int(arena.accepted.sum()),
        collapsed=int((arena.wr <= 0.2).sum()),
        mean_decisive_win_rate=round(float(arena.wr.mean()), 4),
        games_per_arena=int(arena.games.iloc[0]),
        mean_draws_per_arena=round(float(arena.draws.mean()), 2),
        mean_distinct_games_per_arena=round(float(arena.distinct_games.mean()), 2),
        min_distinct_games=int(arena.distinct_games.min()),
        iterations_reached_threshold=[int(i) for i in arena.iteration[arena.wr >= a.threshold]],
        iterations_collapsed=[int(i) for i in arena.iteration[arena.wr <= 0.2]],
        total_train_hours=round(float(arena.seconds.sum()) / 3600, 2),
    )
    sp = os.path.join(logs, 'selfplay.csv')
    if os.path.exists(sp):
        s = pd.read_csv(sp)
        summary['avg_game_length'] = round(float(s.avg_game_length.mean()), 2)
        summary['avg_examples_per_iteration'] = round(float(s.examples.mean()), 1)
        summary['selfplay_first_player_win_rate'] = round(float(s.first_player_wins.sum() / s.games.sum()), 3)
        summary['selfplay_draw_rate'] = round(float(s.draws.sum() / s.games.sum()), 3)
    json.dump(summary, open(os.path.join(a.out_dir, 'arena_summary.json'), 'w'), indent=2)
    print(json.dumps(summary, indent=2))

    # Figure 1: arena win rate per iteration with Wilson 95% intervals
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = arena.iteration.values
    ax.errorbar(x, arena.wr, yerr=[arena.wr - ci[:, 0], ci[:, 1] - arena.wr], fmt='none', ecolor='0.7', capsize=2)
    acc = arena.accepted == 1
    ax.plot(x[~acc], arena.wr[~acc], 'o', color='tab:red', label='rejected')
    ax.plot(x[acc], arena.wr[acc], 'o', color='tab:green', label='accepted')
    ax.axhline(a.threshold, ls='--', color='k', lw=1, label=f'promotion threshold ({a.threshold:.0%})')
    ax.axhline(0.5, ls=':', color='0.5', lw=1)
    ax.set_xlabel('Training iteration'); ax.set_ylabel('Win rate vs previous best (draws excluded)')
    ax.set_ylim(-0.02, 1.02); ax.legend(loc='lower right', fontsize=8)
    ax.set_title('Arena win rate of the new network (bars: 95% Wilson interval)')
    fig.tight_layout(); fig.savefig(os.path.join(a.out_dir, 'fig1_arena_winrate.png'), dpi=200); plt.close(fig)

    # Figure 2: policy and value loss per epoch
    lp = os.path.join(logs, 'losses.csv')
    if os.path.exists(lp):
        l = pd.read_csv(lp)
        l = l[l.iteration <= int(arena.iteration.max())]  # drop an unfinished final iteration
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
        for ax, col, name in zip(axes, ['pi_loss', 'v_loss'], ['Policy loss (cross-entropy)', 'Value loss (MSE)']):
            ax.plot(l.global_epoch, l[col], lw=1)
            for it in l.groupby('iteration').global_epoch.min().values[1:]:
                ax.axvline(it - 0.5, color='0.9', lw=0.6, zorder=0)
            ax.set_xlabel('Global epoch'); ax.set_title(name)
        fig.tight_layout(); fig.savefig(os.path.join(a.out_dir, 'fig2_losses.png'), dpi=200); plt.close(fig)


if __name__ == '__main__':
    main()
