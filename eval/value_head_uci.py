"""Evaluate the network's VALUE HEAD on the public UCI Connect-4 positions (Tromp, 1995).

Each UCI row is an 8-ply position (x to move) labelled win/loss/draw for x under perfect play.
The value head was never trained on these labels. We convert v in [-1, 1] to a class and compare
with the decision-tree baseline on the SAME held-out split (indices saved by the baseline script).

Class rule 1 (fixed, chosen before looking at results): v > +0.1 -> win, v < -0.1 -> loss, else draw.
Class rule 2 (calibrated): the two thresholds are tuned on the UCI TRAIN split (uses labels for two
scalars only) and then applied unchanged to the test split. Report rule 1 as the headline number.
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from eval.common import Connect4Game, load_nnet, wilson

CLASSES = ['loss', 'draw', 'win']


def uci_to_canonical(df):
    """UCI columns are a1..a6, b1..b6, ..., g1..g6 (row 1 = bottom). Returns (n, 6, 7) boards where
    x (the first player, who is to move after 8 plies) = +1 and o = -1; array row 0 = top row."""
    cells = df.iloc[:, :42].replace({'x': 1, 'o': -1, 'b': 0}).astype(int).values.reshape(-1, 7, 6)  # (n, col, row_from_bottom)
    boards = np.zeros((len(df), 6, 7), dtype=int)
    for c in range(7):
        for r in range(6):
            boards[:, 5 - r, c] = cells[:, c, r]
    return boards


def to_class(v, lo, hi):
    return np.where(v > hi, 'win', np.where(v < lo, 'loss', 'draw'))


def tune_thresholds(v, y):
    grid = np.round(np.arange(-0.6, 0.61, 0.05), 2)
    best = (-1, -0.1, 0.1)
    for lo, hi in itertools.product(grid, grid):
        if lo >= hi:
            continue
        acc = accuracy_score(y, to_class(v, lo, hi))
        if acc > best[0]:
            best = (acc, lo, hi)
    return best[1], best[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--split_dir', required=True, help='folder with uci_train_idx.npy / uci_test_idx.npy')
    ap.add_argument('--checkpoints', nargs='+', required=True)
    ap.add_argument('--out', default='results/value_head_uci.json')
    a = ap.parse_args()

    df = pd.read_csv(a.data, header=None)
    y = df.iloc[:, 42].values
    boards = uci_to_canonical(df)
    # sanity: every UCI position has exactly 4 x and 4 o stones
    assert ((boards == 1).sum((1, 2)) == 4).all() and ((boards == -1).sum((1, 2)) == 4).all()
    tr = np.load(os.path.join(a.split_dir, 'uci_train_idx.npy'))
    te = np.load(os.path.join(a.split_dir, 'uci_test_idx.npy'))

    game = Connect4Game()
    results = []
    for ck in a.checkpoints:
        net = load_nnet(game, ck)
        v = np.concatenate([net.predict_batch(boards[i:i + 2048])[1] for i in range(0, len(boards), 2048)])
        pred1 = to_class(v[te], -0.1, 0.1)
        lo, hi = tune_thresholds(v[tr], y[tr])
        pred2 = to_class(v[te], lo, hi)
        dec = y[te] != 'draw'
        res = dict(
            checkpoint=ck, n_test=len(te),
            acc_fixed_thresholds=round(accuracy_score(y[te], pred1), 4),
            acc_fixed_wilson95=[round(x, 4) for x in wilson(int((pred1 == y[te]).sum()), len(te))],
            confusion_fixed=confusion_matrix(y[te], pred1, labels=CLASSES).tolist(),
            calibrated_thresholds=[float(lo), float(hi)],
            acc_calibrated=round(accuracy_score(y[te], pred2), 4),
            # threshold-free: does a higher v mean 'win' rather than 'loss' (decisive positions only)?
            auc_win_vs_loss=round(roc_auc_score((y[te][dec] == 'win').astype(int), v[te][dec]), 4),
            mean_v_by_class={c: round(float(v[te][y[te] == c].mean()), 4) for c in CLASSES})
        results.append(res)
        print(f"{os.path.basename(ck):26s} acc(fixed) {res['acc_fixed_thresholds']:.4f}  "
              f"acc(calibrated) {res['acc_calibrated']:.4f}  AUC(win vs loss) {res['auc_win_vs_loss']:.4f}")
    base = os.path.join(a.split_dir, 'decision_tree_baseline.json')
    if os.path.exists(base):
        b = json.load(open(base))
        print(f"decision tree baseline: test acc {b['test_acc']}  majority-class acc {b['majority_test_acc']}")
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    json.dump(results, open(a.out, 'w'), indent=2)


if __name__ == '__main__':
    main()
