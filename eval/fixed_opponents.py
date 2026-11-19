"""Absolute playing strength: AlphaZero network (+ MCTS) vs fixed opponents.

For each checkpoint and opponent we play `--games` games. Games come in colour-swapped pairs that
share the same random opening (`--opening_plies` random plies, both sides), so that the result is
not decided by who moves first and the games are not duplicates of each other.
The network plays with `--sims` MCTS simulations per move (temperature 0, fresh search tree per game).

Example:
  python eval/fixed_opponents.py --checkpoints runs/main/checkpoints/checkpoint_0.pth.tar \
      runs/main/checkpoints/best.pth.tar --opponents random onestep minimax4 --games 100 --procs 4 \
      --out results/fixed_opponents.json
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from eval.common import Connect4Game, load_nnet, mcts_player, wilson
from eval.players import make_opponent

_cache = {}


def _get_net(path):
    if path not in _cache:
        import torch
        torch.set_num_threads(1)
        _cache[path] = load_nnet(Connect4Game(), path)
    return _cache[path]


def make_opening(seed, plies, game):
    rng = np.random.RandomState(seed)
    board, player, moves = game.getInitBoard(), 1, []
    for _ in range(plies):
        valids = np.flatnonzero(game.getValidMoves(board, player))
        a = int(rng.choice(valids))
        moves.append(a)
        board, player = game.getNextState(board, player, a)
    return moves


def play_one(task):
    """Plays one game; returns the result from the network's point of view (+1 win, 0 draw, -1 loss)."""
    ckpt, opp_name, opening, net_first, sims, seed = task
    np.random.seed(seed)
    game = Connect4Game()
    net = _get_net(ckpt)
    az = mcts_player(game, net, sims)()
    opp = make_opponent(opp_name, game)

    board, player = game.getInitBoard(), 1
    for a in opening:
        board, player = game.getNextState(board, player, a)
    while game.getGameEnded(board, player) == 0:
        net_to_move = (player == 1) == net_first
        a = (az if net_to_move else opp)(game.getCanonicalForm(board, player))
        assert game.getValidMoves(board, player)[a], 'illegal move'
        board, player = game.getNextState(board, player, a)
    r = game.getGameEnded(board, player) * player  # result for player 1 (the first mover)
    r = 0 if abs(r) < 0.5 else int(np.sign(r))
    return r if net_first else -r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoints', nargs='+', required=True)
    ap.add_argument('--opponents', nargs='+', default=['random', 'onestep', 'minimax4'])
    ap.add_argument('--games', type=int, default=100, help='per checkpoint and opponent (even number)')
    ap.add_argument('--sims', type=int, default=100)
    ap.add_argument('--opening_plies', type=int, default=4)
    ap.add_argument('--seed', type=int, default=565)
    ap.add_argument('--procs', type=int, default=1)
    ap.add_argument('--out', default='results/fixed_opponents.json')
    a = ap.parse_args()
    assert a.games % 2 == 0

    game = Connect4Game()
    rows = []
    pool = mp.Pool(a.procs) if a.procs > 1 else None
    for ck in a.checkpoints:
        for opp in a.opponents:
            t0 = time.time()
            tasks = []
            for g in range(a.games // 2):
                opening = make_opening(a.seed * 1000 + g, a.opening_plies, game)
                tasks.append((ck, opp, opening, True, a.sims, a.seed + 2 * g))
                tasks.append((ck, opp, opening, False, a.sims, a.seed + 2 * g + 1))
            res = pool.map(play_one, tasks) if pool else [play_one(t) for t in tasks]
            w, d, l = res.count(1), res.count(0), res.count(-1)
            lo, hi = wilson(w + 0.5 * d, a.games)  # score rate (draw = half a win)
            rows.append(dict(checkpoint=ck, opponent=opp, games=a.games, sims=a.sims,
                             wins=w, draws=d, losses=l, score=round((w + 0.5 * d) / a.games, 4),
                             score_wilson95=[round(lo, 4), round(hi, 4)], seconds=round(time.time() - t0)))
            print(f'{os.path.basename(ck):28s} vs {opp:9s}  W/D/L = {w}/{d}/{l}  score {rows[-1]["score"]:.3f}'
                  f'  95% CI [{lo:.3f}, {hi:.3f}]  ({rows[-1]["seconds"]}s)', flush=True)
            os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
            json.dump(rows, open(a.out, 'w'), indent=2)


if __name__ == '__main__':
    main()
