"""Fixed (non-learning) opponents used to measure absolute playing strength.

All players take a *canonical* board (player to move = +1, opponent = -1, shape 6x7, row 0 = top)
and return a column index.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from connect4.Connect4Players import OneStepLookaheadConnect4Player

H, W = 6, 7
ORDER = [3, 2, 4, 1, 5, 0, 6]  # centre columns first (better alpha-beta pruning)

# all 69 horizontal / vertical / diagonal windows of 4 cells, as flat indices
_WINDOWS = []
for r in range(H):
    for c in range(W):
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            cells = [(r + k * dr, c + k * dc) for k in range(4)]
            if all(0 <= rr < H and 0 <= cc < W for rr, cc in cells):
                _WINDOWS.append([rr * W + cc for rr, cc in cells])
WINDOWS = np.array(_WINDOWS)
assert len(WINDOWS) == 69


class RandomPlayer:
    """Uniformly random legal move."""
    def __init__(self, game):
        self.game = game

    def play(self, board):
        valids = np.flatnonzero(self.game.getValidMoves(board, 1))
        return int(np.random.choice(valids))


class OneStepPlayer(OneStepLookaheadConnect4Player):
    """Takes an immediate win, blocks an immediate loss, otherwise plays randomly."""
    def __init__(self, game):
        super().__init__(game, verbose=False)

    def play(self, board):
        return int(super().play(board))


def _drop(board, col, p):
    for r in range(H - 1, -1, -1):
        if board[r, col] == 0:
            board[r, col] = p
            return r
    return -1


def _wins_at(board, r, c, p):
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        n = 1
        for sgn in (1, -1):
            rr, cc = r + sgn * dr, c + sgn * dc
            while 0 <= rr < H and 0 <= cc < W and board[rr, cc] == p:
                n += 1
                rr += sgn * dr
                cc += sgn * dc
        if n >= 4:
            return True
    return False


def _heuristic(board, p):
    vals = board.ravel()[WINDOWS]
    mine = (vals == p).sum(1)
    theirs = (vals == -p).sum(1)
    weight = np.array([0, 1, 4, 16, 0])
    s = weight[mine][theirs == 0].sum() - weight[theirs][mine == 0].sum()
    s += 3 * ((board[:, 3] == p).sum() - (board[:, 3] == -p).sum())
    return int(s)


def _negamax(board, depth, alpha, beta, p):
    if depth == 0:
        return _heuristic(board, p)
    best = -10 ** 9
    any_move = False
    for col in ORDER:
        if board[0, col] != 0:
            continue
        any_move = True
        r = _drop(board, col, p)
        if _wins_at(board, r, col, p):
            score = 10 ** 6 + depth  # faster wins score higher
        else:
            score = -_negamax(board, depth - 1, -beta, -alpha, -p)
        board[r, col] = 0
        if score > best:
            best = score
        alpha = max(alpha, score)
        if alpha >= beta:
            break
    return best if any_move else 0  # full board = draw


class MinimaxPlayer:
    """Depth-limited alpha-beta (negamax) with a simple window-counting heuristic. Deterministic."""
    def __init__(self, game, depth=4):
        self.game = game
        self.depth = depth

    def play(self, board):
        board = np.array(board, dtype=int)
        best, best_col = -10 ** 9, None
        alpha = -10 ** 9
        for col in ORDER:
            if board[0, col] != 0:
                continue
            r = _drop(board, col, 1)
            if _wins_at(board, r, col, 1):
                return col
            score = -_negamax(board, self.depth - 1, -10 ** 9, -alpha, -1)
            board[r, col] = 0
            if score > best:
                best, best_col = score, col
            alpha = max(alpha, score)
        return int(best_col)


def make_opponent(name, game):
    if name == 'random':
        return RandomPlayer(game).play
    if name == 'onestep':
        return OneStepPlayer(game).play
    if name.startswith('minimax'):
        return MinimaxPlayer(game, depth=int(name[len('minimax'):] or 4)).play
    raise ValueError(name)
