"""Extra unit tests for the Connect 4 game logic (run with:  pytest connect4 ).

They check, independently of the implementation under test:
  * wins are detected in all four directions (horizontal, vertical, both diagonals) for both players
  * a full board with no four-in-a-row is a draw, and a full column rejects further stones
  * win detection agrees with a naive brute-force checker on thousands of random games
  * (if the UCI file is present) the UCI board encoding used by eval/value_head_uci.py is consistent
"""
import os
import random

import numpy as np
import pytest

from .Connect4Game import Connect4Game
from .test_connect4 import init_board_from_moves

H, W = 6, 7


def naive_winner(b):
    """Brute force: return 1 or -1 if that player has four in a row, else 0."""
    for r in range(H):
        for c in range(W):
            p = b[r][c]
            if p == 0:
                continue
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                rr, cc = r + 3 * dr, c + 3 * dc
                if 0 <= rr < H and 0 <= cc < W and all(b[r + k * dr][c + k * dc] == p for k in range(4)):
                    return int(p)
    return 0


def _result_after(moves):
    board, player, game = init_board_from_moves(moves)
    # result from the point of view of the player to move next (+1 win, -1 loss, 0 not over)
    return game.getGameEnded(board, player), player


def test_horizontal_win_player1():
    r, player = _result_after([0, 0, 1, 1, 2, 2, 3])  # player 1 completes 0-1-2-3 on the bottom row
    assert player == -1 and r == -1  # the player to move (player 2) has lost


def test_horizontal_win_player2():
    r, player = _result_after([6, 0, 6, 1, 5, 2, 5, 3])  # player 2 completes 0-1-2-3 on the bottom row
    assert player == 1 and r == -1


def test_vertical_win_both_players():
    r, player = _result_after([0, 1, 0, 1, 0, 1, 0])      # player 1 stacks four in column 0
    assert player == -1 and r == -1
    r, player = _result_after([0, 1, 0, 1, 0, 1, 2, 1])   # player 2 stacks four in column 1
    assert player == 1 and r == -1


# Move sequences ending in a diagonal four-in-a-row, found by random search and checked by the tests below
# (winner, direction, moves). The last move completes the diagonal.
DIAGONAL_GAMES = [
    (1, 'rising to the right', [4, 4, 3, 4, 1, 0, 4, 2, 2, 3, 3]),
    (1, 'rising to the left', [3, 4, 2, 2, 5, 2, 4, 3, 3, 4, 2]),
    (-1, 'rising to the right', [2, 6, 5, 3, 6, 5, 4, 4, 6, 6, 1, 5]),
    (-1, 'rising to the left', [0, 2, 1, 3, 1, 1, 4, 2, 4, 0, 0, 0]),
]


@pytest.mark.parametrize("winner,direction,moves", DIAGONAL_GAMES)
def test_diagonal_wins(winner, direction, moves):
    # the position before the last move must be unfinished, and the last move must win for `winner`
    r_before, _ = _result_after(moves[:-1])
    assert r_before == 0
    r, player = _result_after(moves)
    assert player == -winner and r == -1  # the player to move has lost, i.e. `winner` just won


def test_not_over_midgame():
    r, _ = _result_after([3, 3, 2, 4])
    assert r == 0


def test_full_board_is_draw():
    draw_moves = [5, 4, 5, 0, 6, 2, 4, 5, 5, 0, 4, 1, 1, 0, 4, 5, 6, 5, 3, 1, 1, 2, 2, 6, 2, 6, 6, 3, 6, 2, 0, 3, 0, 3, 3,
                  4, 3, 1, 4, 2, 1, 0]
    assert len(draw_moves) == 42
    board, player, game = init_board_from_moves(draw_moves)
    assert not game.getValidMoves(board, player).any()
    assert naive_winner(board) == 0
    assert game.getGameEnded(board, player) == pytest.approx(1e-4)  # draw value used by AlphaZero-General


def test_full_column_rejects_stone():
    with pytest.raises(ValueError):
        init_board_from_moves([0] * 7)


def test_win_detection_matches_brute_force_on_random_games():
    rng = random.Random(0)
    game = Connect4Game()
    n_wins = n_draws = 0
    for _ in range(3000):
        board, player = game.getInitBoard(), 1
        while True:
            valids = [c for c in range(W) if board[0][c] == 0]
            if not valids:
                break
            board, player = game.getNextState(board, player, rng.choice(valids))
            expected = naive_winner(board)
            result = game.getGameEnded(board, player)  # from the view of the player to move
            if expected != 0:
                assert result == (1 if expected == player else -1)
                n_wins += 1
                break
            assert result == 0 or (abs(result) < 0.5 and not (board[0] == 0).any())
        else:
            n_draws += 1
    assert n_wins > 2500  # sanity: the fuzz test really exercised wins


def test_symmetry_mirror_is_consistent():
    board, player, game = init_board_from_moves([0, 0, 1, 0, 6])
    pi = list(range(7))
    (b1, p1), (b2, p2) = game.getSymmetries(board, pi)
    assert (b2 == b1[:, ::-1]).all() and p2 == p1[::-1]
    assert game.getGameEnded(b2, player) == game.getGameEnded(b1, player)


UCI = os.path.join(os.path.dirname(__file__), '..', 'data', 'connect-4.data')


@pytest.mark.skipif(not os.path.exists(UCI), reason='UCI data file not present')
def test_uci_encoding_is_valid_gravity_boards():
    import pandas as pd
    from eval.value_head_uci import uci_to_canonical
    df = pd.read_csv(UCI, header=None, nrows=3000)
    boards = uci_to_canonical(df)
    assert ((boards == 1).sum((1, 2)) == 4).all() and ((boards == -1).sum((1, 2)) == 4).all()
    for b in boards:
        for c in range(W):
            col = b[:, c]
            filled = np.flatnonzero(col != 0)
            # stones form a contiguous stack resting on the bottom row (no floating stones)
            assert len(filled) == 0 or (filled.max() == H - 1 and len(filled) == H - filled.min())
        assert naive_winner(b) == 0  # UCI positions are non-terminal
