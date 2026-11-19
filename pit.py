"""Play Connect 4 in the terminal against a trained network (human vs AI).

Example:  python pit.py --checkpoint runs/main/checkpoints/best.pth.tar --sims 100
Columns are numbered 0-6 from the left. You move first unless you pass --ai_first.
"""
import argparse

import numpy as np

from eval.common import Connect4Game, load_nnet, mcts_player
from connect4.Connect4Players import HumanConnect4Player

ap = argparse.ArgumentParser()
ap.add_argument('--checkpoint', required=True)
ap.add_argument('--sims', type=int, default=100)
ap.add_argument('--ai_first', action='store_true')
a = ap.parse_args()

game = Connect4Game()
ai = mcts_player(game, load_nnet(game, a.checkpoint), a.sims)()
human = HumanConnect4Player(game).play

board, player = game.getInitBoard(), 1
ai_player = 1 if a.ai_first else -1
while game.getGameEnded(board, player) == 0:
    game.display(board)
    canon = game.getCanonicalForm(board, player)
    action = ai(canon) if player == ai_player else human(canon)
    board, player = game.getNextState(board, player, action)
game.display(board)
r = game.getGameEnded(board, player) * player  # result for player 1
print('Draw' if abs(r) < 0.5 else ('Player 1 (x=+1) wins' if r > 0 else 'Player 2 (o=-1) wins'))
