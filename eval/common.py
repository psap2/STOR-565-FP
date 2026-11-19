"""Helpers shared by the evaluation scripts: loading checkpoints, players, Wilson intervals."""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from connect4.Connect4Game import Connect4Game
from connect4.pytorch import NNet as nnet_module
from connect4.pytorch.NNet import NNetWrapper
from MCTS import MCTS
from utils import dotdict


def wilson(k, n, z=1.96):
    """95% Wilson score interval for k successes out of n."""
    if n == 0:
        return (float('nan'), float('nan'))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def load_nnet(game, path, device='cpu'):
    """Load a checkpoint saved by NNetWrapper.save_checkpoint, rebuilding the architecture it was trained with."""
    ck = torch.load(path, map_location=device)
    cfg = ck.get('config', {})
    nnet_module.args.update({k: cfg[k] for k in ('dropout', 'num_channels') if k in cfg})
    nnet_module.args.update(device=device, loss_log=None)
    net = NNetWrapper(game)
    net.nnet.load_state_dict(ck['state_dict'])
    net.nnet.eval()
    return net


def mcts_player(game, net, sims, cpuct=1.0):
    """Returns a factory: call it at the start of each game to get a fresh-tree move function (temp = 0)."""
    def new_game_player():
        mcts = MCTS(game, net, dotdict({'numMCTSSims': sims, 'cpuct': cpuct}))
        return lambda board: int(np.argmax(mcts.getActionProb(board, temp=0)))
    return new_game_player
