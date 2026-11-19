"""Train a Connect 4 AlphaZero-style agent by self-play.

Default settings reproduce the configuration in the report:
  30 iterations x 50 self-play games x 100 MCTS simulations per move, 10 epochs per iteration,
  batch size 64, 40 arena games, promotion threshold 0.6, seed 565.

Example:  python main.py --run_dir runs/main
"""
import argparse
import logging
import os
import random

import coloredlogs
import numpy as np
import torch

from Coach import Coach
from connect4.Connect4Game import Connect4Game as Game
from connect4.pytorch import NNet as nnet_module
from connect4.pytorch.NNet import NNetWrapper as nn
from utils import *

log = logging.getLogger(__name__)
coloredlogs.install(level='INFO')


def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--run_dir', default='runs/main', help='checkpoints/ and logs/ are created inside')
    p.add_argument('--seed', type=int, default=565)
    # search / self-play
    p.add_argument('--numIters', type=int, default=30)
    p.add_argument('--numEps', type=int, default=50, help='self-play games per iteration')
    p.add_argument('--numMCTSSims', type=int, default=100)
    p.add_argument('--cpuct', type=float, default=1.0)
    p.add_argument('--tempThreshold', type=int, default=15, help='temperature 1 for moves < this, then 0')
    p.add_argument('--arenaCompare', type=int, default=40)
    p.add_argument('--updateThreshold', type=float, default=0.6)
    p.add_argument('--maxlenOfQueue', type=int, default=200000)
    p.add_argument('--numItersForTrainExamplesHistory', type=int, default=20)
    # network / optimisation
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--dropout', type=float, default=0.3)
    p.add_argument('--num_channels', type=int, default=512)
    p.add_argument('--device', default='cpu', choices=['cpu', 'cuda', 'mps'])
    return p.parse_args()


def main():
    a = parse()
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    ckpt_dir = os.path.join(a.run_dir, 'checkpoints')
    log_dir = os.path.join(a.run_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    # network hyper-parameters live in connect4/pytorch/NNet.py (module-level dict); set them first
    nnet_module.args.update(lr=a.lr, dropout=a.dropout, epochs=a.epochs, batch_size=a.batch_size,
                            num_channels=a.num_channels, device=a.device,
                            loss_log=os.path.join(log_dir, 'losses.csv'))

    args = dotdict({
        'numIters': a.numIters, 'numEps': a.numEps, 'tempThreshold': a.tempThreshold,
        'updateThreshold': a.updateThreshold, 'maxlenOfQueue': a.maxlenOfQueue,
        'numMCTSSims': a.numMCTSSims, 'arenaCompare': a.arenaCompare, 'cpuct': a.cpuct,
        'checkpoint': ckpt_dir, 'log_dir': log_dir,
        'load_model': False, 'numItersForTrainExamplesHistory': a.numItersForTrainExamplesHistory,
    })
    # save the full configuration next to the logs so the run can be reproduced
    with open(os.path.join(log_dir, 'config.txt'), 'w') as f:
        f.write(f'torch {torch.__version__}  numpy {np.__version__}\n')
        for k, v in sorted(vars(a).items()):
            f.write(f'{k} = {v}\n')

    log.info('Loading %s...', Game.__name__)
    g = Game()  # 6 x 7, connect 4
    log.info('Loading %s...', nn.__name__)
    nnet = nn(g)
    log.info('Loading the Coach...')
    c = Coach(g, nnet, args)
    log.info('Starting the learning process')
    c.learn()


if __name__ == "__main__":
    main()
