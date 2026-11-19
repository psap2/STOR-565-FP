import csv
import os
import sys

import numpy as np
from tqdm import tqdm

sys.path.append('../../')
from utils import *
from NeuralNet import NeuralNet

import torch
import torch.optim as optim

from .Connect4NNet import Connect4NNet as onnet

# Module-level config. main.py (and the eval scripts, from the checkpoint) update this BEFORE
# a NNetWrapper is built, so that Coach's competitor network gets the same architecture.
args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'num_channels': 512,
    'device': 'cpu',          # 'cpu', 'cuda' or 'mps'
    'loss_log': None,         # path of the per-epoch loss CSV (None = do not log)
})


class NNetWrapper(NeuralNet):
    # Number of epochs trained so far in this process; used as the x-axis of the loss plot.
    global_epoch = 0

    def __init__(self, game):
        self.device = torch.device(args.device)
        self.nnet = onnet(game, args).to(self.device)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.iteration = 0  # set by Coach so loss rows can be tagged

    def train(self, examples):
        """examples: list of (board, pi, v)."""
        optimizer = optim.Adam(self.nnet.parameters(), lr=args.lr)

        for epoch in range(args.epochs):
            print('EPOCH ::: ' + str(epoch + 1))
            self.nnet.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()

            batch_count = int(len(examples) / args.batch_size)

            t = tqdm(range(batch_count), desc='Training Net')
            for _ in t:
                sample_ids = np.random.randint(len(examples), size=args.batch_size)
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                boards = torch.FloatTensor(np.array(boards).astype(np.float64)).to(self.device)
                target_pis = torch.FloatTensor(np.array(pis)).to(self.device)
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64)).to(self.device)

                out_pi, out_v = self.nnet(boards)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(), boards.size(0))
                t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            NNetWrapper.global_epoch += 1
            self._log_epoch(epoch + 1, pi_losses.avg, v_losses.avg, batch_count)

    def _log_epoch(self, local_epoch, pi_loss, v_loss, n_batches):
        if not args.loss_log:
            return
        new = not os.path.exists(args.loss_log)
        os.makedirs(os.path.dirname(args.loss_log) or '.', exist_ok=True)
        with open(args.loss_log, 'a', newline='') as f:
            w = csv.writer(f)
            if new:
                w.writerow(['iteration', 'local_epoch', 'global_epoch', 'pi_loss', 'v_loss', 'n_batches'])
            w.writerow([self.iteration, local_epoch, NNetWrapper.global_epoch,
                        f'{pi_loss:.6f}', f'{v_loss:.6f}', n_batches])

    def predict(self, board):
        """board: np array (canonical form)."""
        board = torch.FloatTensor(board.astype(np.float64)).to(self.device)
        board = board.view(1, self.board_x, self.board_y)
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(board)
        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def predict_batch(self, boards):
        """boards: np array (n, 6, 7) of canonical boards. Returns (policy (n,7), value (n,))."""
        x = torch.FloatTensor(np.asarray(boards).astype(np.float64)).to(self.device)
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(x)
        return torch.exp(pi).cpu().numpy(), v.view(-1).cpu().numpy()

    def loss_pi(self, targets, outputs):
        # cross-entropy between the MCTS policy and the policy head
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        # mean squared error for the value head
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        os.makedirs(folder, exist_ok=True)
        torch.save({'state_dict': self.nnet.state_dict(), 'config': dict(args)}, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError("No model in path {}".format(filepath))
        checkpoint = torch.load(filepath, map_location=self.device)
        self.nnet.load_state_dict(checkpoint['state_dict'])
