import sys
sys.path.append('..')

import torch
import torch.nn as nn
import torch.nn.functional as F


class Connect4NNet(nn.Module):
    """CNN policy/value network for Connect 4 (6x7 board).

    Adapted from the Othello PyTorch network in alpha-zero-general.
    Input : canonical board (player to move = +1, opponent = -1), shape (batch, 6, 7)
    Output: (log-softmax policy over 7 columns, tanh value in [-1, 1] for the player to move)

    Layers: 4 conv (3x3, `num_channels` filters, batch norm, ReLU; the last two are unpadded,
    so the 6x7 map shrinks to 2x3), then 2 fully connected layers (1024, 512) with batch norm,
    ReLU and dropout, then a 7-way policy head and a scalar value head.
    """

    def __init__(self, game, args):
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args
        super().__init__()

        c = args.num_channels
        self.conv1 = nn.Conv2d(1, c, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(c, c, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(c, c, 3, stride=1)
        self.conv4 = nn.Conv2d(c, c, 3, stride=1)

        self.bn1 = nn.BatchNorm2d(c)
        self.bn2 = nn.BatchNorm2d(c)
        self.bn3 = nn.BatchNorm2d(c)
        self.bn4 = nn.BatchNorm2d(c)

        self.fc1 = nn.Linear(c * (self.board_x - 4) * (self.board_y - 4), 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, self.action_size)  # policy head
        self.fc4 = nn.Linear(512, 1)                 # value head

    def forward(self, s):
        c = self.args.num_channels
        s = s.view(-1, 1, self.board_x, self.board_y)
        s = F.relu(self.bn1(self.conv1(s)))
        s = F.relu(self.bn2(self.conv2(s)))
        s = F.relu(self.bn3(self.conv3(s)))
        s = F.relu(self.bn4(self.conv4(s)))
        s = s.view(-1, c * (self.board_x - 4) * (self.board_y - 4))

        s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=self.args.dropout, training=self.training)
        s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=self.args.dropout, training=self.training)

        pi = self.fc3(s)
        v = self.fc4(s)
        return F.log_softmax(pi, dim=1), torch.tanh(v)
