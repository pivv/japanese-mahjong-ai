from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import numpy as np

import time

from collections import defaultdict

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils

class RankModel(nn.Module):
    def __init__(self):
        super(RankModel, self).__init__()
        n_input, d_rank, nc_rank = 9, 256, 4
        self.fc1_rank = nn.Linear(n_input, d_rank)
        nn.init.xavier_normal_(self.fc1_rank.weight)
        self.fc2_rank = nn.Linear(d_rank, d_rank)
        nn.init.xavier_normal_(self.fc2_rank.weight)
        self.fc3_rank = nn.Linear(d_rank, d_rank)
        nn.init.xavier_normal_(self.fc3_rank.weight)
        self.fc4_rank = nn.Linear(d_rank, nc_rank)
        nn.init.xavier_normal_(self.fc4_rank.weight)

    def forward(self, x):
        output1 = F.relu(self.fc1_rank(x))
        output2 = F.relu(self.fc2_rank(output1))
        output3 = F.relu(self.fc3_rank(output2))
        logits_rank = self.fc4_rank(output3)
        return logits_rank
