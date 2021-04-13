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

from ..game.mahjong_constant import *
from ..transformer.Models import get_pos_mask, flatten_maybe_padded_sequences
from ..transformer.Models import PositionwiseEncoder, Encoder, Decoder, Transformer

class ThrownModel(nn.Module):
    def __init__(self, model_type):
        super(ThrownModel, self).__init__() # 0 or 1
        self.model_type = model_type
        d_round, d_safe = 88, 34
        #d_round, d_safe = 52, 68
        len_max_base, n_layers_base, d_model_base, d_inner_base, \
            dropout_base, position_base = PAD_THROWN, 2, 512, 2048, 0.1, False
        d_src, len_max_src, n_layers_enc, n_head_enc, d_k_enc, d_v_enc, d_model_enc, d_inner_enc, \
            dropout_enc, position_enc, relative_enc = 131, PAD_CUM_MELD, 2, 8, 64, 64, 512, 2048, 0.1, False, False
        d_tgt, len_max_tgt, n_layers_dec, n_head_dec, d_k_dec, d_v_dec, d_model_dec, d_inner_dec, \
            dropout_dec, position_dec, relative_dec = 103, PAD_THROWN, 4, 24, 64, 64, 512, 2048, 0.1, True, True

        self.round_encoder = PositionwiseEncoder(d_round, len_max_base, n_layers_base,
            d_model_base, d_inner_base, dropout_base, position_base)
        self.safe_encoder = PositionwiseEncoder(d_safe, len_max_base, n_layers_base,
            d_model_base, d_inner_base, dropout_base, position_base)
        self.transformer = Transformer(d_src, len_max_src, d_tgt, len_max_tgt,
            n_layers_enc, n_head_enc, d_k_enc, d_v_enc, d_model_enc, d_inner_enc, dropout_enc, position_enc, relative_enc,
            n_layers_dec, n_head_dec, d_k_dec, d_v_dec, d_model_dec, d_inner_dec, dropout_dec, position_dec, relative_dec)

        if self.model_type == 0:
            d_tenpai, nc_tenpai = 256, 1
            d_waiting, nc_waiting = 1024, 34
            d_score, nc_score = 1024, 35
            d_win, nc_win = 256, 1
            #d_win_num, nc_win_num = 256, 1

            self.fc1_tenpai = nn.Linear(d_model_dec, d_tenpai)
            nn.init.xavier_normal_(self.fc1_tenpai.weight)
            self.fc2_tenpai = nn.Linear(d_tenpai, nc_tenpai)
            nn.init.xavier_normal_(self.fc2_tenpai.weight)

            self.fc1_waiting = nn.Linear(d_model_dec, d_waiting)
            nn.init.xavier_normal_(self.fc1_waiting.weight)
            self.fc2_waiting = nn.Linear(d_waiting, nc_waiting)
            nn.init.xavier_normal_(self.fc2_waiting.weight)

            self.fc1_score = nn.Linear(d_model_dec, d_score)
            nn.init.xavier_normal_(self.fc1_score.weight)
            self.fc2_score = nn.Linear(d_score, nc_score)
            nn.init.xavier_normal_(self.fc2_score.weight)

            self.fc1_win = nn.Linear(d_model_dec, d_win)
            nn.init.xavier_normal_(self.fc1_win.weight)
            self.fc2_win = nn.Linear(d_win, nc_win)
            nn.init.xavier_normal_(self.fc2_win.weight)

            #self.fc1_win_num = nn.Linear(d_model_dec, d_win_num)
            #nn.init.xavier_normal_(self.fc1_win_num.weight)
            #self.fc2_win_num = nn.Linear(d_win_num, nc_win_num)
            #nn.init.xavier_normal_(self.fc2_win_num.weight)
        else:
            assert(self.model_type == 1)
            d_my_hand, nc_my_hand = 1024, 34
            d_next_use_call, nc_next_use_call = 256, 1
            d_next_call_type, nc_next_call_type = 1024, 55
            d_next_thrown, nc_next_thrown = 1024, 34
            d_win, nc_win = 1024, 3

            self.fc1_my_hand = nn.Linear(d_model_dec, d_my_hand)
            nn.init.xavier_normal_(self.fc1_my_hand.weight)
            self.fc2_my_hand = nn.Linear(d_my_hand, nc_my_hand)
            nn.init.xavier_normal_(self.fc2_my_hand.weight)

            self.fc1_next_use_call = nn.Linear(d_model_dec, d_next_use_call)
            nn.init.xavier_normal_(self.fc1_next_use_call.weight)
            self.fc2_next_use_call = nn.Linear(d_next_use_call, nc_next_use_call)
            nn.init.xavier_normal_(self.fc2_next_use_call.weight)

            self.fc1_next_call_type = nn.Linear(d_model_dec, d_next_call_type)
            nn.init.xavier_normal_(self.fc1_next_call_type.weight)
            self.fc2_next_call_type = nn.Linear(d_next_call_type, nc_next_call_type)
            nn.init.xavier_normal_(self.fc2_next_call_type.weight)

            self.fc1_next_thrown = nn.Linear(d_model_dec, d_next_thrown)
            nn.init.xavier_normal_(self.fc1_next_thrown.weight)
            self.fc2_next_thrown = nn.Linear(d_next_thrown, nc_next_thrown)
            nn.init.xavier_normal_(self.fc2_next_thrown.weight)

            self.fc1_win = nn.Linear(d_model_dec, d_win)
            nn.init.xavier_normal_(self.fc1_win.weight)
            self.fc2_win = nn.Linear(d_win, nc_win)
            nn.init.xavier_normal_(self.fc2_win.weight)

    def forward(self, x_round, x_meld, pos_meld, mask_meld, x_thrown, pos_thrown, x_remain, x_safe,
        output_thrown=None, output_feature=None, user_feature=None, return_output_thrown=False):
        if self.model_type == 0:
            use_mask_enc, use_mask_dec = False, True
        else:
            assert(self.model_type == 1)
            use_mask_enc, use_mask_dec = True, True

        if output_feature is None:
            if output_thrown is None:
                if len(x_round.size()) == 3: # normal
                    output_round = self.round_encoder(torch.cat([x_round, x_remain], dim=-1), pos_thrown)
                else:
                    assert(len(x_round.size()) == 2)
                    pos_round = torch.LongTensor([0]).to(x_round.device).unsqueeze(0).expand(x_round.size(0), -1)
                    output_round = self.round_encoder(torch.cat([x_round, x_remain], dim=-1).unsqueeze(1), pos_round)
                output_thrown = self.transformer(x_meld, pos_meld, x_thrown, pos_thrown, base_output=output_round,
                    use_mask_enc=use_mask_enc, use_mask_dec=use_mask_dec, init_dec_enc_attn_mask=mask_meld)
                if return_output_thrown:
                    return output_thrown
            if len(x_safe.size()) == 3: # normal
                output_safe = self.safe_encoder(x_safe, pos_thrown)
            else:
                assert(len(x_safe.size()) == 2)
                pos_safe = torch.LongTensor([0]).to(x_safe.device).unsqueeze(0).expand(x_safe.size(0), -1)
                output_safe = self.safe_encoder(x_safe.unsqueeze(1), pos_safe)
            output_feature = output_thrown + output_safe
        if user_feature is not None:
            output_feature = output_feature + user_feature

        if self.model_type == 0:
            output_tenpai = self.fc2_tenpai(F.relu(self.fc1_tenpai(output_feature)))
            output_tenpai = output_tenpai[:, :, 0]
            output_waiting = self.fc2_waiting(F.relu(self.fc1_waiting(output_feature)))
            #output_score = F.relu(self.fc2_score(F.relu(self.fc1_score(output_feature))))
            #output_mean_score = F.relu(self.fc2_mean_score(F.relu(self.fc1_mean_score(output_feature))))
            #output_mean_score = output_mean_score[:, :, 0]
            output_init_score = F.relu(self.fc2_score(F.relu(self.fc1_score(output_feature))))
            output_score = output_init_score[:, :, :-1]
            output_mean_score = output_init_score[:, :, -1]
            output_win = self.fc2_win(F.relu(self.fc1_win(output_feature)))
            output_win = output_win[:, :, 0]
            #output_win_num = F.relu(self.fc2_win_num(F.relu(self.fc1_win_num(output_feature))))
            #output_win_num = output_win_num[:, :, 0]
            return output_thrown, output_feature, output_tenpai, output_waiting, output_score, output_mean_score, output_win
                #output_win, output_win_num
        else:
            assert(self.model_type == 1)
            output_my_hand = 4. * F.sigmoid(self.fc2_my_hand(F.relu(self.fc1_my_hand(output_feature))))
            output_next_use_call = self.fc2_next_use_call(F.relu(self.fc1_next_use_call(output_feature)))
            output_next_use_call = output_next_use_call[:, :, 0]
            output_next_call_type = self.fc2_next_call_type(F.relu(self.fc1_next_call_type(output_feature)))
            output_next_thrown = self.fc2_next_thrown(F.relu(self.fc1_next_thrown(output_feature)))

            output_win = self.fc2_win(F.relu(self.fc1_win(output_feature)))
            output_win_tsumo = output_win[:, :, 0]
            output_win_ron = output_win[:, :, 1]
            output_win_lose = output_win[:, :, 2]
            return output_thrown, output_feature, output_my_hand, output_next_use_call, output_next_call_type, output_next_thrown, \
                output_win_tsumo, output_win_ron, output_win_lose
