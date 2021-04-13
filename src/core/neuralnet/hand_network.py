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
from ..transformer.Models import PositionwiseEncoder, MultiTransformer

class HandModel(nn.Module):
    def __init__(self):
        super(HandModel, self).__init__()
        d_round = 122
        len_max_base, n_layers_base, d_model_base, d_inner_base, \
            dropout_base, position_base = 1, 2, 512, 2048, 0.1, False # No length, 0 is used as src_pos
        n_encs = 2
        d_src1, len_max_src1, n_layers_enc1, n_head_enc1, d_k_enc1, d_v_enc1, d_model_enc1, d_inner_enc1, \
            dropout_enc1, position_enc1, relative_enc1 = 131, PAD_MELD, 2, 8, 64, 64, 512, 2048, 0.1, False, False # meld
        d_src2, len_max_src2, n_layers_enc2, n_head_enc2, d_k_enc2, d_v_enc2, d_model_enc2, d_inner_enc2, \
            dropout_enc2, position_enc2, relative_enc2 = 359, NUM_PLAYER-1, 2, 8, 64, 64, 512, 2048, 0.1, True, False # feature
        d_srcs, len_max_srcs, n_layers_encs, n_head_encs, d_k_encs, d_v_encs, d_model_encs, d_inner_encs, \
            dropout_encs, position_encs, relative_encs = [d_src1, d_src2], [len_max_src1, len_max_src2], \
            [n_layers_enc1, n_layers_enc2], [n_head_enc1, n_head_enc2], [d_k_enc1, d_k_enc2], [d_v_enc1, d_v_enc2], \
            [d_model_enc1, d_model_enc2], [d_inner_enc1, d_inner_enc2], [dropout_enc1, dropout_enc2], \
            [position_enc1, position_enc2], [relative_enc1, relative_enc2]

        d_tgt, len_max_tgt, n_layers_dec, n_head_dec, d_k_dec, d_v_dec, d_model_dec, d_inner_dec, \
            dropout_dec, position_dec, relative_dec = 55, PAD_HAND, 4, 24, 64, 64, 512, 2048, 0.1, False, False

        #y_throw_index_train, y_tsumo_train, y_ron_train, y_chi_train, y_pon_train, \
        #y_chakan_train, y_daiminkan_train, y_ankan_train, y_yao9_train, \
        #y_mean_score_train, y_reach_train, y_chied_train, y_poned_train, \
        #y_win_tsumo_train, y_win_ron_train, y_lose_train, y_win_num_train, \

        d_throw_index, nc_throw_index = 256, 1
        d_tsumo, nc_tsumo = 256, 1
        d_ron, nc_ron = 1024, 5
        d_meld, nc_meld = 1024, 8
        d_yao9, nc_yao9 = 256, 1
        d_mean_score, nc_mean_score = 256, 1
        d_reach, nc_reach = 256, 1
        d_melded, nc_melded = 1024, 6
        d_win, nc_win = 1024, 3
        #d_win_num, nc_win_num = 256, 1
        d_next_tile, nc_next_tile = 1024, 34

        d_value, nc_value = 256, 1
        d_mode, nc_mode = 1024, 3

        self.round_encoder = PositionwiseEncoder(d_round, len_max_base, n_layers_base,
            d_model_base, d_inner_base, dropout_base, position_base)
        self.transformer = MultiTransformer(n_encs, d_srcs, len_max_srcs, d_tgt, len_max_tgt,
            n_layers_encs, n_head_encs, d_k_encs, d_v_encs, d_model_encs, d_inner_encs, dropout_encs, position_encs, relative_encs,
            n_layers_dec, n_head_dec, d_k_dec, d_v_dec, d_model_dec, d_inner_dec, dropout_dec, position_dec, relative_dec)

        self.fc1_throw_index = nn.Linear(d_model_dec, d_throw_index)
        nn.init.xavier_normal_(self.fc1_throw_index.weight)
        self.fc2_throw_index = nn.Linear(d_throw_index, nc_throw_index)
        nn.init.xavier_normal_(self.fc2_throw_index.weight)

        self.fc1_tsumo = nn.Linear(d_model_dec, d_tsumo)
        nn.init.xavier_normal_(self.fc1_tsumo.weight)
        self.fc2_tsumo = nn.Linear(d_tsumo, nc_tsumo)
        nn.init.xavier_normal_(self.fc2_tsumo.weight)

        self.fc1_ron = nn.Linear(d_model_dec, d_ron)
        nn.init.xavier_normal_(self.fc1_ron.weight)
        self.fc2_ron = nn.Linear(d_ron, nc_ron)
        nn.init.xavier_normal_(self.fc2_ron.weight)

        self.fc1_meld = nn.Linear(d_model_dec, d_meld)
        nn.init.xavier_normal_(self.fc1_meld.weight)
        self.fc2_meld = nn.Linear(d_meld, nc_meld)
        nn.init.xavier_normal_(self.fc2_meld.weight)

        self.fc1_yao9 = nn.Linear(d_model_dec, d_yao9)
        nn.init.xavier_normal_(self.fc1_yao9.weight)
        self.fc2_yao9 = nn.Linear(d_yao9, nc_yao9)
        nn.init.xavier_normal_(self.fc2_yao9.weight)

        self.fc1_mean_score = nn.Linear(d_model_dec, d_mean_score)
        nn.init.xavier_normal_(self.fc1_mean_score.weight)
        self.fc2_mean_score = nn.Linear(d_mean_score, nc_mean_score)
        nn.init.xavier_normal_(self.fc2_mean_score.weight)

        self.fc1_reach = nn.Linear(d_model_dec, d_reach)
        nn.init.xavier_normal_(self.fc1_reach.weight)
        self.fc2_reach = nn.Linear(d_reach, nc_reach)
        nn.init.xavier_normal_(self.fc2_reach.weight)

        self.fc1_melded = nn.Linear(d_model_dec, d_melded)
        nn.init.xavier_normal_(self.fc1_melded.weight)
        self.fc2_melded = nn.Linear(d_melded, nc_melded)
        nn.init.xavier_normal_(self.fc2_melded.weight)

        self.fc1_win = nn.Linear(d_model_dec, d_win)
        nn.init.xavier_normal_(self.fc1_win.weight)
        self.fc2_win = nn.Linear(d_win, nc_win)
        nn.init.xavier_normal_(self.fc2_win.weight)

        #self.fc1_win_num = nn.Linear(d_model_dec, d_win_num)
        #nn.init.xavier_normal_(self.fc1_win_num.weight)
        #self.fc2_win_num = nn.Linear(d_win_num, nc_win_num)
        #nn.init.xavier_normal_(self.fc2_win_num.weight)

        self.fc1_next_tile = nn.Linear(d_model_dec, d_next_tile)
        nn.init.xavier_normal_(self.fc1_next_tile.weight)
        self.fc2_next_tile = nn.Linear(d_next_tile, nc_next_tile)
        nn.init.xavier_normal_(self.fc2_next_tile.weight)

        # These are made for reinforcement step.

        self.fc1_value = nn.Linear(d_model_dec, d_value)
        nn.init.xavier_normal_(self.fc1_value.weight)
        self.fc2_value = nn.Linear(d_value, nc_value)
        nn.init.xavier_normal_(self.fc2_value.weight)

        self.fc1_mode = nn.Linear(d_model_dec, d_mode)
        nn.init.xavier_normal_(self.fc1_mode.weight)
        self.fc2_mode = nn.Linear(d_mode, nc_mode)
        nn.init.xavier_normal_(self.fc2_mode.weight)

    def forward(self, x_round, x_meld, pos_meld, x_hand, pos_hand, mask_hand, x_remain, x_furiten, thrown_feature2, output_feature=None):
        use_mask_encs, use_mask_dec = [False, False], False

        if output_feature is None:
            pos_round = torch.LongTensor([0]).to(x_round.device).unsqueeze(0).expand(x_round.size(0), -1)
            output_round = self.round_encoder(torch.cat([x_round, x_remain, x_furiten], dim=-1).unsqueeze(1), pos_round)
            pos_thrown_feature2 = torch.LongTensor([0, 1, 2]).to(
                thrown_feature2.device).unsqueeze(0).expand(thrown_feature2.size(0), -1)
            srcs_seq = [x_meld, thrown_feature2]
            srcs_pos = [pos_meld, pos_thrown_feature2]
            output_feature = self.transformer(srcs_seq, srcs_pos, x_hand, pos_hand,
                base_output=output_round, use_mask_encs=use_mask_encs, use_mask_dec=use_mask_dec)
                #base_output=output_round.expand(-1, PAD_HAND, -1), use_mask_encs=use_mask_encs, use_mask_dec=use_mask_dec)

        output_throw_index = self.fc2_throw_index(F.relu(self.fc1_throw_index(output_feature))).squeeze(-1) # sz_b x len_max
        pos_mask = get_pos_mask(pos_hand, PAD_HAND).type(torch.uint8)
        padding_mask = pos_mask.eq(0)
        output_throw_index = output_throw_index.masked_fill(padding_mask, -np.inf) # padding logits
        output_throw_index = output_throw_index.masked_fill(mask_hand.gt(0), -np.inf) # padding logits

        pos_mask_float = pos_mask.type(torch.float).unsqueeze(2)
        output_mean_feature = torch.sum(output_feature * pos_mask_float, dim=1) / torch.sum(pos_mask_float, dim=1) # sz_b x d_model

        output_tsumo = self.fc2_tsumo(F.relu(self.fc1_tsumo(output_mean_feature)))
        output_tsumo = output_tsumo[:, 0] # sz_b
        output_ron = self.fc2_ron(F.relu(self.fc1_ron(output_feature))) # sz_b x len_max x 5
        output_meld = self.fc2_meld(F.relu(self.fc1_meld(output_feature)))
        output_chi = output_meld[:, :, :4] # sz_b x len_max x 4
        output_pon = output_meld[:, :, 4] # sz_b x len_max
        output_chakan = output_meld[:, :, 5] # sz_b x len_max
        output_daiminkan = output_meld[:, :, 6] # sz_b x len_max
        output_ankan = output_meld[:, :, 7] # sz_b x len_max
        output_yao9 = self.fc2_yao9(F.relu(self.fc1_yao9(output_mean_feature)))
        output_yao9 = output_yao9[:, 0] # sz_b
        output_mean_score = F.relu(self.fc2_mean_score(F.relu(self.fc1_mean_score(output_mean_feature))))
        output_mean_score = output_mean_score[:, 0]# sz_b
        output_reach = self.fc2_reach(F.relu(self.fc1_reach(output_feature)))
        output_reach = output_reach[:, :, 0] # sz_b x len_max
        output_melded = self.fc2_melded(F.relu(self.fc1_melded(output_feature)))
        output_chied = output_melded[:, :, :3] # sz_b x len_max x 3
        output_poned = output_melded[:, :, 3:6] # sz_b x len_max x 3
        output_win = self.fc2_win(F.relu(self.fc1_win(output_mean_feature)))
        output_win_tsumo = output_win[:, 0] # sz_b
        output_win_ron = output_win[:, 1] # sz_b
        output_lose = output_win[:, 2] # sz_b
        #output_win_num = F.relu(self.fc2_win_num(F.relu(self.fc1_win_num(output_mean_feature))))
        #output_win_num = output_win_num[:, 0] # sz_b
        output_next_tile = self.fc2_next_tile(F.relu(self.fc1_next_tile(output_mean_feature))) # sz_b x NUM_TILE_TYPE
        output_value = self.fc2_value(F.relu(self.fc1_value(output_mean_feature)))
        output_value = output_value[:, 0] # sz_b

        # no output for mode

        return output_feature, output_throw_index, output_tsumo, output_ron, output_chi, output_pon, \
            output_chakan, output_daiminkan, output_ankan, output_yao9, \
            output_mean_score, output_reach, output_chied, output_poned, \
            output_win_tsumo, output_win_ron, output_lose, \
            output_next_tile, output_value
