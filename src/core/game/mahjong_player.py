''' Mahjong players decide whether to do the action.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time

import numpy as np
import random
from copy import copy, deepcopy

from collections import defaultdict

from ..util.utils import lengths_to_pos, lengths_to_mask
from .mahjong_constant import *
from .mahjong_game import MahjongGame

from ..neuralnet.hand_network import HandModel
from ..neuralnet.thrown_network import ThrownModel
from ..neuralnet.rank_network import RankModel

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

class BaseMahjongPlayer(object):
    def __init__(self, do_meld=True):
        self.do_meld = do_meld

    def play(self, iplayer, mahjong_board, event):
        event_type = event[0]
        assert(event_type in ['defg', 'agari', 'n', 'ryuukyoku'])
        if event_type == 'defg':
            return self.play_reach_and_defg(iplayer, mahjong_board, event)
        elif event_type == 'agari':
            agari_from_who = event[-1][0]
            if iplayer == agari_from_who:
                return self.play_tsumo(iplayer, mahjong_board, event)
            else:
                return self.play_ron(iplayer, mahjong_board, event)
        elif event_type == 'n':
            if not self.do_meld:
                return []
            play_fn = getattr(self, 'play_' + event[1])
            return play_fn(iplayer, mahjong_board, event)
        else:
            assert(event_type == 'ryuukyoku')
            assert(event[1] == 'yao9')
            return self.play_yao9(iplayer, mahjong_board, event)

    def play_reach_and_defg(self, iplayer, mahjong_board, event):
        event_defg = self.play_defg(iplayer, mahjong_board, event)[0]
        if mahjong_board.players[iplayer].is_reach_available:
            thrown_tile = event_defg[2]
            hand = mahjong_board.players[iplayer].hand[:]
            thrown_index = hand.index(thrown_tile)
            del hand[thrown_index]
            if mahjong_board.compute_hand_is_tenpai(hand):
                events_reach = self.play_reach(iplayer, mahjong_board)
                return events_reach + [event_defg]
        return [event_defg]

    def play_reach(self, iplayer, mahjong_board):
        raise NotImplementedError()

    def play_defg(self, iplayer, mahjong_board, event):
        raise NotImplementedError()

    def play_tsumo(self, iplayer, mahjong_board, event):
        raise NotImplementedError()

    def play_ron(self, iplayer, mahjong_board, event):
        raise NotImplementedError()

    def play_chi(self, iplayer, mahjong_board, event):
        raise NotImplementedError()

    def play_pon(self, iplayer, mahjong_board, event):
        raise NotImplementedError()

    def play_daiminkan(self, iplayer, mahjong_board, event):
        raise NotImplementedError()

    def play_chakan(self, iplayer, mahjong_board, event):
        raise NotImplementedError()

    def play_ankan(self, iplayer, mahjong_board, event):
        raise NotImplementedError()

    def play_yao9(self, iplayer, mahjong_board, event):
        raise NotImplementedError()


class RandomMahjongPlayer(BaseMahjongPlayer):
    def __init__(self):
        super(RandomMahjongPlayer, self).__init__()

    def play_reach(self, iplayer, mahjong_board):
        return [('reach', iplayer, 1)] # always reach

    def play_defg(self, iplayer, mahjong_board, event):
        hand = mahjong_board.players[iplayer].hand
        if mahjong_board.players[iplayer].is_reached: # should tsumo-giri
            thrown_tile = hand[-1]
            return [('defg', iplayer, thrown_tile, None)]
        tiles_cannot_be_thrown = event[2]
        available_indices = [index for index in range(len(hand)) if hand[index] not in tiles_cannot_be_thrown]
        if len(available_indices) == 0:
            throw_index = 0
        else:
            throw_index = available_indices[int(np.random.random_sample() * len(available_indices))]
        thrown_tile = hand[throw_index]
        return [('defg', iplayer, thrown_tile, None)]

    def play_tsumo(self, iplayer, mahjong_board, event):
        return [event] # always tsumo

    def play_ron(self, iplayer, mahjong_board, event):
        return [event] # always ron

    def play_chi(self, iplayer, mahjong_board, event):
        chi_availables = event[3]
        chi_index = int(np.random.random_sample() * len(chi_availables)) - 1
        if chi_index == -1:
            return []
        else:
            return [('n', 'chi', iplayer, chi_availables[chi_index])]

    def play_pon(self, iplayer, mahjong_board, event):
        pon_availables = event[3]
        pon_index = int(np.random.random_sample() * len(pon_availables)) - 1
        if pon_index == -1:
            return []
        else:
            return [('n', 'pon', iplayer, pon_availables[pon_index])]

    def play_daiminkan(self, iplayer, mahjong_board, event):
        daiminkan_availables = event[3]
        daiminkan_index = int(np.random.random_sample() * len(daiminkan_availables)) - 1
        if daiminkan_index == -1:
            return []
        else:
            return [('n', 'daiminkan', iplayer, daiminkan_availables[daiminkan_index])]

    def play_chakan(self, iplayer, mahjong_board, event):
        chakan_availables = event[3]
        chakan_index = int(np.random.random_sample() * len(chakan_availables)) - 1
        if chakan_index == -1:
            return []
        else:
            return [('n', 'chakan', iplayer, chakan_availables[chakan_index])]

    def play_ankan(self, iplayer, mahjong_board, event):
        ankan_availables = event[3]
        ankan_index = int(np.random.random_sample() * len(ankan_availables)) - 1
        if ankan_index == -1:
            return []
        else:
            return [('n', 'ankan', iplayer, ankan_availables[ankan_index])]

    def play_yao9(self, iplayer, mahjong_board, event):
        return [('ryuukyoku', 'yao9')] # always ryuukyoku


class VirtualMahjongPlayer(BaseMahjongPlayer):
    def __init__(self):
        super(VirtualMahjongPlayer, self).__init__()

    def set_virtual_params(self, iplayer, prob_tenpai_all, probs_waiting_all, ypred_score_all,
        prob_next_use_call_all, probs_next_call_type_all, probs_next_thrown_all, probs_chied, probs_poned, # my info
        safe_tile_types_all, remain_tile_type_nums): # TODO: more info about opponent meld
        self.my_player = iplayer
        self.prob_tenpai_all = prob_tenpai_all
        self.probs_waiting_all = probs_waiting_all
        self.ypred_score_all = ypred_score_all
        self.prob_next_use_call_all = prob_next_use_call_all
        self.probs_next_call_type_all = probs_next_call_type_all
        self.probs_next_thrown_all = probs_next_thrown_all
        self.probs_chied = probs_chied
        self.probs_poned = probs_poned
        self.safe_tile_types_all = safe_tile_types_all
        self.remain_tile_type_nums = remain_tile_type_nums

    def play_reach_and_defg(self, iplayer, mahjong_board, event): # Virtual player cannot declare reach.
        event_defg = self.play_defg(iplayer, mahjong_board, event)[0]
        return [event_defg]

    def play_reach(self, iplayer, mahjong_board):
        raise NotImplementedError("Virtual player cannot declare reach.")

    def play_defg(self, iplayer, mahjong_board, event):
        assert(iplayer != self.my_player)
        thresh_tenpai = 0.6
        risk_value = 0.4
        thresh_score = 1.2
        thresh_use_call = 0.095
        thresh_call_type = 0.08
        # last-tsumoed tile is in mahjong_board.agari_tile
        if mahjong_board.players[iplayer].is_reached: # should tsumo-giri
            thrown_tile = mahjong_board.agari_tile
            return [('defg', iplayer, thrown_tile, True)]
        # TODO: should block risk tile
        probs_next_thrown = np.copy(self.probs_next_thrown_all[(iplayer-self.my_player)%NUM_PLAYER])
        probs_next_thrown[self.remain_tile_type_nums == 0] = 0.
        for j in range(3):
            jplayer = (iplayer+1+j)%NUM_PLAYER
            if self.prob_tenpai_all[(jplayer-self.my_player)%NUM_PLAYER] > thresh_tenpai:
                risk_block = (self.probs_waiting_all[(jplayer-self.my_player)%NUM_PLAYER] *
                    np.exp(self.ypred_score_all[(jplayer-self.my_player)%NUM_PLAYER] + risk_value) / 4. >= thresh_score)
                probs_next_thrown[risk_block] = 0.
                #print('score risk block: ', np.sum(risk_block))
            if self.prob_next_use_call_all[(jplayer-self.my_player)%NUM_PLAYER] > thresh_use_call:
                risk_block = self.probs_next_call_type_all[(jplayer-self.my_player)%NUM_PLAYER, 21:] >= thresh_call_type
                probs_next_thrown[risk_block] = 0.
                #print('pon risk block: ', np.sum(risk_block))
        if np.sum(probs_next_thrown) == 0: # maybe some problem happened.
            print("Warning: probs of next thrown is vanished. we don't use risk block")
            probs_next_thrown = np.copy(self.probs_next_thrown_all[(iplayer-self.my_player)%NUM_PLAYER])
        probs_next_thrown = probs_next_thrown / np.sum(probs_next_thrown)
        thrown_tile = np.random.choice(NUM_TILE_TYPE, p=probs_next_thrown) * 4 + 1 # No aka
        return [('defg', iplayer, thrown_tile, mahjong_board.players[iplayer].last_tsumo_giri)]

    def play_tsumo(self, iplayer, mahjong_board, event):
        # in simulation there are no tsumo.
        thresh_tenpai = 0.5
        penalty_value = 1.
        assert(iplayer != self.my_player)
        # block safe tile types
        probs_waiting = np.copy(self.probs_waiting_all[(iplayer-self.my_player)%NUM_PLAYER])
        safe_tile_types = self.safe_tile_types_all[(iplayer-self.my_player)%NUM_PLAYER]
        probs_waiting[safe_tile_types == 1] = 0.
        prob_tenpai = self.prob_tenpai_all[(iplayer-self.my_player)%NUM_PLAYER]
        ypred_score = self.ypred_score_all[(iplayer-self.my_player)%NUM_PLAYER]
        risk_value = 0.4
        agari_tile_type = mahjong_board.agari_tile // 4
        if prob_tenpai > thresh_tenpai:
            do_tsumo = int(np.random.random_sample() < probs_waiting[agari_tile_type] * penalty_value)
        else:
            do_tsumo = int(np.random.random_sample() < prob_tenpai * probs_waiting[agari_tile_type] * penalty_value)
        if do_tsumo == 1:
            # score
            assert(event[-1][1] == 'tsumo')
            basic_score = np.exp(ypred_score[agari_tile_type] + risk_value) / 4.
            more_value = 0
            if mahjong_board.check_player_is_menzen(iplayer): # menzen tsumo
                more_value += 1
            if mahjong_board.players[iplayer].is_ippatsu: # ippatsu
                more_value += 1
            if more_value > 0:
                if basic_score < 20. and 2.**(more_value) * basic_score < 20.:
                    basic_score = 2.**(more_value) * basic_score
                elif basic_score < 20. and 2.**(more_value) * basic_score >= 20.:
                    basic_score = 20.
                elif basic_score >= 20. and basic_score < 30. and more_value >= 2:
                    basic_score = 30.
                elif basic_score >= 30. and basic_score < 40. and more_value >= 2:
                    basic_score = 40.
            event = ('agari', iplayer, (iplayer, basic_score, None, None, None, None))
            return [event]
        else:
            return []

    def play_ron(self, iplayer, mahjong_board, event):
        assert(iplayer != self.my_player)
        agari_from_who = event[-1][0]
        thresh_tenpai = 0.5
        penalty_value = 2.
        if agari_from_who != self.my_player: # no ron from other in simulation
            penalty_value = 1.
        # block safe tile types
        probs_waiting = np.copy(self.probs_waiting_all[(iplayer-self.my_player)%NUM_PLAYER])
        safe_tile_types = self.safe_tile_types_all[(iplayer-self.my_player)%NUM_PLAYER]
        probs_waiting[safe_tile_types == 1] = 0.
        prob_tenpai = self.prob_tenpai_all[(iplayer-self.my_player)%NUM_PLAYER]
        ypred_score = self.ypred_score_all[(iplayer-self.my_player)%NUM_PLAYER]
        risk_value = 0.4
        agari_tile_type = mahjong_board.agari_tile // 4
        if prob_tenpai > thresh_tenpai:
            do_ron = int(np.random.random_sample() < probs_waiting[agari_tile_type] * penalty_value)
        else:
            do_ron = int(np.random.random_sample() < prob_tenpai * probs_waiting[agari_tile_type] * penalty_value)
        if do_ron == 1:
            # score
            assert(event[-1][1] in ['ron', 'chakan'])
            basic_score = np.exp(ypred_score[agari_tile_type] + risk_value) / 4.
            more_value = 0
            if event[-1][1] == 'chakan': # chakan
                more_value += 1
            if mahjong_board.players[iplayer].is_ippatsu: # ippatsu
                more_value += 1
            if more_value > 0:
                if basic_score < 20. and 2.**(more_value) * basic_score < 20.:
                    basic_score = 2.**(more_value) * basic_score
                elif basic_score < 20. and 2.**(more_value) * basic_score >= 20.:
                    basic_score = 20.
                elif basic_score >= 20. and basic_score < 30. and more_value >= 2:
                    basic_score = 30.
                elif basic_score >= 30. and basic_score < 40. and more_value >= 2:
                    basic_score = 40.
            event = ('agari', iplayer, (agari_from_who, basic_score, None, None, None, None))
            return [event]
        else:
            return []

    def play_chi(self, iplayer, mahjong_board, event):
        assert(iplayer == (self.my_player+1)%NUM_PLAYER)
        assert(not mahjong_board.players[iplayer].is_reached)
        meld_from_who = event[-1][0]
        assert(meld_from_who == self.my_player)
        penalty_value = 3.
        thrown_tile = event[-1][1]
        base = thrown_tile // 4
        if base >= 27:
            return []
        prob_chied = -np.inf
        meld_called = -1
        if base % 9 <= 6 and self.remain_tile_type_nums[base+1] > 0 and self.remain_tile_type_nums[base+2] > 0:
            if self.probs_chied[0] > prob_chied:
                prob_chied = self.probs_chied[0]
                meld_called = 0
        if base % 9 >= 2 and self.remain_tile_type_nums[base-1] > 0 and self.remain_tile_type_nums[base-2] > 0:
            if self.probs_chied[2] > prob_chied:
                prob_chied = self.probs_chied[2]
                meld_called = 2
        if (1 <= base % 9 <= 7) and self.remain_tile_type_nums[base-1] > 0 and self.remain_tile_type_nums[base+1] > 0:
            if self.probs_chied[1] > prob_chied:
                prob_chied = self.probs_chied[1]
                meld_called = 1
        if meld_called == -1:
            return []
        do_chi = int(np.random.random_sample() < prob_chied * penalty_value)
        if do_chi == 0:
            return []
        else:
            if meld_called == 0:
                meld_tiles = [thrown_tile, (base+1)*4+1, (base+2)*4+1]
            elif meld_called == 1:
                meld_tiles = [(base-1)*4+1, thrown_tile, (base+1)*4+1]
            else:
                assert(meld_called == 2)
                meld_tiles = [(base-2)*4+1, (base-1)*4+1, thrown_tile]
            chi_available = (meld_from_who, meld_called, meld_tiles)
            return [('n', 'chi', iplayer, chi_available)]

    def play_pon(self, iplayer, mahjong_board, event):
        assert(iplayer != self.my_player)
        assert(not mahjong_board.players[iplayer].is_reached)
        prob_poned = self.probs_poned[(iplayer-self.my_player-1)%NUM_PLAYER]
        meld_from_who = event[-1][0]
        assert(meld_from_who == self.my_player)
        penalty_value = 3.
        thrown_tile = event[-1][1]
        base = thrown_tile // 4
        if self.remain_tile_type_nums[base] < 2: # cannot declare pon
            return []
        do_pon = int(np.random.random_sample() < prob_poned * penalty_value)
        if do_pon == 0:
            return []
        else:
            meld_called = 0
            meld_tiles = [thrown_tile, base*4+1, base*4+1]
            pon_available = (meld_from_who, meld_called, meld_tiles)
            return [('n', 'pon', iplayer, pon_available)]

    def play_daiminkan(self, iplayer, mahjong_board, event):
        raise NotImplementedError("Virtual player cannot declare daiminkan.")

    def play_chakan(self, iplayer, mahjong_board, event):
        raise NotImplementedError("Virtual player cannot declare chakan.")

    def play_ankan(self, iplayer, mahjong_board, event):
        raise NotImplementedError("Virtual player cannot declare ankan.")

    def play_yao9(self, iplayer, mahjong_board, event):
        raise NotImplementedError("Virtual player cannot declare yao9.")


class HumanMahjongPlayer(BaseMahjongPlayer):
    def __init__(self):
        super(HumanMahjongPlayer, self).__init__()

    def play_reach(self, iplayer, mahjong_board):
        print('You can declare reach.')
        print('Press 1 to declare reach, and 0 otherwise.')
        do_reach = -1
        while do_reach not in [0, 1]:
            do_reach = input()
        if do_reach == 1:
            return [('reach', iplayer, 1)]
        else:
            return []

    def play_defg(self, iplayer, mahjong_board, event):
        hand = mahjong_board.players[iplayer].hand
        if mahjong_board.players[iplayer].is_reached: # should tsumo-giri
            print('You declared reach, so should tsumo-giri.')
            thrown_tile = hand[-1]
            return [('defg', iplayer, thrown_tile, None)]
        tiles_cannot_be_thrown = event[2]
        available_indices = [index for index in range(len(hand)) if hand[index] not in tiles_cannot_be_thrown]
        if len(available_indices) == 0:
            available_indices = [index for index in range(len(hand))]
        print('You can throw these tile indices : ' + ', '.join('%d'%(index) for index in available_indices))
        print('Choose the index of tile to throw.')
        throw_index = -1
        while throw_index not in available_indices:
            try:
                throw_index = input()
            except:
                print('Invalid input.')
        thrown_tile = hand[throw_index]
        return [('defg', iplayer, thrown_tile, None)]

    def play_tsumo(self, iplayer, mahjong_board, event):
        print('You can declare tsumo.')
        print('Press 1 to declare tsumo, and 0 otherwise.')
        do_tsumo = -1
        while do_tsumo not in [0, 1]:
            try:
                do_tsumo = input()
            except:
                print('Invalid input.')
        if do_tsumo == 1:
            return [event]
        else:
            return []

    def play_ron(self, iplayer, mahjong_board, event):
        print('You can declare ron with tile: %d'%(mahjong_board.agari_tile))
        print('Press 1 to declare ron, and 0 otherwise.')
        do_ron = -1
        while do_ron not in [0, 1]:
            try:
                do_ron = input()
            except:
                print('Invalid input.')
        if do_ron == 1:
            return [event]
        else:
            return []

    def play_chi(self, iplayer, mahjong_board, event):
        hand = mahjong_board.players[iplayer].hand
        chi_availables = event[3]
        meld_from_who = chi_availables[0][0]
        call_tile = chi_availables[0][2][chi_availables[0][1]]
        print('You can declare chi.')
        print('You can meld from %d, Call tile: %d'%((meld_from_who-iplayer)%NUM_PLAYER, call_tile))
        print('Available chis: ' + ', '.join('(' + ','.join(
            '%d'%(hand.index(meld_tiles[tile_index])) for tile_index in range(3) if tile_index != meld_called) +
            ')' for (meld_from_who, meld_called, meld_tiles) in chi_availables))
        print('Press the index of chi, or press -1 not to declare chi.')
        chi_index = -2
        while chi_index not in range(-1, len(chi_availables)):
            try:
                chi_index = input()
            except:
                print('Invalid input.')
        if chi_index == -1:
            return []
        else:
            return [('n', 'chi', iplayer, chi_availables[chi_index])]

    def play_pon(self, iplayer, mahjong_board, event):
        hand = mahjong_board.players[iplayer].hand
        pon_availables = event[3]
        meld_from_who = pon_availables[0][0]
        call_tile = pon_availables[0][2][pon_availables[0][1]]
        print('You can declare pon.')
        print('You can meld from %d, Call tile: %d'%((meld_from_who-iplayer)%NUM_PLAYER, call_tile))
        print('Available pons: ' + ', '.join('(' + ','.join(
            '%d'%(hand.index(meld_tiles[tile_index])) for tile_index in range(3) if tile_index != meld_called) +
            ')' for (meld_from_who, meld_called, meld_tiles) in pon_availables))
        print('Press the index of pon, or press -1 not to declare pon.')
        pon_index = -2
        while pon_index not in range(-1, len(pon_availables)):
            try:
                pon_index = input()
            except:
                print('Invalid input.')
        if pon_index == -1:
            return []
        else:
            return [('n', 'pon', iplayer, pon_availables[pon_index])]

    def play_daiminkan(self, iplayer, mahjong_board, event):
        hand = mahjong_board.players[iplayer].hand
        daiminkan_availables = event[3]
        meld_from_who = daiminkan_availables[0][0]
        call_tile = daiminkan_availables[0][2][daiminkan_availables[0][1]]
        print('You can declare daiminkan.')
        print('You can meld from %d, Call tile: %d'%((meld_from_who-iplayer)%NUM_PLAYER, call_tile))
        print('Available daiminkans: ' + ', '.join('(' + ','.join(
            '%d'%(hand.index(meld_tiles[tile_index])) for tile_index in range(4) if tile_index != meld_called) +
            ')' for (meld_from_who, meld_called, meld_tiles) in daiminkan_availables))
        print('Press the index of daiminkan, or press -1 not to declare daiminkan.')
        daiminkan_index = -2
        while daiminkan_index not in range(-1, len(daiminkan_availables)):
            try:
                daiminkan_index = input()
            except:
                print('Invalid input.')
        if daiminkan_index == -1:
            return []
        else:
            return [('n', 'daiminkan', iplayer, daiminkan_availables[daiminkan_index])]

    def play_chakan(self, iplayer, mahjong_board, event):
        hand = mahjong_board.players[iplayer].hand
        chakan_availables = event[3]
        print('You can declare chakan.')
        print('Available chakans: ' + ', '.join('%d'%(hand.index(meld_tiles[-1])) for
            (meld_from_who, meld_called, meld_tiles) in chakan_availables))
        print('Press the index of chakan, or press -1 not to declare chakan.')
        chakan_index = -2
        while chakan_index not in range(-1, len(chakan_availables)):
            chakan_index = input()
        if chakan_index == -1:
            return []
        else:
            return [('n', 'chakan', iplayer, chakan_availables[chakan_index])]

    def play_ankan(self, iplayer, mahjong_board, event):
        hand = mahjong_board.players[iplayer].hand
        ankan_availables = event[3]
        print('You can declare ankan.')
        print('Available ankans: ' + ', '.join('(' + ','.join(
            '%d'%(hand.index(meld_tiles[tile_index])) for tile_index in range(4)) +
            ')' for (meld_from_who, meld_called, meld_tiles) in ankan_availables))
        print('Press the index of ankan, or press -1 not to declare ankan.')
        ankan_index = -2
        while ankan_index not in range(-1, len(ankan_availables)):
            ankan_index = input()
        if ankan_index == -1:
            return []
        else:
            return [('n', 'ankan', iplayer, ankan_availables[ankan_index])]

    def play_yao9(self, iplayer, mahjong_board, event):
        print('You can declare yao9.')
        print('Press 1 to declare yao9, and 0 otherwise.')
        do_yao9 = -1
        while do_yao9 not in [0, 1]:
            do_yao9 = input()
        if do_yao9 == 1:
            return [('ryuukyoku', 'yao9')]
        else:
            return []

class SimpleDeepAIPlayer(BaseMahjongPlayer):
    def __init__(self, device, netHand, netThrown0, netThrown1, is_greedy=True, do_meld=True):
        super(SimpleDeepAIPlayer, self).__init__(do_meld)
        self.device = device
        self.netHand = netHand
        self.netThrown0 = netThrown0
        self.netThrown1 = netThrown1
        self.is_greedy = is_greedy

        self._prob_reach = None

        #self.netHand.eval()
        #self.netThrown0.eval()
        #self.netThrown1.eval()
        #torch.set_grad_enabled(False)

        self.reset_local_variables()


    def reset_local_variables(self):
        self.safe_tile_types_all = None
        self.prob_tenpai_all = None
        self.probs_waiting_all = None
        self.ypred_score_all = None
        self.ypred_mean_score_all = None
        self.ypred_my_hand_all = None
        self.prob_next_use_call_all = None
        self.probs_next_call_type_all = None
        self.probs_next_thrown_all = None
        self.prob_win_tsumo_all = None
        self.prob_win_ron_all = None
        self.prob_lose_all = None
        self.thrown_feature_batch = None
        self.ypred_mean_score = None
        self.probs_chied = None
        self.probs_poned = None
        self.prob_win_tsumo = None
        self.prob_win_ron = None
        self.prob_lose = None

    def share_local_variables(self):
        self.safe_tile_types_all.share_memory_()
        self.prob_tenpai_all.share_memory_()
        self.probs_waiting_all.share_memory_()
        self.ypred_score_all.share_memory_()
        self.ypred_mean_score_all.share_memory_()
        self.ypred_my_hand_all.share_memory_()
        self.prob_next_use_call_all.share_memory_()
        self.probs_next_call_type_all.share_memory_()
        self.probs_next_thrown_all.share_memory_()
        self.prob_win_tsumo_all.share_memory_()
        self.prob_win_ron_all.share_memory_()
        self.prob_lose_all.share_memory_()
        self.thrown_feature_batch.share_memory_()
        self.ypred_mean_score.share_memory_()
        self.probs_chied.share_memory_()
        self.probs_poned.share_memory_()
        self.prob_win_tsumo.share_memory_()
        self.prob_win_ron.share_memory_()
        self.prob_lose.share_memory_()

    def acquire_basic_hand_data(self, basic_hand_infos, mask_hand_infos):
        assert(len(basic_hand_infos) == 1)
        lengths_hand_train = np.array([len(basic_hand_info) for basic_hand_info in basic_hand_infos]).astype('int')
        len_max = np.max(lengths_hand_train)
        assert(len_max > 0)
        pos_hand_train = lengths_to_pos(lengths_hand_train, len_max, PAD_HAND)
        temp_range = np.arange(len(basic_hand_infos))

        xinit_hand_train = np.array(basic_hand_infos)
        assert(xinit_hand_train.shape == (1, len_max, HAND_TILE_INFO_DIMENSION))
        hand_tiles_train = xinit_hand_train[:, :, 0]
        #hand_aka_infos_train = xinit_hand_train[:, :, 6]
        x_hand_train = np.zeros((1, len_max, HAND_TILE_INFO_DIMENSION+NUM_TILE_TYPE-1), dtype='uint8')
        assert(np.max(xinit_hand_train[:, :, 0]) < NUM_TILE_TYPE)
        for index_pad in range(len_max):
            x_hand_train[temp_range, index_pad, xinit_hand_train[:, index_pad, 0]] = 1
        x_hand_train[:, :, NUM_TILE_TYPE:] = xinit_hand_train[:, :, 1:]

        mask_hand_train = np.array(mask_hand_infos)
        assert(mask_hand_train.shape == (1, len_max))
        if np.all(mask_hand_train == 1): # cannot throw!! activate all
            mask_hand_train[:] = 0

        return x_hand_train, pos_hand_train, mask_hand_train, hand_tiles_train

    def acquire_basic_thrown_data(self, basic_thrown_infos):
        assert(len(basic_thrown_infos) == 4)
        lengths_thrown_train = np.array([len(basic_thrown_info) for basic_thrown_info in basic_thrown_infos]).astype('int')
        len_max = np.max(lengths_thrown_train)
        assert(len_max > 0)
        pos_thrown_train = lengths_to_pos(lengths_thrown_train, len_max, PAD_THROWN)
        pos_mask_train = (pos_thrown_train == (lengths_thrown_train-1).reshape(-1, 1)).astype('uint8')
        assert(np.all(np.sum(pos_mask_train, axis=-1) == 1))
        temp_range = np.arange(len(basic_thrown_infos))
        for itrain, basic_thrown_info in enumerate(basic_thrown_infos):
            basic_thrown_infos[itrain] += [[-1] * THROWN_TILE_INFO_DIMENSION for _ in range(len_max-len(basic_thrown_info))]

        xinit_thrown_train = np.array(basic_thrown_infos)
        assert(xinit_thrown_train.shape == (4, len_max, THROWN_TILE_INFO_DIMENSION))
        thrown_tiles_train = xinit_thrown_train[:, :, 0]
        #thrown_aka_infos_train = xinit_thrown_train[:, :, 6]
        x_thrown_train = np.zeros((4, len_max, THROWN_TILE_INFO_DIMENSION+NUM_TILE_TYPE+NUM_CALL_TYPE), dtype='uint8')
        assert(np.max(xinit_thrown_train[:, :, 0]) < NUM_TILE_TYPE+1)
        for index_pad in range(len_max):
            x_thrown_train[temp_range, index_pad, xinit_thrown_train[:, index_pad, 0]] = 1
        x_thrown_train[:, :, NUM_TILE_TYPE+1:-NUM_CALL_TYPE-1] = xinit_thrown_train[:, :, 1:-1]
        assert(np.max(xinit_thrown_train[:, :, -1]) <= NUM_CALL_TYPE)
        for index_pad in range(len_max):
            x_thrown_train[temp_range, index_pad, -NUM_CALL_TYPE-1+xinit_thrown_train[:, index_pad, -1]] = 1
        x_thrown_train = x_thrown_train[:, :, :-1]

        return x_thrown_train, pos_thrown_train, pos_mask_train, thrown_tiles_train

    def acquire_hand_meld_data(self, melds_infos):
        assert(len(melds_infos) == 1)
        lengths_meld_train = np.array([len(melds_info) for melds_info in melds_infos]).astype('int')
        len_max = np.max(lengths_meld_train)
        if len_max == 0:
            len_max = 1
        pos_meld_train = lengths_to_pos(lengths_meld_train, len_max, PAD_MELD)
        temp_range = np.arange(len(melds_infos))
        for itrain, melds_info in enumerate(melds_infos):
            melds_infos[itrain] += [[-1] * HAND_MELD_INFO_DIMENSION for _ in range(len_max-len(melds_info))]

        xinit_meld_train = np.array(melds_infos)
        assert(xinit_meld_train.shape == (1, len_max, HAND_MELD_INFO_DIMENSION))
        melds_train = xinit_meld_train[:, :, 0]
        #meld_aka_infos_train = xinit_meld_train[:, :, 2]
        x_meld_train = np.zeros((1, len_max, HAND_MELD_INFO_DIMENSION+NUM_MELD_TYPE-1), dtype='uint8')
        for index_pad in range(len_max):
            x_meld_train[temp_range, index_pad, xinit_meld_train[:, index_pad, 0]] = 1
        x_meld_train[:, :, NUM_MELD_TYPE:] = xinit_meld_train[:, :, 1:]

        return x_meld_train, pos_meld_train, melds_train

    def acquire_thrown_meld_data(self, melds_infos, lengths_all_melds_infos, mask_melds_infos):
        assert(len(melds_infos) == 4)
        len_max_thrown = int(np.max([len(lengths_all_melds_info) for lengths_all_melds_info in lengths_all_melds_infos]))
        for itrain, lengths_all_melds_info in enumerate(lengths_all_melds_infos):
            lengths_all_melds_infos[itrain] += [0] * (len_max_thrown-len(lengths_all_melds_info))
        lengths_all_meld_train = np.array(lengths_all_melds_infos)
        assert(lengths_all_meld_train.shape == (4, len_max_thrown))
        lengths_meld_train = np.array([len(melds_info) for melds_info in melds_infos]).astype('int')
        len_max = np.max(lengths_meld_train)
        if len_max == 0:
            len_max = 1
        mask_meld_train = lengths_to_mask(lengths_all_meld_train, len_max)
        pos_meld_train = lengths_to_pos(lengths_meld_train, len_max, PAD_CUM_MELD)
        temp_range = np.arange(len(melds_infos))
        for itrain, melds_info in enumerate(melds_infos):
            melds_infos[itrain] += [[-1] * CUM_MELD_INFO_DIMENSION for _ in range(len_max-len(melds_info))]

        # meld info
        xinit_meld_train = np.array(melds_infos)
        assert(xinit_meld_train.shape == (4, len_max, CUM_MELD_INFO_DIMENSION))
        x_meld_train = np.zeros((4, len_max, CUM_MELD_INFO_DIMENSION+NUM_MELD_TYPE-1), dtype='uint8')
        for index_pad in range(len_max):
            x_meld_train[temp_range, index_pad, xinit_meld_train[:, index_pad, 0]] = 1
        x_meld_train[:, :, NUM_MELD_TYPE:] = xinit_meld_train[:, :, 1:]

        # meld mask info
        for itrain, mask_melds_info in enumerate(mask_melds_infos):
            mask_melds_infos[itrain] += [[0] * 8 for _ in range(len_max_thrown-len(mask_melds_info))]
        more_mask_meld_train = np.array(mask_melds_infos)[:, :, :len_max]
        assert(more_mask_meld_train.shape == (4, len_max_thrown, len_max))
        mask_meld_train = (mask_meld_train + more_mask_meld_train > 0).astype('uint8')
        assert(np.all(np.sum(mask_meld_train == 0, axis=-1) <= 4))

        melds_train = xinit_meld_train[:, :, 0] # do not remove chakan at this time for convenience
        #meld_aka_infos_train = xinit_meld_train[:, :, 2]

        return x_meld_train, pos_meld_train, mask_meld_train, melds_train

    def predict_thrown(self, iplayer, mahjong_board):
        device = self.device

        for j in range(NUM_PLAYER):
            jplayer = (iplayer+j)%NUM_PLAYER
            assert(len(mahjong_board.players[jplayer].thrown_infos) == len(mahjong_board.players[jplayer].thrown)+1)

        thrown_infos = [mahjong_board.players[(iplayer+j)%NUM_PLAYER].thrown_infos for j in range(NUM_PLAYER)]
        melds_infos = [mahjong_board.acquire_player_cum_melds_info((iplayer+j)%NUM_PLAYER) for j in range(NUM_PLAYER)]
        basic_thrown_infos = [[thrown_info_one[:THROWN_TILE_INFO_DIMENSION] for
            thrown_info_one in thrown_info] for thrown_info in thrown_infos]
        lengths_all_melds_infos = [[thrown_info_one[THROWN_TILE_INFO_DIMENSION] for
            thrown_info_one in thrown_info] for thrown_info in thrown_infos]
        mask_melds_infos = [[thrown_info_one[THROWN_TILE_INFO_DIMENSION+1:] for
            thrown_info_one in thrown_info] for thrown_info in thrown_infos]

        x_thrown_batch, pos_thrown_batch, pos_mask_batch, thrown_tiles_batch = \
            self.acquire_basic_thrown_data(basic_thrown_infos)
        x_meld_batch, pos_meld_batch, mask_meld_batch, melds_batch = \
            self.acquire_thrown_meld_data(melds_infos, lengths_all_melds_infos, mask_melds_infos)

        melds_batch[mask_meld_batch[pos_mask_batch == 1] == 1] = -1 # remove chakan at this time
        assert(np.all(np.sum(melds_batch >= 0, axis=-1) <= 4))

        x_thrown_batch = torch.FloatTensor(x_thrown_batch).to(device)
        pos_thrown_batch = torch.LongTensor(pos_thrown_batch).to(device)
        pos_mask_batch = torch.ByteTensor(pos_mask_batch).to(device)
        x_meld_batch = torch.FloatTensor(x_meld_batch).to(device)
        pos_meld_batch = torch.LongTensor(pos_meld_batch).to(device)
        mask_meld_batch = torch.ByteTensor(mask_meld_batch).to(device)

        x_round_batch = np.array([mahjong_board.acquire_round_info((iplayer+j)%NUM_PLAYER) for
            j in range(NUM_PLAYER)])
        x_round_batch = x_round_batch[:, :-1] # remove last dimension
        x_remain_batch = np.array([mahjong_board.acquire_remain_info(iplayer)] * NUM_PLAYER) # use my hand
        x_safe_batch = np.array([mahjong_board.acquire_player_furiten_info((iplayer+j)%NUM_PLAYER,
            use_temp_furiten=True) for j in range(NUM_PLAYER)])

        x_round_batch = torch.FloatTensor(x_round_batch).to(device)
        x_remain_batch = torch.FloatTensor(x_remain_batch).to(device)
        x_safe_batch = torch.FloatTensor(x_safe_batch).to(device)

        _, _, logit_tenpai_batch, logits_waiting_batch, \
            ypred_score_batch, ypred_mean_score_batch, logit_win_batch = self.netThrown0(
            x_round_batch, x_meld_batch, pos_meld_batch, mask_meld_batch,
            x_thrown_batch, pos_thrown_batch, x_remain_batch, x_safe_batch)

        _, _, ypred_my_hand_batch, logit_next_use_call_batch, \
            logits_next_call_type_batch, logits_next_thrown_batch, \
            logit_win_tsumo_batch, logit_win_ron_batch, logit_lose_batch = self.netThrown1(
            x_round_batch, x_meld_batch, pos_meld_batch, mask_meld_batch,
            x_thrown_batch, pos_thrown_batch, x_remain_batch, x_safe_batch)

        safe_tile_types_all = x_safe_batch
        logit_tenpai_all = logit_tenpai_batch[pos_mask_batch]
        logits_waiting_all = logits_waiting_batch[pos_mask_batch]
        ypred_score_all = ypred_score_batch[pos_mask_batch]
        ypred_mean_score_all = ypred_mean_score_batch[pos_mask_batch]
        logit_win_all = logit_win_batch[pos_mask_batch]
        ypred_my_hand_all = ypred_my_hand_batch[pos_mask_batch]
        logit_next_use_call_all = logit_next_use_call_batch[pos_mask_batch]
        logits_next_call_type_all = logits_next_call_type_batch[pos_mask_batch]
        logits_next_thrown_all = logits_next_thrown_batch[pos_mask_batch]
        logit_win_tsumo_all = logit_win_tsumo_batch[pos_mask_batch]
        logit_win_ron_all = logit_win_ron_batch[pos_mask_batch]
        logit_lose_all = logit_lose_batch[pos_mask_batch]

        prob_tenpai_all = torch.sigmoid(logit_tenpai_all)
        probs_waiting_all = torch.sigmoid(logits_waiting_all)
        prob_win_all = torch.sigmoid(logit_win_all)
        prob_next_use_call_all = F.sigmoid(logit_next_use_call_all)
        probs_next_call_type_all = F.softmax(logits_next_call_type_all, dim=-1)
        probs_next_thrown_all = F.softmax(logits_next_thrown_all, dim=-1)
        prob_win_tsumo_all = F.sigmoid(logit_win_tsumo_all)
        prob_win_ron_all = F.sigmoid(logit_win_ron_all)
        prob_lose_all = F.sigmoid(logit_lose_all)

        self.safe_tile_types_all = safe_tile_types_all.cpu().detach()
        self.prob_tenpai_all = prob_tenpai_all.cpu().detach()
        self.probs_waiting_all = probs_waiting_all.cpu().detach()
        self.ypred_score_all = ypred_score_all.cpu().detach()
        self.ypred_mean_score_all = ypred_mean_score_all.cpu().detach()
        self.ypred_my_hand_all = ypred_my_hand_all.cpu().detach()
        self.prob_next_use_call_all = prob_next_use_call_all.cpu().detach()
        self.probs_next_call_type_all = probs_next_call_type_all.cpu().detach()
        self.probs_next_thrown_all = probs_next_thrown_all.cpu().detach()
        self.prob_win_tsumo_all = prob_win_tsumo_all.cpu().detach()
        self.prob_win_ron_all = prob_win_ron_all.cpu().detach()
        self.prob_lose_all = prob_lose_all.cpu().detach()

        return thrown_tiles_batch, melds_batch, \
            prob_tenpai_all, probs_waiting_all, \
            ypred_score_all, ypred_mean_score_all, prob_win_all, \
            ypred_my_hand_all, prob_next_use_call_all, \
            probs_next_call_type_all, probs_next_thrown_all, \
            prob_win_tsumo_all, prob_win_ron_all, prob_lose_all

    def acquire_thrown_feature(self, iplayer, mahjong_board):
        device = self.device

        thrown_tiles_all, melds_all, \
            prob_tenpai_all, probs_waiting_all, \
            ypred_score_all, ypred_mean_score_all, prob_win_all, \
            ypred_my_hand_all, prob_next_use_call_all, \
            probs_next_call_type_all, probs_next_thrown_all, \
            prob_win_tsumo_all, prob_win_ron_all, prob_lose_all = self.predict_thrown(iplayer, mahjong_board)

        is_chin_batch = np.array([int((iplayer+1+j)%NUM_PLAYER == mahjong_board.oya) for
            j in range(NUM_PLAYER-1)])
        score_diff_batch = np.array([mahjong_board.player_scores[(iplayer+1+j)%NUM_PLAYER] -
            mahjong_board.player_scores[iplayer] for j in range(NUM_PLAYER-1)]).astype('float32')
        score_diff_batch[score_diff_batch > 0] = np.log(score_diff_batch[score_diff_batch > 0])
        score_diff_batch[score_diff_batch < 0] = -np.log(-score_diff_batch[score_diff_batch < 0])
        is_reach_batch = np.array([int(mahjong_board.players[(iplayer+1+j)%NUM_PLAYER].is_reached) for
            j in range(NUM_PLAYER-1)])
        is_ippatsu_batch = np.array([int(mahjong_board.players[(iplayer+1+j)%NUM_PLAYER].is_ippatsu) for
            j in range(NUM_PLAYER-1)])

        thrown_tile_vector_batch = np.zeros((NUM_PLAYER-1, NUM_TILE_TYPE), dtype='int')
        for j in range(NUM_PLAYER-1):
            for tile_type in thrown_tiles_all[1+j, 1:]: # 0 related to NUM_TILE_TYPE
                if tile_type >= 0:
                    assert(tile_type < NUM_TILE_TYPE)
                    thrown_tile_vector_batch[j, tile_type] += 1
        meld_vector_batch = np.zeros((NUM_PLAYER-1, NUM_MELD_TYPE), dtype='int')
        for j in range(NUM_PLAYER-1):
            for meld in melds_all[1+j]:
                if meld >= 0:
                    meld_vector_batch[j, meld] += 1

        is_chin_batch = torch.FloatTensor(is_chin_batch).to(device).view(-1, 1)
        score_diff_batch = torch.FloatTensor(score_diff_batch).to(device).view(-1, 1)
        is_reach_batch = torch.FloatTensor(is_reach_batch).to(device).view(-1, 1)
        is_ippatsu_batch = torch.FloatTensor(is_ippatsu_batch).to(device).view(-1, 1)
        thrown_tile_vector_batch = torch.FloatTensor(thrown_tile_vector_batch).to(device)
        meld_vector_batch = torch.FloatTensor(meld_vector_batch).to(device)

        thrown_feature_batch = torch.cat([is_chin_batch, score_diff_batch, is_reach_batch, is_ippatsu_batch,
            prob_tenpai_all[1:].view(-1, 1), probs_waiting_all[1:], ypred_score_all[1:],
            ypred_mean_score_all[1:].view(-1, 1), prob_win_all[1:].view(-1, 1),
            ypred_my_hand_all[1:], prob_next_use_call_all[1:].view(-1, 1),
            probs_next_call_type_all[1:], probs_next_thrown_all[1:],
            prob_win_tsumo_all[1:].view(-1, 1), prob_win_ron_all[1:].view(-1, 1), prob_lose_all[1:].view(-1, 1),
            thrown_tile_vector_batch, meld_vector_batch], dim=1)
        assert(thrown_feature_batch.size(0) == 3 and thrown_feature_batch.size(1) == 359)
        thrown_feature_batch = thrown_feature_batch.unsqueeze(0)
        self.thrown_feature_batch = thrown_feature_batch.cpu().detach()
        #self.thrown_feature_batch = thrown_feature_batch

        return thrown_feature_batch

    def predict_hand(self, iplayer, mahjong_board, predict_type=None):
        device = self.device

        basic_hand_infos = [mahjong_board.acquire_player_basic_hand_info(iplayer)]
        mask_hand_infos = [mahjong_board.acquire_player_mask_hand_info(iplayer)]
        melds_infos = [mahjong_board.acquire_player_melds_info(iplayer)]

        x_hand_batch, pos_hand_batch, mask_hand_batch, _ = self.acquire_basic_hand_data(basic_hand_infos, mask_hand_infos)
        x_meld_batch, pos_meld_batch, _ = self.acquire_hand_meld_data(melds_infos)

        x_hand_batch = torch.FloatTensor(x_hand_batch).to(device)
        pos_hand_batch = torch.LongTensor(pos_hand_batch).to(device)
        mask_hand_batch = torch.ByteTensor(mask_hand_batch).to(device)
        x_meld_batch = torch.FloatTensor(x_meld_batch).to(device)
        pos_meld_batch = torch.LongTensor(pos_meld_batch).to(device)

        x_round_batch = np.array([mahjong_board.acquire_round_info(iplayer)])
        x_round_batch = x_round_batch[:, :-1] # remove last dimension
        x_remain_batch = np.array([mahjong_board.acquire_remain_info(iplayer)])
        x_furiten_batch = np.array([mahjong_board.acquire_player_furiten_info(iplayer, use_temp_furiten=False)])

        x_round_batch = torch.FloatTensor(x_round_batch).to(device)
        x_remain_batch = torch.FloatTensor(x_remain_batch).to(device)
        x_furiten_batch = torch.FloatTensor(x_furiten_batch).to(device)

        if self.thrown_feature_batch is None: # When only this function is called
            thrown_feature_batch = self.acquire_thrown_feature(iplayer, mahjong_board)
            self.thrown_feature_batch = None
        else: # When acquire_thrown_feature is called individually
            thrown_feature_batch = self.thrown_feature_batch.to(device)

        #start_time = time.time()
        _, logits_throw_index_batch, logit_tsumo_batch, logits_ron_batch, \
            logits_chi_batch, logit_pon_batch, \
            logit_chakan_batch, logit_daiminkan_batch, logit_ankan_batch, logit_yao9_batch, \
            ypred_mean_score_batch, logit_reach_batch, logits_chied_batch, logits_poned_batch, \
            logit_win_tsumo_batch, logit_win_ron_batch, logit_lose_batch, \
            logits_next_tile_batch, ypred_value_batch = self.netHand(
            x_round_batch, x_meld_batch, pos_meld_batch, x_hand_batch, pos_hand_batch, mask_hand_batch,
            x_remain_batch, x_furiten_batch, thrown_feature_batch)
        #end_time = time.time()
        #elapsed_time = end_time - start_time
        #print('Elapsed time: %f seconds'%(elapsed_time))

        probs_throw_index = F.softmax(logits_throw_index_batch[0], dim=-1).cpu().detach()
        prob_tsumo = F.sigmoid(logit_tsumo_batch[0]).cpu().detach()
        probs_ron = F.sigmoid(logits_ron_batch[0]).cpu().detach()
        probs_chi = F.sigmoid(logits_chi_batch[0]).cpu().detach()
        prob_pon = F.sigmoid(logit_pon_batch[0]).cpu().detach()
        prob_chakan = F.sigmoid(logit_chakan_batch[0]).cpu().detach()
        prob_daiminkan = F.sigmoid(logit_daiminkan_batch[0]).cpu().detach()
        prob_ankan = F.sigmoid(logit_ankan_batch[0]).cpu().detach()
        prob_yao9 = F.sigmoid(logit_yao9_batch[0]).cpu().detach()
        ypred_mean_score = ypred_mean_score_batch[0].cpu().detach()
        prob_reach = F.sigmoid(logit_reach_batch[0]).cpu().detach()
        probs_chied = F.sigmoid(logits_chied_batch[0]).cpu().detach()
        probs_poned = F.sigmoid(logits_poned_batch[0]).cpu().detach()
        prob_win_tsumo = F.sigmoid(logit_win_tsumo_batch[0]).cpu().detach()
        prob_win_ron = F.sigmoid(logit_win_ron_batch[0]).cpu().detach()
        prob_lose = F.sigmoid(logit_lose_batch[0]).cpu().detach()

        self.ypred_mean_score = ypred_mean_score
        self.probs_chied = probs_chied
        self.probs_poned = probs_poned
        self.prob_win_tsumo = prob_win_tsumo
        self.prob_win_ron = prob_win_ron
        self.prob_lose = prob_lose

        if predict_type == 'reach': # process in 'defg'
            pass
        elif predict_type == 'defg':
            return (probs_throw_index.numpy(), prob_reach.numpy()) # 14
        elif predict_type == 'tsumo':
            return prob_tsumo.numpy() # scalar
        elif predict_type == 'ron':
            return probs_ron.numpy() # 14 * 5
        elif predict_type == 'chi':
            return probs_chi.numpy() # 14 * 4
        elif predict_type == 'pon':
            return prob_pon.numpy() # 14
        elif predict_type == 'daiminkan':
            return prob_daiminkan.numpy() # 14
        elif predict_type == 'chakan':
            return prob_chakan.numpy() # 14
        elif predict_type == 'ankan':
            return prob_ankan.numpy() # 14
        elif predict_type == 'yao9':
            return prob_yao9.numpy() # scalar
        else:
            assert(predict_type is None)

    def play_reach(self, iplayer, mahjong_board):
        # prob is already computed when considering defg.
        #if self.is_test: # always reach
        #    return [('reach', iplayer, 1)]
        prob = self._prob_reach
        if self.is_greedy:
            prob = float(prob >= 0.5)
        do_reach = int(np.random.random_sample() < prob)
        if do_reach == 1:
            return [('reach', iplayer, 1)]
        else:
            return []

    def play_defg(self, iplayer, mahjong_board, event):
        hand = mahjong_board.players[iplayer].hand
        if mahjong_board.players[iplayer].is_reached: # should tsumo-giri
            thrown_tile = hand[-1]
            return [('defg', iplayer, thrown_tile, None)]
        probs_defg, prob_reach = self.predict_hand(iplayer, mahjong_board, 'defg')
        if self.is_greedy:
            probs = np.zeros((14,), dtype='float')
            probs[np.argmax(probs_defg)] = 1.
        else:
            probs = probs_defg
        throw_index = np.random.choice(len(probs), p=probs)
        thrown_tile = hand[throw_index]
        self._prob_reach = prob_reach[throw_index]
        return [('defg', iplayer, thrown_tile, None)]

    def play_tsumo(self, iplayer, mahjong_board, event):
        prob = self.predict_hand(iplayer, mahjong_board, 'tsumo')
        if self.is_greedy:
            prob = float(prob >= 0.5)
        do_tsumo = int(np.random.random_sample() < prob)
        if do_tsumo == 1:
            return [event]
        else:
            return []

    def play_ron(self, iplayer, mahjong_board, event):
        hand = mahjong_board.players[iplayer].hand
        hand_array = np.array(hand)
        agari_tile = mahjong_board.agari_tile
        probs_init = self.predict_hand(iplayer, mahjong_board, 'ron') # 14 * 5
        prob = 1.
        if np.sum(hand_array//4 == agari_tile//4) > 0:
            prob = min(prob, np.min(probs_init[hand_array//4 == agari_tile//4, 2]))
        if agari_tile//4 < 27:
            if (agari_tile//4)%9 >= 1 and np.sum(hand_array//4 == agari_tile//4-1) > 0:
                prob = min(prob, np.min(probs_init[hand_array//4 == agari_tile//4-1, 3]))
            if (agari_tile//4)%9 >= 2 and np.sum(hand_array//4 == agari_tile//4-2) > 0:
                prob = min(prob, np.min(probs_init[hand_array//4 == agari_tile//4-2, 4]))
            if (agari_tile//4)%9 <= 7 and np.sum(hand_array//4 == agari_tile//4+1) > 0:
                prob = min(prob, np.min(probs_init[hand_array//4 == agari_tile//4+1, 1]))
            if (agari_tile//4)%9 <= 6 and np.sum(hand_array//4 == agari_tile//4+2) > 0:
                prob = min(prob, np.min(probs_init[hand_array//4 == agari_tile//4+2, 0]))
        if self.is_greedy:
            prob = float(prob >= 0.5)
        do_ron = int(np.random.random_sample() < prob)
        if do_ron == 1:
            return [event]
        else:
            return []

    def play_chi(self, iplayer, mahjong_board, event):
        def decide_to_chi(chi_type, probs):
            if self.is_greedy:
                #if self.is_test:
                #    pass
                #else:
                probs = (probs >= 0.5).astype('int') # greedy
            chi_flag = False
            #chi_flag = True
            for prob in probs[chi_type-1:chi_type+1]:
                #if np.random.random_sample() < 1. - prob:
                if np.random.random_sample() < prob:
                    chi_flag = True
                    #chi_flag = False
                    break
            if not chi_flag:
                return 0
            else:
                return chi_type
        def compete_two_chi_types(chi_type_1, chi_type_2, probs):
            prob_1 = probs[chi_type_1-1] * probs[chi_type_1]
            prob_2 = probs[chi_type_2-1] * probs[chi_type_2]
            if prob_1 >= prob_2:
                return decide_to_chi(chi_type_1, probs)
            else:
                return decide_to_chi(chi_type_2, probs)

        hand = mahjong_board.players[iplayer].hand
        hand_array = np.array(hand)
        chi_availables = event[3]
        call_tile = chi_availables[0][2][chi_availables[0][1]]

        probs_init = self.predict_hand(iplayer, mahjong_board, 'chi') # 14 * 4
        type_availables = [False] * 3
        probs = np.array([0.] * 4)
        for (meld_from_who, meld_called, meld_tiles) in chi_availables:
            type_availables[meld_called] = True
        chi_univs = (hand_array//4 == call_tile//4+2, hand_array//4 == call_tile//4+1,
            hand_array//4 == call_tile//4-1, hand_array//4 == call_tile//4-2)
        if type_availables[0]:
            assert(np.sum(chi_univs[0]) > 0)
            probs[0] = np.max(probs_init[chi_univs[0], 0])
        if type_availables[0] or type_availables[1]:
            assert(np.sum(chi_univs[1]) > 0)
            probs[1] = np.max(probs_init[chi_univs[1], 1])
        if type_availables[1] or type_availables[2]:
            assert(np.sum(chi_univs[2]) > 0)
            probs[2] = np.max(probs_init[chi_univs[2], 2])
        if type_availables[2]:
            assert(np.sum(chi_univs[3]) > 0)
            probs[3] = np.max(probs_init[chi_univs[3], 3])
        type_available_num = np.sum(type_availables)
        if type_available_num == 1:
            chi_type = 1 + np.argmax(type_availables)
            do_chi_pred = decide_to_chi(chi_type, probs)
        elif type_available_num == 2:
            assert(type_availables[1])
            chi_type_1 = 2
            if type_availables[0]:
                chi_type_2 = 1
            else:
                assert(type_availables[2])
                chi_type_2 = 3
            do_chi_pred = compete_two_chi_types(chi_type_1, chi_type_2, probs)
        else:
            assert(type_available_num == 3)
            if probs[0] < 0.5 and probs[0] <= probs[3]:
                do_chi_pred = compete_two_chi_types(2, 3, probs)
            elif probs[3] < 0.5 and probs[3] <= probs[0]:
                do_chi_pred = compete_two_chi_types(2, 1, probs)
            else:
                assert(probs[0] >= 0.5 and probs[3] >= 0.5)
                do_chi_pred = compete_two_chi_types(1, 3, probs)
        chi_index = -1
        if do_chi_pred >= 1:
            tile1 = hand_array[chi_univs[do_chi_pred-1]][np.argmax(probs_init[chi_univs[do_chi_pred-1], do_chi_pred-1])]
            tile2 = hand_array[chi_univs[do_chi_pred]][np.argmax(probs_init[chi_univs[do_chi_pred], do_chi_pred])]
            for ichi, (meld_from_who, meld_called, meld_tiles) in enumerate(chi_availables):
                if set(meld_tiles) == set([tile1, tile2, call_tile]):
                    chi_index = ichi
                    break
            assert(chi_index >= 0)
        if chi_index == -1:
            return []
        else:
            return [('n', 'chi', iplayer, chi_availables[chi_index])]

    def play_pon(self, iplayer, mahjong_board, event):
        hand = mahjong_board.players[iplayer].hand
        hand_array = np.array(hand)
        pon_availables = event[3]
        call_tile = pon_availables[0][2][pon_availables[0][1]]

        prob_init = self.predict_hand(iplayer, mahjong_board, 'pon') # 14
        pon_univ = hand_array//4 == call_tile//4
        assert(np.sum(pon_univ) >= 2)
        prob = np.max(prob_init[pon_univ])
        if self.is_greedy:
            #if self.is_test:
            prob = float(prob >= 0.377)
            #else:
            #    prob = float(prob >= 0.5)
        do_pon = int(np.random.random_sample() < prob)
        pon_index = -1
        if do_pon == 1:
            argsort_tiles = hand_array[pon_univ][np.argsort(prob_init[pon_univ])]
            tile1 = argsort_tiles[-1]
            tile2 = argsort_tiles[-2]
            for ipon, (meld_from_who, meld_called, meld_tiles) in enumerate(pon_availables):
                if set(meld_tiles) == set([tile1, tile2, call_tile]):
                    pon_index = ipon
                    break
            assert(pon_index >= 0)
        if pon_index == -1:
            return []
        else:
            return [('n', 'pon', iplayer, pon_availables[pon_index])]

    def play_daiminkan(self, iplayer, mahjong_board, event):
        hand = mahjong_board.players[iplayer].hand
        hand_array = np.array(hand)
        daiminkan_availables = event[3]
        assert(len(daiminkan_availables) == 1)
        call_tile = daiminkan_availables[0][2][daiminkan_availables[0][1]]

        prob_init = self.predict_hand(iplayer, mahjong_board, 'daiminkan') # 14
        daiminkan_univ = hand_array//4 == call_tile//4
        assert(np.sum(daiminkan_univ) == 3)
        prob = np.max(prob_init[daiminkan_univ])
        if self.is_greedy:
            prob = float(prob >= 0.5)
        do_daiminkan = int(np.random.random_sample() < prob)

        if do_daiminkan == 0:
            return []
        else:
            return [('n', 'daiminkan', iplayer, daiminkan_availables[0])]

    def play_chakan(self, iplayer, mahjong_board, event):
        hand = mahjong_board.players[iplayer].hand
        chakan_availables = event[3]

        prob_init = self.predict_hand(iplayer, mahjong_board, 'chakan') # 14
        chakan_index = -1
        for ichakan, (meld_from_who, meld_called, meld_tiles) in enumerate(chakan_availables):
            prob = prob_init[hand.index(meld_tiles[-1])]
            if self.is_greedy:
                prob = float(prob >= 0.5)
            do_chakan = int(np.random.random_sample() < prob)
            if do_chakan:
                chakan_index = ichakan
                break
        if chakan_index == -1:
            return []
        else:
            return [('n', 'chakan', iplayer, chakan_availables[chakan_index])]

    def play_ankan(self, iplayer, mahjong_board, event):
        hand = mahjong_board.players[iplayer].hand
        hand_array = np.array(hand)
        ankan_availables = event[3]

        prob_init = self.predict_hand(iplayer, mahjong_board, 'ankan') # 14
        ankan_index = -1
        for iankan, (meld_from_who, meld_called, meld_tiles) in enumerate(ankan_availables):
            ankan_univ = hand_array//4 == meld_tiles[0]//4
            assert(np.sum(ankan_univ) == 4)
            prob = np.max(prob_init[ankan_univ])
            if self.is_greedy:
                prob = float(prob >= 0.377)
            do_ankan = int(np.random.random_sample() < prob)
            if do_ankan:
                ankan_index = iankan
                break
        if ankan_index == -1:
            return []
        else:
            return [('n', 'ankan', iplayer, ankan_availables[ankan_index])]

    def play_yao9(self, iplayer, mahjong_board, event):
        prob = self.predict_hand(iplayer, mahjong_board, 'yao9')
        if self.is_greedy:
            prob = float(prob >= 0.5)
        do_yao9 = int(np.random.random_sample() < prob)
        if do_yao9 == 1:
            return [('ryuukyoku', 'yao9')]
        else:
            return []




class MCTSDeepAIPlayer(BaseMahjongPlayer):
    def __init__(self, device, netHand, netThrown0, netThrown1, netRank, is_greedy=True, do_meld=True,
        simulation_num=100, max_defg_count=3, max_branch_num=5, cpuct=1., num_thread=1, max_time_for_search = 2.,
        seed=1):
        super(MCTSDeepAIPlayer, self).__init__(do_meld)
        self.device = device
        self.netHand = netHand
        self.netThrown0 = netThrown0
        self.netThrown1 = netThrown1
        self.netRank = netRank

        self.is_greedy = is_greedy
        self.simulation_num = simulation_num
        self.max_defg_count = max_defg_count
        self.max_branch_num = max_branch_num
        self.cpuct = cpuct
        self.num_thread = num_thread
        self.max_time_for_search = max_time_for_search
        self.seed = seed

        self._prob_reach = None
        self._thrown_tile = None

        #self.netHand.eval()
        #self.netThrown0.eval()
        #self.netThrown1.eval()
        #self.netRank.eval()
        #torch.set_grad_enabled(False)

        self.sub_player = SimpleDeepAIPlayer(device, netHand, netThrown0, netThrown1, is_greedy=True, do_meld=True)
        #self.thread_sub_players = []
        self.thread_netThrown0s = []
        self.thread_netThrown1s = []
        self.thread_netHands = []
        self.thread_netRanks = []
        for ithread in range(num_thread):
            thread_netThrown0 = netThrown0
            thread_netThrown1 = netThrown1
            thread_netHand = netHand
            thread_netRank = netRank
            #thread_netThrown0 = ThrownModel(0)
            #thread_netThrown0.load_state_dict(netThrown0.state_dict())
            #thread_netThrown1 = ThrownModel(1)
            #thread_netThrown1.load_state_dict(netThrown1.state_dict())
            #thread_netHand = HandModel()
            #thread_netHand.load_state_dict(netHand.state_dict())
            #thread_netRank = RankModel()
            #thread_netRank.load_state_dict(netRank.state_dict())
            #thread_netThrown0.to(device)
            #thread_netThrown1.to(device)
            #thread_netHand.to(device)
            #thread_netRank.to(device)
            #thread_netThrown0.eval()
            #thread_netThrown1.eval()
            #thread_netHand.eval()
            #thread_netRank.eval()
            #torch.set_grad_enabled(False)
            #thread_netThrown0.share_memory()
            #thread_netThrown1.share_memory()
            #thread_netHand.share_memory()
            #thread_netRank.share_memory()
            #thread_sub_player = SimpleDeepAIPlayer(device, thread_netHand,
            #    thread_netThrown0, thread_netThrown1, is_greedy=True, do_meld=True)
            #self.thread_sub_players.append(thread_sub_player)
            self.thread_netThrown0s.append(thread_netThrown0)
            self.thread_netThrown1s.append(thread_netThrown1)
            self.thread_netHands.append(thread_netHand)
            self.thread_netRanks.append(thread_netRank)

    def reset_local_variables(self):
        self.sub_player.reset_local_variables()
        #for thread_sub_player in self.thread_sub_players:
        #    thread_sub_player.reset_local_variables()

    def get_action_prob_thread(self, iplayer, mahjong_board, events, probs):
        self.sub_player.share_local_variables()

        nsa1 = np.zeros((len(probs),), dtype='float32') # for search
        nsa2 = np.zeros((len(probs),), dtype='float32') # for computing qsa
        qsa = np.zeros((len(probs),), dtype='float32')
        if len(probs) > self.max_branch_num:
            probs[np.argsort(probs)[:-self.max_branch_num]] = 0.
        valids = np.array(probs) > 0.

        start_time = time.time()

        thrown_feature_batch = self.sub_player.thrown_feature_batch
        safe_tile_types_all = self.sub_player.safe_tile_types_all.numpy()
        prob_tenpai_all = self.sub_player.prob_tenpai_all.numpy()
        probs_waiting_all = self.sub_player.probs_waiting_all.numpy()
        ypred_score_all = self.sub_player.ypred_score_all.numpy()
        ypred_mean_score_all = self.sub_player.ypred_mean_score_all.numpy()
        ypred_my_hand_all = self.sub_player.ypred_my_hand_all.numpy()
        prob_next_use_call_all = self.sub_player.prob_next_use_call_all.numpy()
        probs_next_call_type_all = self.sub_player.probs_next_call_type_all.numpy()
        probs_next_thrown_all = self.sub_player.probs_next_thrown_all.numpy()
        prob_win_tsumo_all = self.sub_player.prob_win_tsumo_all.numpy()
        prob_win_ron_all = self.sub_player.prob_win_ron_all.numpy()
        prob_lose_all = self.sub_player.prob_lose_all.numpy()
        probs_chied = self.sub_player.probs_chied.numpy()
        probs_poned = self.sub_player.probs_poned.numpy()

        sub_player_infos = (thrown_feature_batch, safe_tile_types_all, prob_tenpai_all, probs_waiting_all,
            ypred_score_all, ypred_mean_score_all, ypred_my_hand_all, prob_next_use_call_all,
            probs_next_call_type_all, probs_next_thrown_all, prob_win_tsumo_all, prob_win_ron_all,
            prob_lose_all, probs_chied, probs_poned)

        nsa1 = mp.Array('d', nsa1)
        nsa2 = mp.Array('d', nsa2)
        qsa = mp.Array('d', qsa)
        n = mp.Value('d', 0)
        lock = mp.Lock()
        processes = []
        for ithread in range(self.num_thread):
            p = mp.Process(target=self.get_action_prob_sub, args=(iplayer, mahjong_board,
                events, probs, n, nsa1, nsa2, qsa, valids, start_time,
                self.thread_netHands[ithread], self.thread_netThrown0s[ithread],
                self.thread_netThrown1s[ithread], self.thread_netRanks[ithread],
                sub_player_infos, lock, ithread))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        nsa1 = np.frombuffer(nsa1.get_obj())
        nsa2 = np.frombuffer(nsa2.get_obj())
        qsa = np.frombuffer(qsa.get_obj())
        n = n.value
        print(nsa1, nsa2, qsa, n)
        assert(0==1)

        counts = nsa2
        counts = counts.astype('float32')  # ** (1./temp) # TODO: Lp-norm
        probs_mtcs = counts / np.sum(counts)
        return probs_mtcs


    def get_action_prob_sub(self, iplayer, mahjong_board, events, probs,
        n, nsa1, nsa2, qsa, valids, start_time,
        netHand, netThrown0, netThrown1, netRank, sub_player_infos,
        lock, ithread):
        seed = self.seed + ithread * 1000
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        cpuct = self.cpuct
        max_time_for_search = self.max_time_for_search
        eps = 1e-8
        while(True):
            with lock:
                cur_best = -float('inf')
                best_act = -1
                # pick the action with the highest upper confidence bound
                for a in range(len(events)):
                    if valids[a]:
                        if a in qsa:
                            u = qsa[a] + cpuct * probs[a] * np.sqrt(n.value) / (1. + nsa1[a])
                        else:
                            u = cpuct * probs[a] * np.sqrt(n.value + eps)
                        if u > cur_best:
                            cur_best = u
                            best_act = a
                n.value += 1
                nsa1[best_act] += 1 # nsa1 is updated here
            v = self.virtual_compete(iplayer, mahjong_board, events[best_act],
                netHand, netThrown0, netThrown1, netRank, sub_player_infos, ithread)
            with lock:
                qsa[best_act] = (nsa2[best_act] * qsa[best_act] + v) / (nsa2[best_act] + 1.)
                nsa2[best_act] += 1 # nsa2 is updated here

            end_time = time.time()
            elapsed_time = end_time - start_time
            if elapsed_time > max_time_for_search:
                break

    def get_action_prob(self, iplayer, mahjong_board, events, probs):
        start_time = time.time()
        eps = 1e-8
        nsa = np.zeros((len(probs),), dtype='int')
        qsa = defaultdict(float)
        if len(probs) > self.max_branch_num:
            probs[np.argsort(probs)[:-self.max_branch_num]] = 0.
        valids = np.array(probs) > 0.

        thrown_feature_batch = self.sub_player.thrown_feature_batch
        safe_tile_types_all = self.sub_player.safe_tile_types_all.numpy()
        prob_tenpai_all = self.sub_player.prob_tenpai_all.numpy()
        probs_waiting_all = self.sub_player.probs_waiting_all.numpy()
        ypred_score_all = self.sub_player.ypred_score_all.numpy()
        ypred_mean_score_all = self.sub_player.ypred_mean_score_all.numpy()
        ypred_my_hand_all = self.sub_player.ypred_my_hand_all.numpy()
        prob_next_use_call_all = self.sub_player.prob_next_use_call_all.numpy()
        probs_next_call_type_all = self.sub_player.probs_next_call_type_all.numpy()
        probs_next_thrown_all = self.sub_player.probs_next_thrown_all.numpy()
        prob_win_tsumo_all = self.sub_player.prob_win_tsumo_all.numpy()
        prob_win_ron_all = self.sub_player.prob_win_ron_all.numpy()
        prob_lose_all = self.sub_player.prob_lose_all.numpy()
        probs_chied = self.sub_player.probs_chied.numpy()
        probs_poned = self.sub_player.probs_poned.numpy()

        sub_player_infos = (thrown_feature_batch, safe_tile_types_all, prob_tenpai_all, probs_waiting_all,
            ypred_score_all, ypred_mean_score_all, ypred_my_hand_all, prob_next_use_call_all,
            probs_next_call_type_all, probs_next_thrown_all, prob_win_tsumo_all, prob_win_ron_all,
            prob_lose_all, probs_chied, probs_poned)

        for n in range(self.simulation_num):
            cur_best = -float('inf')
            best_act = -1
            # pick the action with the highest upper confidence bound
            for a in range(len(events)):
                if valids[a]:
                    if a in qsa:
                        u = qsa[a] + self.cpuct * probs[a] * np.sqrt(n) / (1. + nsa[a])
                    else:
                        u = self.cpuct * probs[a] * np.sqrt(n + eps)
                    if u > cur_best:
                        cur_best = u
                        best_act = a
            v = self.virtual_compete(iplayer, mahjong_board, events[best_act],
                self.netHand, self.netThrown0, self.netThrown1, self.netRank, sub_player_infos, 0)
            qsa[best_act] = (nsa[best_act] * qsa[best_act] + v) / (nsa[best_act] + 1.)
            nsa[best_act] += 1

        counts = nsa
        counts = counts.astype('float32')  # ** (1./temp) # TODO: Lp-norm
        probs_mtcs = counts / np.sum(counts)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print('Elapsed time: %d seconds.'%(elapsed_time))
        assert(0==1)

        return probs_mtcs

    def virtual_compete(self, iplayer, mahjong_board, events,
        netHand, netThrown0, netThrown1, netRank, sub_player_infos, ithread):
        device = self.device
        #netThrown0 = self.netThrown0
        #netThrown1 = self.netThrown1
        #netHand = self.netHand
        #temp_netRank = self.netRank

        #thread_netThrown0 = ThrownModel(0)
        #thread_netThrown0.load_state_dict(netThrown0.state_dict())
        #thread_netThrown1 = ThrownModel(1)
        #thread_netThrown1.load_state_dict(netThrown1.state_dict())
        #thread_netHand = HandModel()
        #thread_netHand.load_state_dict(netHand.state_dict())
        #thread_netRank = RankModel()
        #thread_netRank.load_state_dict(netRank.state_dict())
        #thread_netThrown0.to(device)
        #thread_netThrown1.to(device)
        #thread_netHand.to(device)
        #thread_netRank.to(device)
        #thread_netThrown0.eval()
        #thread_netThrown1.eval()
        #thread_netHand.eval()
        #thread_netRank.eval()
        #temp_sub_player = SimpleDeepAIPlayer(device, thread_netHand,
        #    thread_netThrown0, thread_netThrown1, is_greedy=True, do_meld=True)

        temp_netRank = netRank
        temp_sub_player = SimpleDeepAIPlayer(device, netHand, netThrown0, netThrown1, is_greedy=True, do_meld=True)

        #temp_netRank = self.thread_netRanks[ithread]
        #temp_sub_player = self.thread_sub_players[ithread]

        #device = torch.device('cpu')
        rank_value = np.array([0.7, 0.3, 0., -1.])

        virtual_mahjong_game = MahjongGame(mahjong_board.game_type, 'virtual_play')
        virtual_mahjong_board = deepcopy(mahjong_board)
        virtual_mahjong_board.blind_board(iplayer, play_mode='virtual_play')

        max_defg_count = self.max_defg_count

        thrown_feature_batch, safe_tile_types_all, prob_tenpai_all, probs_waiting_all, \
            ypred_score_all, ypred_mean_score_all, ypred_my_hand_all, prob_next_use_call_all, \
            probs_next_call_type_all, probs_next_thrown_all, prob_win_tsumo_all, prob_win_ron_all, \
            prob_lose_all, probs_chied, probs_poned = sub_player_infos

        #assert(self.sub_player.thrown_feature_batch is not None)
        #temp_sub_player.thrown_feature_batch = self.sub_player.thrown_feature_batch
        temp_sub_player.thrown_feature_batch = torch.FloatTensor(thrown_feature_batch).to(device)

        temp_virtual_player = VirtualMahjongPlayer()

        #safe_tile_types_all = np.copy(self.sub_player.safe_tile_types_all.numpy())
        #prob_tenpai_all = np.copy(self.sub_player.prob_tenpai_all.numpy())
        #probs_waiting_all = np.copy(self.sub_player.probs_waiting_all.numpy())
        #ypred_score_all = np.copy(self.sub_player.ypred_score_all.numpy())
        #ypred_mean_score_all = np.copy(self.sub_player.ypred_mean_score_all.numpy())
        #ypred_my_hand_all = np.copy(self.sub_player.ypred_my_hand_all.numpy())
        #prob_next_use_call_all = np.copy(self.sub_player.prob_next_use_call_all.numpy())
        #probs_next_call_type_all = np.copy(self.sub_player.probs_next_call_type_all.numpy())
        #probs_next_thrown_all = np.copy(self.sub_player.probs_next_thrown_all.numpy())
        #prob_win_tsumo_all = np.copy(self.sub_player.prob_win_tsumo_all.numpy())
        #prob_win_ron_all = np.copy(self.sub_player.prob_win_ron_all.numpy())
        #prob_lose_all = np.copy(self.sub_player.prob_lose_all.numpy())
        #probs_chied = np.copy(self.sub_player.probs_chied.numpy())
        #probs_poned = np.copy(self.sub_player.probs_poned.numpy())

        probs_chied_one = None
        probs_poned_one = None

        defg_count = 0
        if events[-1][0] == 'defg':
            defg_count += 1
            last_thrown_index = virtual_mahjong_board.players[iplayer].hand.index(events[-1][2])
            probs_chied_one = probs_chied[last_thrown_index]
            probs_poned_one = probs_poned[last_thrown_index]

        remain_tile_type_nums = np.array(virtual_mahjong_board.acquire_remain_info(iplayer))
        virtual_mahjong_game.set_virtual_params(iplayer, remain_tile_type_nums, prob_tenpai_all, ypred_my_hand_all)
        current_player, virtual_mahjong_board, event = virtual_mahjong_game.get_next_state(virtual_mahjong_board, events)
        #print(events, event)

        while current_player != -2 and defg_count < max_defg_count and event[0] != 'init':
            if len(events) > 0 and events[0][0] in ['n', 'reach']: # recompute features
                temp_sub_player.acquire_thrown_feature(iplayer, virtual_mahjong_board)

                safe_tile_types_all = temp_sub_player.safe_tile_types_all.numpy()
                prob_tenpai_all = temp_sub_player.prob_tenpai_all.numpy()
                probs_waiting_all = temp_sub_player.probs_waiting_all.numpy()
                ypred_score_all = temp_sub_player.ypred_score_all.numpy()
                ypred_mean_score_all = temp_sub_player.ypred_mean_score_all.numpy()
                ypred_my_hand_all = temp_sub_player.ypred_my_hand_all.numpy()
                prob_next_use_call_all = temp_sub_player.prob_next_use_call_all.numpy()
                probs_next_call_type_all = temp_sub_player.probs_next_call_type_all.numpy()
                probs_next_thrown_all = temp_sub_player.probs_next_thrown_all.numpy()
                prob_win_tsumo_all = temp_sub_player.prob_win_tsumo_all.numpy()
                prob_win_ron_all = temp_sub_player.prob_win_ron_all.numpy()
                prob_lose_all = temp_sub_player.prob_lose_all.numpy()

            if event[0] == 'reach' and event[2] == 1:  # skip the event
                events = []
            elif current_player == -1:
                events = [event]
            else:
                if current_player == iplayer:
                    assert(temp_sub_player.thrown_feature_batch is not None)
                    events = temp_sub_player.play(current_player, virtual_mahjong_board, event) # use existing feature
                    if event[0] == 'defg':
                        defg_count += 1
                        if virtual_mahjong_board.players[iplayer].is_reached: # No prediction, repredict
                            temp_sub_player.predict_hand(iplayer, virtual_mahjong_board)
                        last_thrown_index = virtual_mahjong_board.players[iplayer].hand.index(events[-1][2])
                        probs_chied_one = temp_sub_player.probs_chied.numpy()[last_thrown_index]
                        probs_poned_one = temp_sub_player.probs_poned.numpy()[last_thrown_index]
                else:
                    assert (event[0] in ['n', 'defg', 'agari'])
                    remain_tile_type_nums = np.array(virtual_mahjong_board.acquire_remain_info(iplayer))
                    temp_virtual_player.set_virtual_params(iplayer, prob_tenpai_all, probs_waiting_all, ypred_score_all,
                        prob_next_use_call_all, probs_next_call_type_all, probs_next_thrown_all,
                        probs_chied_one, probs_poned_one, safe_tile_types_all, remain_tile_type_nums)
                    events = temp_virtual_player.play(current_player, virtual_mahjong_board, event)

            remain_tile_type_nums = np.array(virtual_mahjong_board.acquire_remain_info(iplayer))
            virtual_mahjong_game.set_virtual_params(iplayer, remain_tile_type_nums, prob_tenpai_all, ypred_my_hand_all)
            current_player, virtual_mahjong_board, event = virtual_mahjong_game.get_next_state(virtual_mahjong_board, events)
            #print(events, event)

            #print('HELLO3', ithread, current_player, defg_count, events, event)

        if current_player == -2:  # owari points to owari rank
            owari_ranks = np.argsort(np.argsort(-virtual_mahjong_board.owari_points))
            value = rank_value[owari_ranks[iplayer]]

        elif event[0] == 'init':  # score to owari class
            player_scores = list(virtual_mahjong_board.player_scores)
            player_scores = player_scores[iplayer:] + player_scores[:iplayer] # iplayer at first
            logits_owari_class = temp_netRank(torch.FloatTensor(player_scores + [virtual_mahjong_board.num_round,
                (virtual_mahjong_board.oya-iplayer)%NUM_PLAYER, virtual_mahjong_board.current_round,
                virtual_mahjong_board.combo, virtual_mahjong_board.reach_stick]).to(device).unsqueeze(0))[0]
            probs_owari_class = F.softmax(logits_owari_class, dim=-1).cpu().detach().numpy()
            value = np.sum(probs_owari_class * rank_value)

        else: # predicted score to owari class
            temp_sub_player.predict_hand(iplayer, virtual_mahjong_board)

            ypred_mean_score = temp_sub_player.ypred_mean_score.numpy()
            prob_win_tsumo = temp_sub_player.prob_win_tsumo.numpy()
            prob_win_ron = temp_sub_player.prob_win_ron.numpy()
            prob_lose = temp_sub_player.prob_lose.numpy()

            basic_score = np.exp(ypred_mean_score) / 4.
            if ypred_mean_score < np.log(10.) - 0.1:
                basic_score = 0.

            if virtual_mahjong_board.check_player_is_tenpai(iplayer):
                waiting_score_np = np.array(virtual_mahjong_board.players[iplayer].waiting_score)
                if np.all(waiting_score_np == 0):
                    basic_score = 0.
                else:
                    basic_score = np.mean(waiting_score_np[waiting_score_np > 0]) / 4.
                gariten_flag = True
                for tile_type in virtual_mahjong_board.players[iplayer].waiting_type:
                    if remain_tile_type_nums[tile_type] > 0:
                        gariten_flag = False
                        break
                if gariten_flag:
                    basic_score = 0.

            if basic_score == 0.:
                prob_win_tsumo = 0.
                prob_win_ron = 0.

            if virtual_mahjong_board.players[iplayer].is_furiten:
                prob_win_ron = 0.

            probs = np.zeros((NUM_PLAYER, NUM_PLAYER), dtype='float32')

            probs[0][0] = prob_win_tsumo
            probs[0][1] = prob_win_ron * prob_lose_all[1] / np.sum(prob_lose_all[1:])
            probs[0][2] = prob_win_ron * prob_lose_all[2] / np.sum(prob_lose_all[1:])
            probs[0][3] = prob_win_ron * prob_lose_all[3] / np.sum(prob_lose_all[1:])

            probs[1][0] = prob_lose * prob_win_ron_all[1] / np.sum(prob_win_ron_all[1:])
            probs[1][1] = prob_win_tsumo_all[1]
            probs[1][2] = max((prob_win_ron_all[1] - probs[1][0]) / 2., 0.)
            probs[1][3] = max((prob_win_ron_all[1] - probs[1][0]) / 2., 0.)

            probs[2][0] = prob_lose * prob_win_ron_all[2] / np.sum(prob_win_ron_all[1:])
            probs[2][1] = max((prob_win_ron_all[2] - probs[2][0]) / 2., 0.)
            probs[2][2] = prob_win_tsumo_all[2]
            probs[2][3] = max((prob_win_ron_all[2] - probs[2][0]) / 2., 0.)

            probs[3][0] = prob_lose * prob_win_ron_all[3] / np.sum(prob_win_ron_all[1:])
            probs[3][1] = max((prob_win_ron_all[3] - probs[3][0]) / 2., 0.)
            probs[3][2] = max((prob_win_ron_all[3] - probs[3][0]) / 2., 0.)
            probs[3][3] = prob_win_tsumo_all[3]
            #print('prob:', np.sum(probs))

            if np.sum(probs) > 1.:
                probs = probs / np.sum(probs)

            basic_scores = np.array([basic_score] + [np.exp(ypred_mean_score_all[j])/4. for j in range(1, NUM_PLAYER)])

            value = 0.
            for j in range(NUM_PLAYER):
                for k in range(NUM_PLAYER):
                    jplayer = (iplayer+j)%NUM_PLAYER
                    kplayer = (iplayer+k)%NUM_PLAYER

                    round_scores = virtual_mahjong_board.compute_score(jplayer, kplayer, basic_scores[j], use_sticks=True)
                    virtual_mahjong_board.player_scores = virtual_mahjong_board.player_scores + round_scores

                    oya, current_round, combo = (virtual_mahjong_board.oya,
                        virtual_mahjong_board.current_round, virtual_mahjong_board.combo)
                    last_chin = oya
                    if jplayer == virtual_mahjong_board.oya:
                        combo += 1
                    else:
                        oya = (oya+1)%NUM_PLAYER
                        current_round += 1


                    if ((current_round >= virtual_mahjong_board.max_num_round) or
                        (current_round >= virtual_mahjong_board.num_round and
                            np.max(virtual_mahjong_board.player_scores) >= virtual_mahjong_board.target_score and
                            (combo == 0 or np.argmax(virtual_mahjong_board.player_scores) == last_chin)) or
                        (current_round == virtual_mahjong_board.num_round-1 and
                            np.max(virtual_mahjong_board.player_scores) >= virtual_mahjong_board.target_score and
                            (combo >= 1 and np.argmax(virtual_mahjong_board.player_scores) == last_chin)) or
                        (np.min(virtual_mahjong_board.player_scores) < 0.)):
                        virtual_mahjong_board.process_owari()
                        owari_ranks = np.argsort(np.argsort(-virtual_mahjong_board.owari_points))
                        value += probs[j][k] * rank_value[owari_ranks[iplayer]]
                    else:
                        player_scores = list(virtual_mahjong_board.player_scores)
                        player_scores = player_scores[iplayer:] + player_scores[:iplayer] # iplayer at first
                        logits_owari_class = temp_netRank(torch.FloatTensor(player_scores + [virtual_mahjong_board.num_round,
                            (oya-iplayer)%NUM_PLAYER, current_round, combo, 0]).to(device).unsqueeze(0))[0]
                        probs_owari_class = F.softmax(logits_owari_class, dim=-1).cpu().detach().numpy()
                        value += probs[j][k] * np.sum(probs_owari_class * rank_value)

                    virtual_mahjong_board.player_scores = virtual_mahjong_board.player_scores - round_scores

            # ryuukyoku
            tenpai_players = virtual_mahjong_game.compute_virtual_tenpai_players(virtual_mahjong_board)
            chin_is_tenpai = virtual_mahjong_board.oya in tenpai_players
            round_scores = virtual_mahjong_board.compute_ryuukyoku_scores(tenpai_players)
            virtual_mahjong_board.player_scores = virtual_mahjong_board.player_scores + round_scores

            oya, current_round, combo = (virtual_mahjong_board.oya,
                virtual_mahjong_board.current_round, virtual_mahjong_board.combo)
            last_chin = oya
            combo += 1
            if not chin_is_tenpai:
                oya = (oya+1)%NUM_PLAYER
                current_round += 1

            if ((current_round >= virtual_mahjong_board.max_num_round) or
                (current_round >= virtual_mahjong_board.num_round and
                    ((np.max(virtual_mahjong_board.player_scores) >= virtual_mahjong_board.target_score and not chin_is_tenpai) or
                    (np.max(virtual_mahjong_board.player_scores) >= virtual_mahjong_board.target_score and chin_is_tenpai and
                    np.argmax(virtual_mahjong_board.player_scores) == last_chin))) or
                (current_round == virtual_mahjong_board.num_round-1 and
                    (np.max(virtual_mahjong_board.player_scores) >= virtual_mahjong_board.target_score and chin_is_tenpai and
                    np.argmax(virtual_mahjong_board.player_scores) == last_chin)) or
                (np.min(virtual_mahjong_board.player_scores) < 0)):
                virtual_mahjong_board.process_owari()
                owari_ranks = np.argsort(np.argsort(-virtual_mahjong_board.owari_points))
                value += (1.-np.sum(probs)) * rank_value[owari_ranks[iplayer]]
            else:
                player_scores = list(virtual_mahjong_board.player_scores)
                player_scores = player_scores[iplayer:] + player_scores[:iplayer] # iplayer at first
                logits_owari_class = temp_netRank(torch.FloatTensor(player_scores + [virtual_mahjong_board.num_round,
                    (oya-iplayer)%NUM_PLAYER, current_round, combo,
                    virtual_mahjong_board.reach_stick]).to(device).unsqueeze(0))[0]
                probs_owari_class = F.softmax(logits_owari_class, dim=-1).cpu().detach().numpy()
                value += (1.-np.sum(probs)) * np.sum(probs_owari_class * rank_value)

            virtual_mahjong_board.player_scores = virtual_mahjong_board.player_scores - round_scores

        del virtual_mahjong_game
        del virtual_mahjong_board
        del temp_virtual_player

        #print('value:', value)

        return value


    def play_reach(self, iplayer, mahjong_board):
        # prob is already computed when considering defg.
        #if self.is_test: # always reach
        #    return [('reach', iplayer, 1)]
        prob = self._prob_reach
        thrown_tile = self._thrown_tile

        virtual_events = [[('reach', iplayer, 1), ('defg', iplayer, thrown_tile, None)],
            [('defg', iplayer, thrown_tile, None)]]
        virtual_probs = [prob, 1.-prob]
        if self.num_thread == 1:
            probs = self.get_action_prob(iplayer, mahjong_board, virtual_events, virtual_probs)
        else:
            probs = self.get_action_prob_thread(iplayer, mahjong_board, virtual_events, virtual_probs)
        if self.is_greedy:
            prob = float(probs[0] >= 0.5)
        else:
            prob = probs[0]
        do_reach = int(np.random.random_sample() < prob)

        if do_reach == 1:
            return [('reach', iplayer, 1)]
        else:
            return []

    def play_defg(self, iplayer, mahjong_board, event):
        hand = mahjong_board.players[iplayer].hand
        if mahjong_board.players[iplayer].is_reached: # should tsumo-giri
            thrown_tile = hand[-1]
            return [('defg', iplayer, thrown_tile, None)]

        self.reset_local_variables()
        self.sub_player.acquire_thrown_feature(iplayer, mahjong_board)
        probs, prob_reach = self.sub_player.predict_hand(iplayer, mahjong_board, 'defg')

        virtual_events = [[('defg', iplayer, thrown_tile, None)] for thrown_tile in hand]
        virtual_probs = probs
        if self.num_thread == 1:
            probs_init = self.get_action_prob(iplayer, mahjong_board, virtual_events, virtual_probs)
        else:
            probs_init = self.get_action_prob_thread(iplayer, mahjong_board, virtual_events, virtual_probs)
        #print(virtual_probs, probs_init)
        if self.is_greedy:
            probs = np.zeros((14,), dtype='float')
            probs[np.argmax(probs_init)] = 1.
        else:
            probs = probs_init
        throw_index = np.random.choice(len(probs), p=probs)
        thrown_tile = hand[throw_index]
        self._prob_reach = prob_reach[throw_index]
        self._thrown_tile = thrown_tile
        return [('defg', iplayer, thrown_tile, None)]

    def play_tsumo(self, iplayer, mahjong_board, event):
        self.reset_local_variables()
        self.sub_player.acquire_thrown_feature(iplayer, mahjong_board)
        prob = self.sub_player.predict_hand(iplayer, mahjong_board, 'tsumo')

        defg_events = self.sub_player.play_defg(iplayer, mahjong_board, ('defg', iplayer)) # We use sub-player at this time.
        virtual_events = [[event], defg_events]
        virtual_probs = [prob, 1.-prob]
        if self.num_thread == 1:
            probs = self.get_action_prob(iplayer, mahjong_board, virtual_events, virtual_probs)
        else:
            probs = self.get_action_prob_thread(iplayer, mahjong_board, virtual_events, virtual_probs)
        if self.is_greedy:
            prob = float(probs[0] >= 0.5)
        else:
            prob = probs[0]
        do_tsumo = int(np.random.random_sample() < prob)

        if do_tsumo == 1:
            return [event]
        else:
            return []

    def play_ron(self, iplayer, mahjong_board, event):
        hand = mahjong_board.players[iplayer].hand
        hand_array = np.array(hand)
        agari_tile = mahjong_board.agari_tile

        self.reset_local_variables()
        self.sub_player.acquire_thrown_feature(iplayer, mahjong_board)
        probs_init = self.sub_player.predict_hand(iplayer, mahjong_board, 'ron') # 14 * 5
        prob = 1.
        if np.sum(hand_array//4 == agari_tile//4) > 0:
            prob = min(prob, np.min(probs_init[hand_array//4 == agari_tile//4, 2]))
        if agari_tile//4 < 27:
            if (agari_tile//4)%9 >= 1 and np.sum(hand_array//4 == agari_tile//4-1) > 0:
                prob = min(prob, np.min(probs_init[hand_array//4 == agari_tile//4-1, 3]))
            if (agari_tile//4)%9 >= 2 and np.sum(hand_array//4 == agari_tile//4-2) > 0:
                prob = min(prob, np.min(probs_init[hand_array//4 == agari_tile//4-2, 4]))
            if (agari_tile//4)%9 <= 7 and np.sum(hand_array//4 == agari_tile//4+1) > 0:
                prob = min(prob, np.min(probs_init[hand_array//4 == agari_tile//4+1, 1]))
            if (agari_tile//4)%9 <= 6 and np.sum(hand_array//4 == agari_tile//4+2) > 0:
                prob = min(prob, np.min(probs_init[hand_array//4 == agari_tile//4+2, 0]))

        agari_from_who = event[2][0]
        virtual_events = [[event], [('tuvw', (agari_from_who+1)%NUM_PLAYER)]]
        virtual_probs = [prob, 1.-prob]
        if self.num_thread == 1:
            probs = self.get_action_prob(iplayer, mahjong_board, virtual_events, virtual_probs)
        else:
            probs = self.get_action_prob_thread(iplayer, mahjong_board, virtual_events, virtual_probs)
        if self.is_greedy:
            prob = float(probs[0] >= 0.5)
        else:
            prob = probs[0]
        do_ron = int(np.random.random_sample() < prob)

        if do_ron == 1:
            return [event]
        else:
            return []

    def play_chi(self, iplayer, mahjong_board, event):
        def compute_chi_prob(chi_type, probs):
            prob = 1. - (1. - probs[chi_type-1]) * (1. - probs[chi_type])
            return prob

        def compete_two_chi_types(chi_type_1, chi_type_2, probs):
            prob_1 = probs[chi_type_1-1] * probs[chi_type_1]
            prob_2 = probs[chi_type_2-1] * probs[chi_type_2]
            if prob_1 >= prob_2:
                return chi_type_1
            else:
                return chi_type_2

        hand = mahjong_board.players[iplayer].hand
        hand_array = np.array(hand)
        chi_availables = event[3]
        call_tile = chi_availables[0][2][chi_availables[0][1]]

        self.reset_local_variables()
        self.sub_player.acquire_thrown_feature(iplayer, mahjong_board)
        probs_init = self.sub_player.predict_hand(iplayer, mahjong_board, 'chi') # 14 * 4
        type_availables = [False] * 3
        probs = np.array([0.] * 4)
        for (meld_from_who, meld_called, meld_tiles) in chi_availables:
            type_availables[meld_called] = True
        chi_univs = (hand_array//4 == call_tile//4+2, hand_array//4 == call_tile//4+1,
            hand_array//4 == call_tile//4-1, hand_array//4 == call_tile//4-2)
        if type_availables[0]:
            assert(np.sum(chi_univs[0]) > 0)
            probs[0] = np.max(probs_init[chi_univs[0], 0])
        if type_availables[0] or type_availables[1]:
            assert(np.sum(chi_univs[1]) > 0)
            probs[1] = np.max(probs_init[chi_univs[1], 1])
        if type_availables[1] or type_availables[2]:
            assert(np.sum(chi_univs[2]) > 0)
            probs[2] = np.max(probs_init[chi_univs[2], 2])
        if type_availables[2]:
            assert(np.sum(chi_univs[3]) > 0)
            probs[3] = np.max(probs_init[chi_univs[3], 3])
        type_available_num = np.sum(type_availables)
        if type_available_num == 1:
            chi_type = 1 + np.argmax(type_availables)
        elif type_available_num == 2:
            assert(type_availables[1])
            chi_type_1 = 2
            if type_availables[0]:
                chi_type_2 = 1
            else:
                assert(type_availables[2])
                chi_type_2 = 3
            chi_type = compete_two_chi_types(chi_type_1, chi_type_2, probs)
        else:
            assert(type_available_num == 3)
            if probs[0] < 0.5 and probs[0] <= probs[3]:
                chi_type = compete_two_chi_types(2, 3, probs)
            elif probs[3] < 0.5 and probs[3] <= probs[0]:
                chi_type = compete_two_chi_types(2, 1, probs)
            else:
                assert(probs[0] >= 0.5 and probs[3] >= 0.5)
                chi_type = compete_two_chi_types(1, 3, probs)

        chi_index = -1
        tile1 = hand_array[chi_univs[chi_type-1]][np.argmax(probs_init[chi_univs[chi_type-1], chi_type-1])]
        tile2 = hand_array[chi_univs[chi_type]][np.argmax(probs_init[chi_univs[chi_type], chi_type])]
        for ichi, (meld_from_who, meld_called, meld_tiles) in enumerate(chi_availables):
            if set(meld_tiles) == set([tile1, tile2, call_tile]):
                chi_index = ichi
                break
        assert(chi_index >= 0)
        prob = compute_chi_prob(chi_type, probs)

        meld_from_who = chi_availables[0][0]
        return_event = ('n', 'chi', iplayer, chi_availables[chi_index])
        virtual_events = [[return_event], [('tuvw', (meld_from_who+1)%NUM_PLAYER)]]
        virtual_probs = [prob, 1.-prob]
        if self.num_thread == 1:
            probs = self.get_action_prob(iplayer, mahjong_board, virtual_events, virtual_probs)
        else:
            probs = self.get_action_prob_thread(iplayer, mahjong_board, virtual_events, virtual_probs)
        if self.is_greedy:
            prob = float(probs[0] >= 0.5)
        else:
            prob = probs[0]
        do_chi = int(np.random.random_sample() < prob)

        if do_chi == 0:
            return []
        else:
            return [return_event]

    def play_pon(self, iplayer, mahjong_board, event):
        hand = mahjong_board.players[iplayer].hand
        hand_array = np.array(hand)
        pon_availables = event[3]
        call_tile = pon_availables[0][2][pon_availables[0][1]]

        self.reset_local_variables()
        self.sub_player.acquire_thrown_feature(iplayer, mahjong_board)
        prob_init = self.sub_player.predict_hand(iplayer, mahjong_board, 'pon') # 14
        pon_univ = hand_array//4 == call_tile//4
        assert(np.sum(pon_univ) >= 2)
        prob = np.max(prob_init[pon_univ])

        pon_index = -1
        argsort_tiles = hand_array[pon_univ][np.argsort(prob_init[pon_univ])]
        tile1 = argsort_tiles[-1]
        tile2 = argsort_tiles[-2]
        for ipon, (meld_from_who, meld_called, meld_tiles) in enumerate(pon_availables):
            if set(meld_tiles) == set([tile1, tile2, call_tile]):
                pon_index = ipon
                break
        assert(pon_index >= 0)

        meld_from_who = pon_availables[0][0]
        return_event = ('n', 'pon', iplayer, pon_availables[pon_index])
        virtual_events = [[return_event], [('tuvw', (meld_from_who+1)%NUM_PLAYER)]]
        virtual_probs = [prob, 1.-prob]
        if self.num_thread == 1:
            probs = self.get_action_prob(iplayer, mahjong_board, virtual_events, virtual_probs)
        else:
            probs = self.get_action_prob_thread(iplayer, mahjong_board, virtual_events, virtual_probs)
        if self.is_greedy:
            prob = float(probs[0] >= 0.5)
        else:
            prob = probs[0]
        do_pon = int(np.random.random_sample() < prob)

        if do_pon == 0:
            return []
        else:
            return [return_event]

    def play_daiminkan(self, iplayer, mahjong_board, event):
        self.reset_local_variables()
        return self.sub_player.play_daiminkan(iplayer, mahjong_board, event) # No tree search for daiminkan

        hand = mahjong_board.players[iplayer].hand
        hand_array = np.array(hand)
        daiminkan_availables = event[3]
        assert(len(daiminkan_availables) == 1)
        call_tile = daiminkan_availables[0][2][daiminkan_availables[0][1]]

        self.reset_local_variables()
        self.sub_player.acquire_thrown_feature(iplayer, mahjong_board)
        prob_init = self.sub_player.predict_hand(iplayer, mahjong_board, 'daiminkan') # 14
        daiminkan_univ = hand_array//4 == call_tile//4
        assert(np.sum(daiminkan_univ) == 3)
        prob = np.max(prob_init[daiminkan_univ])

        meld_from_who = daiminkan_availables[0][0]
        return_event = ('n', 'daiminkan', iplayer, daiminkan_availables[0])
        virtual_events = [[return_event], [('tuvw', (meld_from_who+1)%NUM_PLAYER)]]
        virtual_probs = [prob, 1.-prob]
        if self.num_thread == 1:
            probs = self.get_action_prob(iplayer, mahjong_board, virtual_events, virtual_probs)
        else:
            probs = self.get_action_prob_thread(iplayer, mahjong_board, virtual_events, virtual_probs)
        if self.is_greedy:
            prob = float(probs[0] >= 0.5)
        else:
            prob = probs[0]
        do_daiminkan = int(np.random.random_sample() < prob)

        if do_daiminkan == 0:
            return []
        else:
            return [return_event]

    def play_chakan(self, iplayer, mahjong_board, event):
        self.reset_local_variables()
        self.sub_player.acquire_thrown_feature(iplayer, mahjong_board)
        return self.sub_player.play_chakan(iplayer, mahjong_board, event) # No tree search for chakan

    def play_ankan(self, iplayer, mahjong_board, event):
        self.reset_local_variables()
        self.sub_player.acquire_thrown_feature(iplayer, mahjong_board)
        return self.sub_player.play_ankan(iplayer, mahjong_board, event) # No tree search for ankan

    def play_yao9(self, iplayer, mahjong_board, event):
        self.reset_local_variables()
        self.sub_player.acquire_thrown_feature(iplayer, mahjong_board)
        return self.sub_player.play_yao9(iplayer, mahjong_board, event) # No tree search for yao9

