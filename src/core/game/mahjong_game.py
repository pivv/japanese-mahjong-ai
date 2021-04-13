#-*-coding:utf-8-*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time

import numpy as np
from copy import copy, deepcopy

from collections import defaultdict

from .mahjong_logic import MahjongBoard
from .mahjong_constant import *

class MahjongGame():
    def __init__(self, game_type, play_mode):
        self.game_type = game_type
        self.play_mode = play_mode
        assert(self.play_mode in ['self_play', 'blind_play', 'virtual_play', 'log_play', 'log_self_play', 'log_debug_play'])
        self.mahjong_tiles = None
        if self.play_mode == 'self_play':
            self.mahjong_tiles = np.arange(NUM_TILE)
        self.tile_index = 0
        self.events = []
        self.event_agaris = []

    def get_game_ended(self, mahjong_board, iplayer): # return the rank of player
        return mahjong_board.owari_points is not None

    def get_init_board(self):
        self.mahjong_tiles = None
        if self.play_mode == 'self_play':
            self.mahjong_tiles = np.arange(NUM_TILE)
        self.tile_index = 0
        self.events = []
        self.event_agaris = []
        mahjong_board = MahjongBoard(self.game_type, self.play_mode)
        return mahjong_board

    def is_post_processing_event(self, event):
        event_type = event[0]
        if self.play_mode == 'self_play':
            return event_type in ['init', 'tuvw', 'dora', 'agari', 'ryuukyoku']
        elif self.play_mode == 'virtual_play':
            return event_type in ['tuvw', 'dora', 'agari', 'ryuukyoku']
        else: # blind_play, log_play, log_self_play, log_debug_play
            return False

    def is_no_output_event(self, event):
        return event[0] in ['un', 'bye', 'dora', 'reach', 'owari']

    def get_current_player_from_event(self, event): # who will process the event?
        event_type = event[0]
        if event_type in ['defg', 'agari']:
            return event[1]
        elif event_type == 'n' or (event_type == 'ryuukyoku' and event[1] == 'yao9'):
            return event[2]
        else:
            return -1

    def post_process_event(self, mahjong_board, event): # do player independent processes
        assert(self.play_mode in ['self_play', 'virtual_play'])
        event_type = event[0]
        if self.play_mode == 'self_play':
            if event_type == 'init':
                assert(len(event) == 1)
                np.random.shuffle(self.mahjong_tiles)
                self.tile_index = 0
                dora_indicators = [self.mahjong_tiles[self.tile_index]]
                self.tile_index += 1
                hands = []
                for iplayer in range(4):
                    hands.append(np.copy(self.mahjong_tiles[
                        self.tile_index:self.tile_index+NUM_STARTING_HAND_TILE]).tolist())
                    self.tile_index += NUM_STARTING_HAND_TILE
                return ('init', False, dora_indicators, hands, None, None, None, None, None, None, None)
            elif event_type == 'tuvw':
                assert(len(event) == 2)
                iplayer = event[1]
                new_tile = self.mahjong_tiles[self.tile_index]
                self.tile_index += 1
                return ('tuvw', iplayer, new_tile)
            #elif event_type in ['un', 'bye']: cannot be happened
            elif event_type == 'dora':
                assert(len(event) == 1)
                dora_indicator = self.mahjong_tiles[self.tile_index]
                self.tile_index += 1
                return ('dora', dora_indicator)
            elif event_type == 'agari':
                assert(len(event) == 3)
                uradora_indicators = np.copy(self.mahjong_tiles[
                    self.tile_index:self.tile_index+len(mahjong_board.dora_indicators)]).tolist()
                # no adding for self.tile_index, considering multiple agari
                return event + (uradora_indicators,)
            elif event_type == 'ryuukyoku':
                assert(len(event) == 2 or len(event) == 3)
                if len(event) == 2:
                    return event + (None, None)
                else:
                    return event + (None,)
            else:
                raise ValueError('Wrong event type: %s'%(event_type))
        elif self.play_mode == 'virtual_play':
            if event_type == 'tuvw':
                assert(len(event) == 2)
                iplayer = event[1]
                new_tile = self.get_virtual_tile(mahjong_board)
                return ('tuvw', iplayer, new_tile)
            elif event_type == 'dora':
                assert(len(event) == 1)
                dora_indicator = self.get_virtual_tile(mahjong_board)
                return ('dora', dora_indicator)
            elif event_type == 'agari':
                assert(len(event) == 3)
                uradora_indicators = []
                for idora in range(len(mahjong_board.dora_indicators)):
                    uradora_indicators.append(self.get_virtual_tile(mahjong_board))
                    # We don't remove uradora from remain_tile_type_nums for stability
                return event + (uradora_indicators,)
            elif event_type == 'ryuukyoku':
                assert(len(event) == 2 or len(event) == 3)
                if len(event) == 2:
                    if event[1] == '':
                        tenpai_players = self.compute_virtual_tenpai_players(mahjong_board)
                        chin_is_tenpai = mahjong_board.oya in tenpai_players
                        ryuukyoku_scores = mahjong_board.compute_ryuukyoku_scores(tenpai_players)
                        return ('ryuukyoku', '', ryuukyoku_scores, chin_is_tenpai)
                    else:
                        return event + (None, None)
                else:
                    return event + (None,)
            else:
                raise ValueError('Wrong event type: %s'%(event_type))

    # Virtual functions

    def set_virtual_params(self, iplayer, remain_tile_type_nums, prob_tenpai_all, ypred_my_hand_all):
        self.my_player = iplayer
        self.remain_tile_type_nums = remain_tile_type_nums
        self.prob_tenpai_all = prob_tenpai_all
        self.ypred_my_hand_all = ypred_my_hand_all

    def get_virtual_tile(self, mahjong_board):
        # TODO: remove hand predicted tiles.
        remain_tile_type_nums = np.copy(self.remain_tile_type_nums)
        for j in range(1, NUM_PLAYER):
            jplayer = (self.my_player+j)%NUM_PLAYER
            p_hand = self.ypred_my_hand_all[j] / np.sum(self.ypred_my_hand_all[j])
            len_hand = (13 - 3 * (len(mahjong_board.players[jplayer].chis)+len(mahjong_board.players[jplayer].pons)+
                len(mahjong_board.players[jplayer].daiminkans)+len(mahjong_board.players[jplayer].chakans)+
                len(mahjong_board.players[jplayer].ankans)))
            for _ in range(len_hand):
                tile_type = np.random.choice(NUM_TILE_TYPE, p=p_hand)
                if remain_tile_type_nums[tile_type] > 0:
                    remain_tile_type_nums[tile_type] -= 1
        p = remain_tile_type_nums.astype('float32') / float(np.sum(remain_tile_type_nums))
        virtual_tile = np.random.choice(NUM_TILE_TYPE, p=p) * 4 + 1 # We assume no aka tiles.
        return virtual_tile

    def compute_virtual_tenpai_players(self, mahjong_board):
        tenpai_players = []
        for iplayer in range(NUM_PLAYER):
            if iplayer == self.my_player:
                if mahjong_board.check_player_is_tenpai(iplayer):
                    tenpai_players.append(iplayer)
            else:
                if np.random.random_sample() < self.prob_tenpai_all[(iplayer-self.my_player)%NUM_PLAYER]:
                    tenpai_players.append(iplayer)
        return tenpai_players

    def get_next_state(self, mahjong_board, events):
        for ievent, event in enumerate(events):
            event_type = event[0]
            if self.is_post_processing_event(event): # post-processing
                event = self.post_process_event(mahjong_board, event)
            if event_type == 'agari': # special case
                self.event_agaris.append(event)
                self.events = []
            else:
                if self.play_mode in ['self_play', 'virtual_play']:
                    assert(len(self.event_agaris) == 0)
                if (self.play_mode in ['blind_play', 'log_play', 'log_self_play', 'log_debug_play'] and
                    len(self.event_agaris) > 0):
                    # For those modes, agari update is done here.
                    self.events = mahjong_board.process_agari(self.event_agaris)
                    del self.event_agaris[:]
                process_fn = getattr(mahjong_board, 'process_' + event_type)
                if event_type == 'owari':
                    process_fn()
                    return -2, mahjong_board, None
                elif self.is_no_output_event(event):
                    process_fn(event)
                else:
                    self.events = process_fn(event)
        if self.play_mode in ['self_play', 'virtual_play'] and len(self.event_agaris) > 0 and len(self.events) == 0:
            # For self play mode and virtual play mode, agari update is done here.
            self.events = mahjong_board.process_agari(self.event_agaris)
            del self.event_agaris[:]

        if len(self.events) == 0: # Currently do not have anything to do: yet not initialized.
            return -1, mahjong_board, ('',)
        event = self.events[0]
        del self.events[0]
        return self.get_current_player_from_event(event), mahjong_board, event

    '''
    def getCanonicalForm(self, mahjong_board, iplayer):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            canonicalBoard: returns canonical form of board. The canonical form
                            should be independent of player. For e.g. in chess,
                            the canonical form can be chosen to be from the pov
                            of white. When the player is white, we can return
                            board as is. When the player is black, we can invert
                            the colors and return the board.
        """
        players =

    def stringRepresentation(self, mahjong_board):
        """
        Input:
            board: current board

        Returns:
            boardString: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """
        pass
    '''
