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

from .mahjong_logic_cython import acquire_waiting_tile_types_from_keep_nums_cython, check_agari_from_keep_nums_cython
from .mahjong_constant import *

class MahjongPlayerData(object):
    def __init__(self):
        self.initialize_variables()

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

    def initialize_variables(self):
        self.hand = [] ## should not be used for unknown

        self.cumulative_melds = []
        self.cumulative_melds_from_who = []
        self.cumulative_melds_called = []
        self.cumulative_melds_type = []

        self.chis = []
        self.chis_from_who = []
        self.chis_called = []

        self.pons = []
        self.pons_from_who = []
        self.pons_called = []

        self.daiminkans = []
        self.daiminkans_from_who = []
        self.daiminkans_called = []

        self.chakans = []
        self.chakans_from_who = []
        self.chakans_called = []

        self.ankans = []

        self.thrown = []
        self.thrown_is_called = []
        self.thrown_is_reached = []
        self.waiting_type = [] ## should not be used for unknown
        self.waiting_score = [] ## should not be used for unknown
        self.waiting_limit = [] ## should not be used for unknown

        self.agari_infos = []

        self.is_first = True
        self.is_reached = False
        self.is_double_reached = False
        self.is_ippatsu = False
        self.is_furiten = False ## should not be used for unknown
        self.is_temp_furiten = False ## should not be used for unknown
        self.is_called_by_other = False
        self.is_connected = True

        self.pao_who = -1

        # Other infos
        self.hand_infos = [] ## should not be used for unknown
        self.thrown_infos = []
        self.is_tsumo_giri = False
        self.is_reach_available = False
        self.last_tsumoed = False
        self.last_minkanned = False
        self.last_rinshan = False
        self.is_ready_to_reach = False
        self.last_chi = -1
        self.last_pon = -1
        self.opponent = -1
        self.safe_tile_types = []
        self.tiles_cannot_be_thrown = []
        self.last_tsumo_giri = False

        self.is_agari = False
        self.is_agari_by_who = False
        self.agari_score = 0

    def blind_variables(self):
        self.hand = None
        self.waiting_type = None
        self.waiting_score = None
        self.waiting_limit = None
        self.is_furiten = None
        self.is_temp_furiten = None
        self.hand_infos = None
        self.is_reach_available = None
        self.tiles_cannot_be_thrown = None


class MahjongBoard(object): # Implement mahjong logics.

    def __init__(self, game_type=169, play_mode='self_play'):
        self.initialize_game_variables(game_type, play_mode)
        self.initialize_round_variables()

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

    def initialize_game_variables(self, game_type, play_mode):
        # 1. Fixed information for whole game.

        self.game_type = game_type
        assert(play_mode in ['self_play', 'blind_play', 'virtual_play', 'log_play', 'log_self_play', 'log_debug_play'])
        self.play_mode = play_mode
        self.start_score = 250
        self.target_score = 300
        """
        Game type decoding:
        0 - 1 - online, 0 - bots
        1 - aka forbiden
        2 - kuitan forbidden
        3 - hanchan
        4 - 3man
        5 - dan flag
        6 - fast game
        7 - dan flag
        Game type:   9, 00001001, game number :      2, type strings : 四鳳南喰赤－
        Game type: 161, 10100001, game number :   1393, type strings : 四鳳東喰赤  四鳳東喰赤－
        Game type: 163, 10100011, game number :     20, type strings : 四鳳東喰－  四鳳東喰－－
        Game type: 167, 10100111, game number :    204, type strings : 四鳳東－－  四鳳東－－－
        Game type: 169, 10101001, game number : 720170, type strings : 四鳳南喰赤  四鳳南喰赤－
        Game type: 171, 10101011, game number :   1994, type strings : 四鳳南喰－  四鳳南喰－－
        Game type: 175, 10101111, game number :     21, type strings : 四鳳南－－  四鳳南－－－
        Game type: 177, 10110001, game number :   3523, type strings : 三鳳東喰赤  三鳳東喰赤－
        Game type: 185, 10111001, game number : 322061, type strings : 三鳳南喰赤  三鳳南喰赤－
        Game type: 225, 11100001, game number : 403019, type strings : 四鳳東喰赤  四鳳東喰赤－
        Game type: 233, 11101001, game number :   1676, type strings : 四鳳南喰赤  四鳳南喰赤－
        Game type: 241, 11110001, game number :    340, type strings : 三鳳東喰赤  三鳳東喰赤－
        Game type: 249, 11111001, game number :    797, type strings : 三鳳南喰赤  三鳳南喰赤－
        """
        self.use_akadora = (game_type//2)%2 == 0
        if self.use_akadora:
            self.akadoras = AKADORAS[:]
        else:
            self.akadoras = []
        self.allow_kuitan = (game_type//4)%2 == 0
        if (game_type//16)%2 == 1:
            self.num_player = 3
        else:
            self.num_player = 4
        if (game_type//8)%2 == 1:
            self.num_round = 2*self.num_player
            self.uma = (20, 10)
        else:
            self.num_round = self.num_player
            #self.uma = (10, 5)
            self.uma = (20, 10)
        self.max_num_round = self.num_round + self.num_player
        self.oka = self.num_player * (self.target_score - self.start_score)

        assert(self.num_player == 4)

        self.my_player = -1 # used for blind play

    def initialize_round_variables(self):

        # 2. Fixed information for each game.
        self.owari_points = None
        self.player_names = np.array(['', '', '', ''])
        self.player_scores = np.ones((self.num_player,), dtype='int') * self.start_score
        self.oya = 0
        self.current_round = 0
        self.combo = 0
        self.reach_stick = 0

        # 3. Main information for each game.
        self.remain_tile_num = NUM_USABLE_TILE
        self.dora_indicators = []
        self.doras = []
        self.uradora_indicators = []
        self.uradoras = []

        # 4. Main player data
        self.players = [MahjongPlayerData() for _ in range(self.num_player)]

        # 5. Other Informations
        self.num_kan = 0
        self.remain_tile_type_nums = [4] * NUM_TILE_TYPE
        self.agari_tile = -1

        #if self.play_mode == 'blind_play':
        #    self.blind_board(2)

    def blind_board(self, my_player, play_mode='blind_play'):
        assert(play_mode in ['blind_play', 'virtual_play'])
        self.play_mode = play_mode
        self.my_player = my_player
        for j in range(self.num_player-1):
            jplayer = (self.my_player+1+j)%(self.num_player)
            self.players[jplayer].blind_variables()

    # These processes may have next processes
    # next process is form of (type, who, details)

    def process_init(self, event_init): # Start process
        _, is_reinit, dora_indicators, hands, throwns, melds, oya, round, combo, reach_stick, total_scores = event_init
        #assert(not is_reinit)
        assert(self.play_mode != 'virtual_play') # For virtual play no init signal

        if is_reinit:
            assert(self.play_mode == 'blind_play')

        if self.play_mode in ['blind_play', 'log_play', 'log_self_play']:
            #assert(np.all(self.player_scores == np.array(total_scores)))
            self.player_scores = np.array(total_scores).astype('int')
            self.oya = oya
            self.current_round = round
            self.combo = combo
            self.reach_stick = reach_stick

        if self.play_mode != 'blind_play':
            assert(self.oya == self.current_round%self.num_player)

        self.remain_tile_num = NUM_USABLE_TILE
        self.dora_indicators = dora_indicators[:]
        if not is_reinit:
            assert(len(self.dora_indicators) == 1)
        self.doras = [self.compute_dora_tile_type(dora_indicator) for dora_indicator in self.dora_indicators]
        self.uradora_indicators = []
        self.uradoras = []
        assert(len(hands) == self.num_player)
        for jplayer, hand in enumerate(hands):
            self.players[jplayer].initialize_variables()
            if self.play_mode != 'blind_play' or jplayer == self.my_player:
                self.players[jplayer].hand = hand[:]
                self.players[jplayer].hand = sorted(self.players[jplayer].hand)
                self.acquire_waiting_tile_types(jplayer)
                if not is_reinit:
                    assert(len(hand) == NUM_STARTING_HAND_TILE)
            else:
                self.players[jplayer].blind_variables()
            self.remain_tile_num -= NUM_STARTING_HAND_TILE

        if is_reinit:
            for iplayer, meld in enumerate(melds):
                for (meld_type, meld_from_who, meld_called, meld_tiles) in meld:
                    # 1. add meld
                    getattr(self.players[iplayer], meld_type + 's').append(meld_tiles)
                    if meld_type != 'ankan':
                        getattr(self.players[iplayer], meld_type + 's_from_who').append(meld_from_who)
                        getattr(self.players[iplayer], meld_type + 's_called').append(meld_called)
                    else:
                        assert(meld_from_who == iplayer)
                        assert(meld_called == -1)
                    # cumulative infos
                    if meld_type == 'chakan': # add previous pon to cumulative melds to avoid confusion.
                        self.players[iplayer].cumulative_melds.append(meld_tiles[:-1])
                        self.players[iplayer].cumulative_melds_from_who.append(meld_from_who)
                        self.players[iplayer].cumulative_melds_called.append(meld_called)
                        self.players[iplayer].cumulative_melds_type.append('pon')
                    self.players[iplayer].cumulative_melds.append(meld_tiles)
                    self.players[iplayer].cumulative_melds_from_who.append(meld_from_who)
                    self.players[iplayer].cumulative_melds_called.append(meld_called)
                    self.players[iplayer].cumulative_melds_type.append(meld_type)
                    # 2. add thrown
                    if meld_type != 'ankan':
                        assert(iplayer != meld_from_who)
                        self.players[meld_from_who].thrown.append(meld_tiles[meld_called])
                        self.players[meld_from_who].thrown_is_called.append(True)
                        self.players[meld_from_who].thrown_is_reached.append(False)
                    # 3. move remain_tile_num
                    if meld_type in ['chi', 'pon']:
                        self.remain_tile_num += 1
                    elif meld_type == 'ankan':
                        self.remain_tile_num -= 1
            for iplayer, thrown in enumerate(throwns):
                is_reached_flag = False
                for thrown_tile in thrown:
                    # 1. add thrown
                    if thrown_tile == 255:
                        is_reached_flag = True
                    else:
                        assert(0 <= thrown_tile < NUM_TILE)
                        self.players[iplayer].thrown.append(thrown_tile)
                        self.players[iplayer].thrown_is_called.append(False)
                        self.players[iplayer].thrown_is_reached.append(is_reached_flag)
                        is_reached_flag = False
                     # 2. move remain_tile_num
                    self.remain_tile_num -= 1
                if is_reached_flag:
                    self.players[iplayer].thrown_is_reached[-1] = True
            for iplayer in range(self.num_player):
                if self.play_mode != 'blind_play' or iplayer == self.my_player:
                    self.players[iplayer].is_furiten = self.check_player_is_furiten(iplayer)

        self.num_kan = 0
        if is_reinit: # recompute number of kans
            for jplayer in range(self.num_player):
                player_num_kan = (len(self.players[jplayer].daiminkans) +
                    len(self.players[jplayer].chakans) + len(self.players[jplayer].ankans))
                self.num_kan += player_num_kan

        self.remain_tile_type_nums = [4] * NUM_TILE_TYPE
        for dora_indicator in self.dora_indicators:
            self.remain_tile_type_nums[dora_indicator//4] -= 1
            assert(self.remain_tile_type_nums[dora_indicator//4] >= 0)
        if is_reinit: # compute remain_tile_type_nums for throwns and melds
            for jplayer in range(self.num_player):
                for tile in self.players[jplayer].thrown:
                    self.remain_tile_type_nums[tile//4] -= 1
                    assert(self.remain_tile_type_nums[tile//4] >= 0)
            for jplayer in range(self.num_player):
                for meld_called, meld_tiles in zip(self.players[jplayer].chis_called, self.players[jplayer].chis):
                    for it in range(3):
                        if meld_called != it:
                            self.remain_tile_type_nums[meld_tiles[it]//4] -= 1
                            assert(self.remain_tile_type_nums[meld_tiles[it]//4] >= 0)
                for meld_tiles in self.players[jplayer].pons:
                    self.remain_tile_type_nums[meld_tiles[0]//4] -= 2
                    assert(self.remain_tile_type_nums[meld_tiles[0]//4] >= 0)
                for meld_tiles in self.players[jplayer].daiminkans:
                    self.remain_tile_type_nums[meld_tiles[0]//4] -= 3
                    assert(self.remain_tile_type_nums[meld_tiles[0]//4] == 0)
                for meld_tiles in self.players[jplayer].chakans:
                    self.remain_tile_type_nums[meld_tiles[0]//4] -= 3
                    assert(self.remain_tile_type_nums[meld_tiles[0]//4] == 0)
                for meld_tiles in self.players[jplayer].ankans:
                    self.remain_tile_type_nums[meld_tiles[0]//4] -= 4
                    assert(self.remain_tile_type_nums[meld_tiles[0]//4] == 0)

        self.agari_tile = -1

        if self.play_mode == 'log_play':
            for iplayer in range(self.num_player):
                jplayer = (iplayer + 1 + int((self.num_player-1) * np.random.random_sample()))%self.num_player
                assert(iplayer != jplayer)
                self.players[iplayer].opponent = jplayer

        if self.play_mode in ['self_play', 'blind_play', 'log_play', 'log_self_play']:
            for iplayer in range(self.num_player):
                jplayer = self.players[iplayer].opponent
                self.add_thrown_info_to_player(iplayer, jplayer, -1)
            if is_reinit:
                meld_flag = False
                for iplayer, meld in enumerate(melds):
                    for (meld_type, meld_from_who, meld_called, meld_tiles) in meld:
                        meld_flag = True
                        if meld_type != 'ankan':
                            assert(iplayer != meld_from_who)
                            self.add_thrown_info_to_player(meld_from_who, -1, meld_tiles[meld_called])
                            self.players[meld_from_who].is_first = False
                        if meld_type in ['chi', 'pon', 'daiminkan']:
                            self.players[meld_from_who].is_called_by_other = True
                if meld_flag:
                    for iplayer in range(self.num_player):
                        self.players[iplayer].is_first = False
                for iplayer, thrown in enumerate(throwns):
                    for ithrown, thrown_tile in enumerate(thrown):
                        if thrown_tile == 255:
                            self.players[iplayer].is_ready_to_reach = True
                            self.players[iplayer].is_reached = True
                            # No ippatsu, since may be called by other
                            if ithrown == 0:
                                self.players[iplayer].is_double_reached = True
                        else:
                            if self.players[iplayer].is_reached and not self.players[iplayer].is_ready_to_reach:
                                self.players[iplayer].is_tsumo_giri = True
                            self.add_thrown_info_to_player(iplayer, -1, thrown_tile)
                            self.players[iplayer].is_first = False
                            self.players[iplayer].is_ready_to_reach = False
                            self.players[iplayer].is_tsumo_giri = False
                    self.players[iplayer].is_ready_to_reach = False

        next_processes = []
        if self.play_mode in ['self_play', 'blind_play', 'log_self_play', 'log_debug_play']:
            if not is_reinit:
                # next event : tuvw
                iplayer = self.oya
                next_processes.append(('tuvw', iplayer))
            else:
                next_processes.append(('tuvw', -1)) # Who is next turn?

        return next_processes

    def process_owari(self): # Last process
        oya_init = (self.oya - self.current_round)%self.num_player
        eps = 1e-8
        owari_points = np.copy(self.player_scores).astype('float')
        owari_points = np.concatenate([owari_points[oya_init:], owari_points[:oya_init]])
        owari_points -= self.target_score
        argsort = np.argsort(-owari_points+eps*np.arange(len(owari_points))) # argsort[0] is top
        owari_points[argsort[0]] += 10 * self.reach_stick
        owari_points[argsort[0]] += self.oka
        owari_points /= 10.
        for i, v in enumerate(self.uma):
            owari_points[argsort[i]] += v
            owari_points[argsort[-1-i]] -= v
        owari_points = np.concatenate([owari_points[self.num_player-oya_init:], owari_points[:self.num_player-oya_init]])
        self.owari_points = owari_points

    def process_n(self, event_n):
        _, meld_type, iplayer, (meld_from_who, meld_called, meld_tiles) = event_n
        #iplayer = event_n.who
        #meld_type = event_n.type
        #meld_from_who = event_n.from_who
        #meld_called = event_n.called
        #meld_tiles = list(event_n.tiles)

        if self.play_mode == 'log_play': # data acquire
            # 1. add late hand info
            self._players_late_hand_infos_to_add = defaultdict(dict)
            meld_done = (meld_from_who, meld_called, meld_tiles)
            assert(iplayer in self._players_hand_infos_to_add and
                meld_type in self._players_hand_infos_to_add[iplayer])
            self._players_late_hand_infos_to_add[iplayer][meld_type + '_done'] = meld_done
            if meld_type == 'chi':
                self._players_late_hand_infos_to_add[meld_from_who]['chi_ed'] = (meld_tiles[meld_called], meld_called)
            elif meld_type == 'pon':
                self._players_late_hand_infos_to_add[meld_from_who]['pon_ed'] = (meld_tiles[meld_called], iplayer)
            if meld_type == 'pon' or meld_type == 'daiminkan':
                jplayer = (meld_from_who+1)%self.num_player
                if (jplayer != iplayer and jplayer in self._players_hand_infos_to_add and
                    'chi' in self._players_hand_infos_to_add[jplayer]): # remove chi from player
                    self._players_late_hand_infos_to_add[jplayer]['chi_remove'] = meld_done
            for jplayer in self._players_late_hand_infos_to_add:
                self.add_late_hand_info_to_player(jplayer, self._players_late_hand_infos_to_add[jplayer])

            # 2. add hand info
            self._players_hand_infos_to_add = defaultdict(dict)
            if meld_type == 'chakan':
                for j in range(self.num_player-1):
                    jplayer = (iplayer+1+j)%self.num_player
                    if not self.players[jplayer].is_furiten and not self.players[jplayer].is_temp_furiten:
                        agari_availables = self.compute_agari_availables(jplayer, iplayer, meld_tiles[-1], 'chakan')
                        if len(agari_availables) > 0: # agari acquire
                            self._players_hand_infos_to_add[jplayer]['ron'] = meld_tiles[-1]
            for jplayer in self._players_hand_infos_to_add:
                self.add_hand_info_to_player(jplayer, hand_infos_to_add=self._players_hand_infos_to_add[jplayer])

        #if meld_type in ['chakan', 'ankan']:
        #    if self.play_mode not in ['blind_play', 'virtual_play'] or iplayer == self.my_player:
        #    assert(int(self.players[iplayer].last_chi >= 0) + int(self.players[iplayer].last_pon >= 0) +
        #        int(self.players[iplayer].last_tsumoed) == 1)
        #    else:
        #        assert (int(self.players[iplayer].last_chi >= 0) + int(self.players[iplayer].last_pon >= 0) +
        #            int(self.players[iplayer].last_tsumoed) == 0) # should tsumo
        #        self.remain_tile_num -= 1

        if self.play_mode not in ['blind_play', 'virtual_play'] or iplayer == self.my_player:
            self.remove_meld_tiles_from_player_hand(iplayer, meld_type, meld_from_who, meld_called, meld_tiles)
        self.add_meld_tiles_to_player(iplayer, meld_type, meld_from_who, meld_called, meld_tiles)
        # check pao
        if self.check_pao(iplayer, meld_type, meld_from_who, meld_tiles):
            self.players[iplayer].pao_who = meld_from_who
        self.agari_tile = meld_tiles[-1]
        if meld_type == 'pon':
            base = meld_tiles[0] // 4
            self.players[iplayer].tiles_cannot_be_thrown = [base*4+t for t in range(4)]
        elif meld_type == 'chi':
            base = meld_tiles[meld_called] // 4
            self.players[iplayer].tiles_cannot_be_thrown = [base*4+t for t in range(4)]
            if meld_called == 0 and base % 9 <= 5:
                self.players[iplayer].tiles_cannot_be_thrown += [(base+3)*4+t for t in range(4)]
            elif meld_called == 2 and base % 9 >= 3:
                self.players[iplayer].tiles_cannot_be_thrown += [(base-3)*4+t for t in range(4)]

        next_processes = []
        if self.play_mode in ['self_play', 'blind_play', 'virtual_play', 'log_self_play', 'log_debug_play']:
            # next event of chi : defg
            # next event of pon : defg
            # next event of daiminkan : tuvw
            # next event of ankan : (dora)-dora-tuvw
            # next event of chakan : (dora)-tuvw, (dora)-agari
            if meld_type == 'chi' or meld_type == 'pon':
                assert(not self.players[iplayer].last_minkanned)
                next_processes.append(('defg', iplayer, self.players[iplayer].tiles_cannot_be_thrown))
            elif meld_type == 'daiminkan':
                assert(not self.players[iplayer].last_minkanned)
                next_processes.append(('tuvw', iplayer))
            elif meld_type == 'ankan':
                if self.players[iplayer].last_minkanned: # dora of last minkan
                    next_processes.append(('dora',))
                next_processes.append(('dora',))
                next_processes.append(('tuvw', iplayer))
            else:
                assert(meld_type == 'chakan')
                for j in range(self.num_player-1):
                    jplayer = (iplayer+1+j)%self.num_player
                    if self.play_mode not in ['blind_play', 'virtual_play'] or jplayer == self.my_player:
                        if not self.players[jplayer].is_furiten and not self.players[jplayer].is_temp_furiten:
                            agari_availables = self.compute_agari_availables(jplayer, iplayer, meld_tiles[-1], 'chakan')
                            if len(agari_availables) > 0:
                                next_processes.append(('agari', jplayer, agari_availables[0]))
                    elif self.play_mode == 'virtual_play' and jplayer != self.my_player:
                        next_processes.append(('agari', jplayer, (iplayer, 'chakan')))
                if self.players[iplayer].last_minkanned: # dora of last minkan
                    next_processes.append(('dora',))
                next_processes.append(('tuvw', iplayer))

        # set conditions
        self.players[iplayer].last_tsumoed = False
        if meld_type in ['chi', 'pon']:
            self.players[iplayer].is_reach_available = False
        if meld_type in ['daiminkan', 'chakan', 'ankan']:
            self.num_kan += 1
            self.players[iplayer].last_rinshan = True
            # recompute waiting tile types
            if self.play_mode not in ['blind_play', 'virtual_play'] or iplayer == self.my_player:
                if not self.players[iplayer].is_reached:
                    self.acquire_waiting_tile_types(iplayer)
                    self.players[iplayer].is_furiten = self.check_player_is_furiten(iplayer)
        if meld_type in ['daiminkan', 'chakan']:
            self.players[iplayer].last_minkanned = True
        else:
            self.players[iplayer].last_minkanned = False
        if meld_type in ['chi', 'pon', 'daiminkan']:
            assert(self.players[meld_from_who].thrown[-1] == meld_tiles[meld_called])
            self.players[meld_from_who].thrown_is_called[-1] = True
            self.players[meld_from_who].is_called_by_other = True
        for jplayer in range(self.num_player):
            self.players[jplayer].is_first = False
            self.players[jplayer].is_ippatsu = False
        if meld_type == 'chi':
            self.players[iplayer].last_chi = meld_tiles[0] // 4
        elif meld_type == 'pon':
            self.players[iplayer].last_pon = meld_tiles[0] // 4

        if not self.players[iplayer].is_reached: # player is reached
            del self.players[iplayer].safe_tile_types[:]
        if meld_type == 'chakan':
            for j in range(self.num_player - 1):
                jplayer = (iplayer+1+j)%self.num_player
                self.players[jplayer].safe_tile_types.append(meld_tiles[-1]//4)
                if self.play_mode not in ['blind_play', 'virtual_play'] or jplayer == self.my_player:
                    if meld_tiles[-1]//4 in self.players[jplayer].waiting_type: # temp furiten
                        if self.players[jplayer].is_reached:
                            self.players[jplayer].is_furiten = True
                        else:
                            self.players[jplayer].is_temp_furiten = True

        if meld_type == 'chi':
            for it in range(3):
                if meld_called != it:
                    self.remain_tile_type_nums[meld_tiles[it]//4] -= 1
                    assert(self.remain_tile_type_nums[meld_tiles[it]//4] >= 0)
        elif meld_type == 'pon':
            self.remain_tile_type_nums[meld_tiles[0]//4] -= 2
            assert(self.remain_tile_type_nums[meld_tiles[0]//4] >= 0)
        elif meld_type == 'daiminkan':
            self.remain_tile_type_nums[meld_tiles[0]//4] -= 3
            assert(self.remain_tile_type_nums[meld_tiles[0]//4] == 0)
        elif meld_type == 'chakan':
            self.remain_tile_type_nums[meld_tiles[0]//4] -= 1
            assert(self.remain_tile_type_nums[meld_tiles[0]//4] == 0)
        elif meld_type == 'ankan':
            self.remain_tile_type_nums[meld_tiles[0]//4] -= 4
            assert(self.remain_tile_type_nums[meld_tiles[0]//4] == 0)

        return next_processes

    def process_tuvw(self, event_tuvw):
        _, iplayer, new_tile = event_tuvw
        #iplayer = event_tuvw.who
        #new_tile = event_tuvw.tile

        if self.play_mode == 'virtual_play' and new_tile in self.players[self.my_player].hand: # multiple tile problem.
            base = new_tile // 4
            if base*4+1 not in self.players[self.my_player].hand:
                new_tile = base*4+1
            elif base*4+2 not in self.players[self.my_player].hand:
                new_tile = base*4+2
            elif base*4+3 not in self.players[self.my_player].hand:
                new_tile = base*4+3
            else:
                assert(base*4 not in self.players[self.my_player].hand)
                new_tile = base*4

        if self.play_mode not in ['blind_play', 'virtual_play'] or iplayer == self.my_player:
            self.add_tile_to_player(iplayer, new_tile)
        self.remain_tile_num -= 1
        self.agari_tile = new_tile
        self.players[iplayer].tiles_cannot_be_thrown = []
        self.players[iplayer].is_reach_available = self.check_reach_available(iplayer)

        if self.play_mode == 'log_play': # data acquire
            # 1. add late hand info
            if not self.players[iplayer].last_rinshan: # not rinshan_tsumo
                if not (self.players[iplayer].is_first and iplayer == self.oya): # not first tsumo
                    late_hand_infos_to_add = {}
                    late_hand_infos_to_add['next_tile'] = new_tile
                    self.add_late_hand_info_to_player((iplayer-1)%self.num_player, late_hand_infos_to_add)
            # 2. add hand info
            self._players_hand_infos_to_add = defaultdict(dict)
            agari_type = 'tsumo' if not self.players[iplayer].last_rinshan else 'rinshan_tsumo'
            agari_availables = self.compute_agari_availables(iplayer, iplayer, new_tile, agari_type)
            if len(agari_availables) > 0: # agari acquire
                self._players_hand_infos_to_add[iplayer]['tsumo'] = True
            if self.check_yao9(iplayer): # yao9 acquire
                self._yao9_player = iplayer
                self._players_hand_infos_to_add[iplayer]['yao9'] = True
            if self.remain_tile_num > 0 and self.num_kan < 4:
                if self.players[iplayer].is_reached:
                    ankan_availables = self.compute_reach_ankan_availables(iplayer, new_tile)
                    if len(ankan_availables) > 0: # ankan acquire
                        self._players_hand_infos_to_add[iplayer]['ankan'] = ankan_availables
                if not self.players[iplayer].is_reached:
                    ankan_availables = self.compute_ankan_availables(iplayer)
                    if len(ankan_availables) > 0: # ankan acquire
                        self._players_hand_infos_to_add[iplayer]['ankan'] = ankan_availables
                    chakan_availables = self.compute_chakan_availables(iplayer)
                    if len(chakan_availables) > 0: # chakan acquire
                        self._players_hand_infos_to_add[iplayer]['chakan'] = chakan_availables
            for jplayer in self._players_hand_infos_to_add:
                self.add_hand_info_to_player(jplayer, hand_infos_to_add=self._players_hand_infos_to_add[jplayer])

        next_processes = []
        if self.play_mode in ['self_play', 'blind_play', 'virtual_play', 'log_self_play', 'log_debug_play']:
            # next event of tuvw : (dora)-defg, (reach1)-defg, n, agari, ryuukyoku
            # agari
            if self.play_mode not in ['blind_play', 'virtual_play'] or iplayer == self.my_player:
                agari_type = 'tsumo' if not self.players[iplayer].last_rinshan else 'rinshan_tsumo'
                agari_availables = self.compute_agari_availables(iplayer, iplayer, new_tile, agari_type)
                if len(agari_availables) > 0:
                    next_processes.append(('agari', iplayer, agari_availables[0]))
                # yao9
                if self.check_yao9(iplayer):
                    next_processes.append(('ryuukyoku', 'yao9', iplayer))
                # n
                if self.remain_tile_num > 0 and self.num_kan < 4:
                    if self.players[iplayer].is_reached:
                        ankan_availables = self.compute_reach_ankan_availables(iplayer, new_tile)
                        if len(ankan_availables) > 0:
                            next_processes.append(('n', 'ankan', iplayer, ankan_availables))
                    else:
                        ankan_availables = self.compute_ankan_availables(iplayer)
                        if len(ankan_availables) > 0:
                            next_processes.append(('n', 'ankan', iplayer, ankan_availables))
                        chakan_availables = self.compute_chakan_availables(iplayer)
                        if len(chakan_availables) > 0:
                            next_processes.append(('n', 'chakan', iplayer, chakan_availables))
            elif self.play_mode == 'virtual_play' and iplayer != self.my_player:
                assert(not self.players[iplayer].last_rinshan)
                next_processes.append(('agari', iplayer, (iplayer, 'tsumo')))
            # dora
            if self.players[iplayer].last_minkanned:
                next_processes.append(('dora',))
            # reach1
            if self.players[iplayer].is_reach_available:
                next_processes.append(('reach', iplayer, 1))
            # defg
            next_processes.append(('defg', iplayer, self.players[iplayer].tiles_cannot_be_thrown))

        # set conditions
        self.players[iplayer].last_tsumoed = True

        return next_processes

    def process_defg(self, event_defg):
        _, iplayer, thrown_tile, is_tsumo_giri = event_defg
        #iplayer = event_defg.who
        #thrown_tile = event_defg.tile
        #if self.play_mode not in ['blind_play', 'virtual_play'] or iplayer == self.my_player:
        #assert(int(self.players[iplayer].last_chi >= 0) + int(self.players[iplayer].last_pon >= 0) +
        #    int(self.players[iplayer].last_tsumoed) == 1)
        #else:
        #    assert(int(self.players[iplayer].last_tsumoed) == 0)
        #    if (int(self.players[iplayer].last_chi >= 0) + int(self.players[iplayer].last_pon >= 0) +
        #        int(self.players[iplayer].last_tsumoed) == 0): # should tsumo
        #        self.remain_tile_num -= 1

        if (self.play_mode == 'virtual_play' and iplayer != self.my_player and
            thrown_tile in self.players[self.my_player].hand): # multiple tile problem.
            base = thrown_tile // 4
            if base*4+1 not in self.players[self.my_player].hand:
                thrown_tile = base*4+1
            elif base*4+2 not in self.players[self.my_player].hand:
                thrown_tile = base*4+2
            elif base*4+3 not in self.players[self.my_player].hand:
                thrown_tile = base*4+3
            else:
                assert(base*4 not in self.players[self.my_player].hand)
                thrown_tile = base*4

        if self.play_mode not in ['blind_play', 'virtual_play'] or iplayer == self.my_player:
            self.players[iplayer].is_tsumo_giri = False
            if self.players[iplayer].last_tsumoed and thrown_tile == self.players[iplayer].hand[-1]:
                self.players[iplayer].is_tsumo_giri = True
        else:
            assert(is_tsumo_giri is not None)
            self.players[iplayer].is_tsumo_giri = is_tsumo_giri

        self.throw_tile_from_player(iplayer, thrown_tile)

        self.agari_tile = thrown_tile

        if self.play_mode == 'log_play': # data acquire
            # 1. add late hand info
            # 2. add hand info
            self._players_hand_infos_to_add = defaultdict(dict)
            for j in range(self.num_player-1):
                jplayer = (iplayer+1+j)%self.num_player
                if not self.players[jplayer].is_furiten and not self.players[jplayer].is_temp_furiten:
                    agari_availables = self.compute_agari_availables(jplayer, iplayer, thrown_tile, 'ron')
                    if len(agari_availables) > 0: # agari acquire
                        self._players_hand_infos_to_add[jplayer]['ron'] = thrown_tile
            if self.remain_tile_num > 0:
                for j in range(self.num_player-1):
                    jplayer = (iplayer+1+j)%self.num_player
                    if not self.players[jplayer].is_reached:
                        pon_availables = self.compute_pon_availables(jplayer, iplayer, thrown_tile)
                        if len(pon_availables) > 0: # pon acquire
                            self._players_hand_infos_to_add[jplayer]['pon'] = pon_availables
                            if self.num_kan < 4:
                                daiminkan_availables = self.compute_daiminkan_availables(jplayer, iplayer, thrown_tile)
                                if len(daiminkan_availables) > 0: # daiminkan acquire
                                    self._players_hand_infos_to_add[jplayer]['daiminkan'] = daiminkan_availables
                jplayer = (iplayer+1)%self.num_player
                if not self.players[jplayer].is_reached:
                    chi_availables = self.compute_chi_availables(jplayer, iplayer, thrown_tile)
                    if len(chi_availables) > 0: # chi acquire
                        self._players_hand_infos_to_add[jplayer]['chi'] = chi_availables
            for jplayer in self._players_hand_infos_to_add:
                self.add_hand_info_to_player(jplayer, hand_infos_to_add=self._players_hand_infos_to_add[jplayer])

        next_processes = []
        if self.play_mode in ['self_play', 'blind_play', 'virtual_play', 'log_self_play', 'log_debug_play']:
            # next event of defg : (reach2)-tuvw, (reach2)-n, (reach2)-ryuukyoku, agari
            # agari
            for j in range(self.num_player-1):
                jplayer = (iplayer+1+j)%self.num_player
                if self.play_mode not in ['blind_play', 'virtual_play'] or jplayer == self.my_player:
                    if not self.players[jplayer].is_furiten and not self.players[jplayer].is_temp_furiten:
                        agari_availables = self.compute_agari_availables(jplayer, iplayer, thrown_tile, 'ron')
                        if len(agari_availables) > 0:
                            next_processes.append(('agari', jplayer, agari_availables[0]))
                elif self.play_mode == 'virtual_play' and jplayer != self.my_player:
                    next_processes.append(('agari', jplayer, (iplayer, 'ron')))
            # reach2
            if self.players[iplayer].is_ready_to_reach:
                next_processes.append(('reach', iplayer, 2))
                if self.check_reach4():
                    next_processes.append(('ryuukyoku', 'reach4'))
            # kaze4
            if self.check_kaze4(iplayer):
                next_processes.append(('ryuukyoku', 'kaze4'))
            # kan4
            if self.num_kan == 4 and self.check_kan4():
                next_processes.append(('ryuukyoku', 'kan4'))
            # ryuukyoku, nm
            if self.remain_tile_num == 0: # ryuukyoku
                nm_players = self.compute_nm_players()
                if len(nm_players) > 0:
                    ryuukyoku_scores = self.compute_nm_scores(nm_players)
                    next_processes.append(('ryuukyoku', 'nm', ryuukyoku_scores))
                else:
                    if self.play_mode not in ['blind_play', 'virtual_play']:
                        tenpai_players = self.compute_tenpai_players()
                        ryuukyoku_scores = self.compute_ryuukyoku_scores(tenpai_players)
                        next_processes.append(('ryuukyoku', '', ryuukyoku_scores))
                    else:
                        next_processes.append(('ryuukyoku', ''))
            # n
            if self.remain_tile_num > 0:
                for j in range(self.num_player-1):
                    jplayer = (iplayer+1+j)%self.num_player
                    if self.play_mode not in ['blind_play', 'virtual_play'] or jplayer == self.my_player:
                        if not self.players[jplayer].is_reached:
                            pon_availables = self.compute_pon_availables(jplayer, iplayer, thrown_tile)
                            if len(pon_availables) > 0:
                                next_processes.append(('n', 'pon', jplayer, pon_availables))
                                if self.num_kan < 4:
                                    daiminkan_availables = self.compute_daiminkan_availables(
                                        jplayer, iplayer, thrown_tile)
                                    if len(daiminkan_availables) > 0:
                                        next_processes.append(('n', 'daiminkan', jplayer, daiminkan_availables))
                    elif self.play_mode == 'virtual_play' and iplayer == self.my_player:
                        if not self.players[jplayer].is_reached:
                            next_processes.append(('n', 'pon', jplayer, (iplayer, thrown_tile)))
                jplayer = (iplayer+1)%self.num_player
                if self.play_mode not in ['blind_play', 'virtual_play'] or jplayer == self.my_player:
                    if not self.players[jplayer].is_reached:
                        chi_availables = self.compute_chi_availables(jplayer, iplayer, thrown_tile)
                        if len(chi_availables) > 0:
                            next_processes.append(('n', 'chi', jplayer, chi_availables))
                elif self.play_mode == 'virtual_play' and iplayer == self.my_player:
                    if not self.players[jplayer].is_reached:
                        next_processes.append(('n', 'chi', jplayer, (iplayer, thrown_tile)))
            # tuvw
            next_processes.append(('tuvw', (iplayer+1)%self.num_player))

        # set conditions
        self.players[iplayer].last_rinshan = False
        self.players[iplayer].last_tsumoed = False
        self.players[iplayer].last_minkanned = False
        self.players[iplayer].last_chi = -1
        self.players[iplayer].last_pon = -1

        self.players[iplayer].is_first = False
        self.players[iplayer].is_ippatsu = False
        if not (self.players[iplayer].is_reached and not self.players[iplayer].is_ready_to_reach): # player is reached
            if not self.players[iplayer].is_tsumo_giri: # TEST: we assume there is no yamagosi
                del self.players[iplayer].safe_tile_types[:]
        if self.play_mode not in ['blind_play', 'virtual_play'] or iplayer == self.my_player:
            self.players[iplayer].is_temp_furiten = False
            if not self.players[iplayer].is_tsumo_giri:
                self.players[iplayer].is_furiten = self.check_player_is_furiten(iplayer)

        for j in range(self.num_player - 1):
            jplayer = (iplayer+1+j)%self.num_player
            self.players[jplayer].safe_tile_types.append(thrown_tile//4)
            if self.play_mode not in ['blind_play', 'virtual_play'] or jplayer == self.my_player:
                if thrown_tile//4 in self.players[jplayer].waiting_type: # temp furiten
                    if self.players[jplayer].is_reached:
                        self.players[jplayer].is_furiten = True
                    else:
                        self.players[jplayer].is_temp_furiten = True

        return next_processes

    def process_agari(self, event_agaris): # multiple agari can be exist
        # who, from_who, fu, yaku_style, yaku_types, yaku_values, uradora_indicators are used

        if self.play_mode == 'log_play': # data acquire
            # 1. add late hand info
            self._players_late_hand_infos_to_add = defaultdict(dict)
            agari_players = []
            agari_from_who = -1
            for event_agari in event_agaris:
                _, iplayer, (agari_from_who, _, _, _, _, _), _ = event_agari
                #iplayer = event_agari.who
                #agari_from_who = event_agari.from_who
                agari_players.append(iplayer)
                if iplayer == agari_from_who:
                    self._players_late_hand_infos_to_add[iplayer]['tsumo_done'] = True
                else:
                    self._players_late_hand_infos_to_add[iplayer]['ron_done'] = self.agari_tile
            for jplayer in range(self.num_player):
                if (jplayer in self._players_hand_infos_to_add and
                    'chi' in self._players_hand_infos_to_add[jplayer] and
                    jplayer not in agari_players): # remove chi from player
                    assert(jplayer == (agari_from_who+1)%self.num_player)
                    self._players_late_hand_infos_to_add[jplayer]['chi_remove'] = True
                if (jplayer in self._players_hand_infos_to_add and
                    'pon' in self._players_hand_infos_to_add[jplayer] and
                    jplayer not in agari_players): # remove pon from player
                    self._players_late_hand_infos_to_add[jplayer]['pon_remove'] = True
                if (jplayer in self._players_hand_infos_to_add and
                    'daiminkan' in self._players_hand_infos_to_add[jplayer] and
                    jplayer not in agari_players): # remove daiminkan from player
                    self._players_late_hand_infos_to_add[jplayer]['daiminkan_remove'] = True
            for jplayer in self._players_late_hand_infos_to_add:
                self.add_late_hand_info_to_player(jplayer, self._players_late_hand_infos_to_add[jplayer])
            # 2. add hand info

        last_chin = self.oya
        if len(event_agaris) < 3:
            chin_is_agari = False
            for iagari, event_agari in enumerate(event_agaris):
                _, iplayer, (agari_from_who, basic_score, fu,
                    yaku_style, yaku_types, yaku_values), uradora_indicators = event_agari

                if iplayer == self.oya:
                    chin_is_agari = True

                if self.play_mode == 'self_play':
                    assert(basic_score > 0)
                if basic_score > 0:
                    assert(self.play_mode in ['self_play', 'virtual_play'])
                    if self.play_mode != 'virtual_play' or iplayer == self.my_player:
                        self.uradora_indicators = uradora_indicators[:]
                        assert(len(self.uradora_indicators) > 0)
                        self.uradoras = [self.compute_dora_tile_type(tile) for tile in self.uradora_indicators]
                        if self.players[iplayer].is_reached and yaku_style == 'yaku':
                            if iplayer != agari_from_who: # ron / chakan
                                self.players[iplayer].hand.append(self.agari_tile)
                            num_dora, num_uradora, num_akadora = self.compute_player_dora_num(iplayer)
                            del self.players[iplayer].hand[-1]
                            if num_uradora > 0:
                                assert(53 not in yaku_types)
                                yaku_types = yaku_types[:]
                                yaku_values = yaku_values[:]
                                yaku_values.append(num_uradora)
                                yaku_types.append(53)
                                value = self.compute_value(yaku_style, yaku_values)
                                basic_score, _ = self.compute_basic_score(value, fu)
                else:
                    assert(basic_score == -1)
                    value = self.compute_value(yaku_style, yaku_values)
                    basic_score, _ = self.compute_basic_score(value, fu)

                if iagari == 0:
                    round_scores = self.compute_score(iplayer, agari_from_who, basic_score, use_sticks = True)
                else:
                    round_scores = self.compute_score(iplayer, agari_from_who, basic_score, use_sticks = False)

                self.player_scores += round_scores
                self.reach_stick = 0

                self.players[iplayer].is_agari = True
                self.players[agari_from_who].is_agari_by_who = True
                self.players[iplayer].agari_score = int(np.ceil(basic_score * 4))

            if chin_is_agari:
                self.combo += 1
            else:
                self.combo = 0
                self.oya = (self.oya+1)%self.num_player
                self.current_round += 1

            if self.play_mode in ['self_play', 'log_play', 'log_self_play']:
                for iplayer in range(self.num_player):
                    self.add_agari_info(iplayer)

        next_processes = []
        if self.play_mode in ['self_play', 'blind_play', 'virtual_play', 'log_self_play', 'log_debug_play']:
            if len(event_agaris) >= 3:
                assert(len(event_agaris) == 3)
                next_processes.append(('ryuukyoku', 'ron3',))
            elif self.current_round >= self.max_num_round:
                next_processes.append(('owari',))
            elif (self.current_round >= self.num_round and
                np.max(self.player_scores) >= self.target_score and
                (self.combo == 0 or np.argmax(self.player_scores) == last_chin)):
                next_processes.append(('owari',))
            elif (self.current_round == self.num_round-1 and
                np.max(self.player_scores) >= self.target_score and
                (self.combo >= 1 and np.argmax(self.player_scores) == last_chin)):
                next_processes.append(('owari',))
            elif np.min(self.player_scores) < 0:
                next_processes.append(('owari',))
            else:
                next_processes.append(('init',))
            #if next_processes[0][0] == 'owari':
            #    print(self.game_type, self.max_num_round, self.num_round, self.current_round, last_chin, self.combo, self.player_scores)

        return next_processes

    def process_ryuukyoku(self, event_ryuukyoku):
        # only type and round_scores are used
        _, ryuukyoku_type, round_scores, chin_is_tenpai = event_ryuukyoku

        if self.play_mode == 'log_play': # data acquire
            # 1. add late hand info
            self._players_late_hand_infos_to_add = defaultdict(dict)
            if ryuukyoku_type == 'yao9':
                iplayer = self._yao9_player
                self._players_late_hand_infos_to_add[iplayer]['yao9_done'] = True
            for jplayer in self._players_late_hand_infos_to_add:
                self.add_late_hand_info_to_player(jplayer, self._players_late_hand_infos_to_add[jplayer])
            # 2. add hand info

        if ryuukyoku_type == 'nm' or ryuukyoku_type == '':
            assert(self.remain_tile_num == 0)
            self.player_scores += np.array(round_scores)

        self.combo += 1

        last_chin = self.oya
        if self.play_mode in ['self_play', 'log_play', 'log_self_play', 'log_debug_play']:
            chin_is_tenpai = True
            if ryuukyoku_type in ['', 'nm'] and not self.check_player_is_tenpai(last_chin):
                chin_is_tenpai = False

        if not chin_is_tenpai:
            self.oya = (self.oya+1)%self.num_player
            self.current_round += 1

        if self.play_mode in ['self_play', 'log_play', 'log_self_play']:
            for iplayer in range(self.num_player):
                self.add_agari_info(iplayer)

        next_processes = []
        if self.play_mode in ['self_play', 'blind_play', 'virtual_play', 'log_self_play', 'log_debug_play']:
            if ryuukyoku_type not in ['', 'nm']:
                next_processes.append(('init',))
            elif self.current_round >= self.max_num_round:
                next_processes.append(('owari',))
            elif (self.current_round >= self.num_round and
                ((np.max(self.player_scores) >= self.target_score and not chin_is_tenpai) or
                (np.max(self.player_scores) >= self.target_score and chin_is_tenpai and
                np.argmax(self.player_scores) == last_chin))):
                next_processes.append(('owari',))
            elif (self.current_round == self.num_round-1 and
                (np.max(self.player_scores) >= self.target_score and chin_is_tenpai and
                np.argmax(self.player_scores) == last_chin)):
                next_processes.append(('owari',))
            elif np.min(self.player_scores) < 0:
                next_processes.append(('owari',))
            else:
                next_processes.append(('init',))
            #if next_processes[0][0] == 'owari':
            #    print(self.game_type, self.max_num_round, self.num_round, self.current_round, last_chin, self.combo, self.player_scores)

        return next_processes

    # These processes have no next process

    def process_un(self, event_un):
        _, iplayer, names = event_un
        #iplayer = event_un.who

        if len(names) > 0:
            self.player_names = np.array(names)
            for iplayer in range(self.num_player):
                self.players[iplayer].is_connected = True
        else:
            self.players[iplayer].is_connected = True

    def process_bye(self, event_bye):
        _, iplayer = event_bye
        #iplayer = event_bye.who

        self.players[iplayer].is_connected = False

    def process_dora(self, event_dora):
        _, dora_indicator = event_dora

        self.dora_indicators.append(dora_indicator)
        self.doras.append(self.compute_dora_tile_type(self.dora_indicators[-1]))
        self.remain_tile_type_nums[self.dora_indicators[-1]//4] -= 1
        assert(self.remain_tile_type_nums[self.dora_indicators[-1]//4] >= 0)
        # next event : tuvw, defg, dora

    def process_reach(self, event_reach):
        _, iplayer, step = event_reach
        #iplayer = event_reach.who

        if step == 1:
            assert(not self.players[iplayer].is_reached)
            self.players[iplayer].is_ready_to_reach = True
            self.players[iplayer].is_reached = True
            if self.players[iplayer].is_first:
                self.players[iplayer].is_double_reached = True
            # next event : defg
        else:
            assert(step == 2)
            assert(self.players[iplayer].is_reached)
            self.players[iplayer].is_ippatsu = True
            self.player_scores[iplayer] -= 10
            self.reach_stick += 1
            self.players[iplayer].is_ready_to_reach = False
            # next event : tuvw, chi, pon, ryuukyoku, daiminkan

    def compute_agari_availables(self, iplayer, agari_from_who, agari_tile, agari_type):
        agari_availables = []
        if agari_tile//4 not in self.players[iplayer].waiting_type:
            return agari_availables
        if agari_type in ['ron', 'chakan']:
            self.players[iplayer].hand.append(agari_tile)
        value, fu, agari, yaku_style, yaku_values, yaku_types = \
            self.compute_best_agari(iplayer, agari_tile, agari_type)
        if value > 0:
            basic_score, _ = self.compute_basic_score(value, fu)
            #scores = self.compute_score(iplayer, agari_from_who, basic_score, use_sticks = True)
            agari_availables.append((agari_from_who, basic_score, fu, yaku_style, yaku_types, yaku_values))
        if agari_type in ['ron', 'chakan']:
            del self.players[iplayer].hand[-1]
        return agari_availables

    def check_pao(self, iplayer, meld_type, meld_from_who, meld_tiles):
        if iplayer == meld_from_who:
            return False
        if meld_type in ['chakan', 'ankan']:
            return False
        if meld_tiles[0]//4 < 27:
            return False
        kous_first = [kou[0]//4 for kou in self.players[iplayer].pons + self.players[iplayer].ankans +
            self.players[iplayer].daiminkans + self.players[iplayer].chakans]
        if meld_tiles[0]//4 < 31:
            return (27 in kous_first) and (28 in kous_first) and (29 in kous_first) and (30 in kous_first)
        else:
            return (31 in kous_first) and (32 in kous_first) and (33 in kous_first)

    def check_kaze4(self, iplayer):
        if iplayer != (self.oya-1)%self.num_player:
            return False
        #print(iplayer, self.oya, self.current_round, self.players[iplayer].is_first)
        #print(self.players[iplayer].thrown)
        #for j in range(self.num_player-1):
        #    jplayer = (iplayer+1+j)%self.num_player
        #    print(self.players[jplayer].thrown)
        #print('')
        if not self.players[iplayer].is_first:
            return False
        assert(len(self.players[iplayer].thrown) == 1)
        if not (27 <= self.players[iplayer].thrown[0]//4 < 31):
            return False
        for j in range(self.num_player-1):
            jplayer = (iplayer+1+j)%self.num_player
            assert(len(self.players[jplayer].thrown) == 1)
            if self.players[iplayer].thrown[0]//4 != self.players[jplayer].thrown[0]//4:
                return False
        return True

    def check_reach4(self):
        reach4_flag = True
        for iplayer in range(self.num_player):
            if not self.players[iplayer].is_reached:
                reach4_flag = False
                break
        return reach4_flag

    def check_kan4(self):
        num_kan = 0
        for iplayer in range(self.num_player):
            player_num_kan = (len(self.players[iplayer].daiminkans) +
                len(self.players[iplayer].chakans) +
                len(self.players[iplayer].ankans))
            if player_num_kan == 4:
                return False
            num_kan += player_num_kan
        assert(num_kan == 4)
        return num_kan == 4

    def compute_nm_players(self):
        nm_players = []
        for iplayer in range(self.num_player):
            if self.check_nm(iplayer):
                nm_players.append(iplayer)
        return nm_players

    def compute_tenpai_players(self):
        tenpai_players = []
        for iplayer in range(self.num_player):
            if self.check_player_is_tenpai(iplayer):
                tenpai_players.append(iplayer)
        return tenpai_players

    def compute_nm_scores(self, nm_players):
        ryuukyoku_scores = np.zeros((self.num_player,), dtype='int')
        #assert(len(nm_players) == 1)
        chin_player = self.oya
        for iplayer in nm_players:
            nm_scores = np.zeros((self.num_player,), dtype='int')
            is_chin = iplayer == self.oya
            if is_chin:
                nm_scores[:] = -40
                nm_scores[iplayer] = 120
            else:
                nm_scores[:] = -20
                nm_scores[chin_player] = -40
                nm_scores[iplayer] = 80
            ryuukyoku_scores += nm_scores
        return ryuukyoku_scores

    def compute_ryuukyoku_scores(self, tenpai_players):
        ryuukyoku_scores = np.zeros((self.num_player,), dtype='int')
        noten_players = []
        for iplayer in range(self.num_player):
            if iplayer not in tenpai_players:
                noten_players.append(iplayer)
        if len(tenpai_players) > 0 and len(noten_players) > 0:
            ryuukyoku_scores[tenpai_players] = 30 // len(tenpai_players)
            ryuukyoku_scores[noten_players] = - 30 // len(noten_players)
        return ryuukyoku_scores

    def check_nm(self, iplayer):
        if self.players[iplayer].is_called_by_other:
            return False
        thrown = [t//4 for t in self.players[iplayer].thrown]
        tile_type_keep_nums = np.zeros((NUM_TILE_TYPE,), dtype='int32')
        for tile_type in thrown:
            tile_type_keep_nums[tile_type] += 1
        kokushi_list = [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33]
        return np.sum(tile_type_keep_nums[kokushi_list]) == len(thrown)

    def check_yao9(self, iplayer):
        if not self.players[iplayer].is_first:
            return False
        hand = [t//4 for t in self.players[iplayer].hand]
        tile_type_keep_nums = np.zeros((NUM_TILE_TYPE,), dtype='int32')
        for tile_type in hand:
            tile_type_keep_nums[tile_type] += 1
        kokushi_list = [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33]
        return np.sum(tile_type_keep_nums[kokushi_list] > 0) >= 9

    def check_reach_available(self, iplayer):
        if self.remain_tile_num < 4:
            return False
        if self.players[iplayer].is_reached:
        #if self.players[iplayer].is_reached and not self.players[iplayer].is_ready_to_reach:
            return False
        if not self.check_player_is_menzen(iplayer):
            return False
        if self.player_scores[iplayer] < 10:
            return False
        # In fact, we should compute whether player is tenpai, but for faster code, we omitted it.
        #if not self.check_player_is_tenpai(iplayer):
        #    return False
        return True

    def compute_chi_availables(self, iplayer, from_who, tile):
        chi_availables = []
        tile_type = tile//4
        if tile_type >= 27:
            return chi_availables
        hand = np.array([t//4 for t in self.players[iplayer].hand])
        tile_exist = [False, False, False, False]
        if tile_type%9 >= 1 and np.sum(hand == tile_type-1) >= 1:
            tile_exist[1] = True
            if tile_type%9 >= 2 and np.sum(hand == tile_type-2) >= 1:
                tile_exist[0] = True
        if tile_type%9 <= 7 and np.sum(hand == tile_type+1) >= 1:
            tile_exist[2] = True
            if tile_type%9 <= 6 and np.sum(hand == tile_type+2) >= 1:
                tile_exist[3] = True
        meld_from_who = from_who
        if tile_exist[0] and tile_exist[1]:
            for itile1 in np.where(hand == tile_type-2)[0]:
                for itile2 in np.where(hand == tile_type-1)[0]:
                    tile1 = self.players[iplayer].hand[itile1]
                    tile2 = self.players[iplayer].hand[itile2]
                    meld_tiles = [tile1, tile2, tile]
                    meld_called = 2
                    chi_availables.append((meld_from_who, meld_called, meld_tiles))
        if tile_exist[1] and tile_exist[2]:
            for itile1 in np.where(hand == tile_type-1)[0]:
                for itile2 in np.where(hand == tile_type+1)[0]:
                    tile1 = self.players[iplayer].hand[itile1]
                    tile2 = self.players[iplayer].hand[itile2]
                    meld_tiles = [tile1, tile, tile2]
                    meld_called = 1
                    chi_availables.append((meld_from_who, meld_called, meld_tiles))
        if tile_exist[2] and tile_exist[3]:
            for itile1 in np.where(hand == tile_type+1)[0]:
                for itile2 in np.where(hand == tile_type+2)[0]:
                    tile1 = self.players[iplayer].hand[itile1]
                    tile2 = self.players[iplayer].hand[itile2]
                    meld_tiles = [tile, tile1, tile2]
                    meld_called = 0
                    chi_availables.append((meld_from_who, meld_called, meld_tiles))
        return chi_availables

    def compute_pon_availables(self, iplayer, from_who, tile):
        hand = np.array([t//4 for t in self.players[iplayer].hand])
        hand_indices = np.where(hand == tile//4)[0]
        pon_availables = []
        if len(hand_indices) >= 2:
            base = tile // 4
            if len(hand_indices) == 2:
                tile1 = self.players[iplayer].hand[hand_indices[0]]
                tile2 = self.players[iplayer].hand[hand_indices[1]]
                t4s = [t for t in range(4) if t != tile%4 and t != tile1%4 and t != tile2%4]
                assert(len(t4s) == 1)
            else:
                assert(len(hand_indices) == 3)
                t4s = [t for t in range(4) if t != tile%4]
            for t4 in t4s:
                t0, t1, t2 = ((1,2,3),(0,2,3),(0,1,3),(0,1,2))[t4]
                meld_tiles = [t0+4*base, t1+4*base, t2+4*base]
                meld_from_who = from_who
                if tile%4 == t0:
                    meld_called = 0
                elif tile%4 == t1:
                    meld_called = 1
                else:
                    assert(tile%4 == t2)
                    meld_called = 2
                pon_availables.append((meld_from_who, meld_called, meld_tiles))
        return pon_availables

    def compute_daiminkan_availables(self, iplayer, from_who, tile):
        hand = np.array([t//4 for t in self.players[iplayer].hand])
        hand_indices = np.where(hand == tile//4)[0]
        daiminkan_availables = []
        if len(hand_indices) >= 3:
            assert(len(hand_indices) == 3)
            base = tile // 4
            meld_tiles = [4*base, 1+4*base, 2+4*base, 3+4*base]
            meld_from_who = from_who
            meld_called = tile - 4*base
            daiminkan_availables.append((meld_from_who, meld_called, meld_tiles))
        return daiminkan_availables

    def compute_chakan_availables(self, iplayer):
        hand = [t//4 for t in self.players[iplayer].hand]
        chakan_availables = []
        for ipon, pon in enumerate(self.players[iplayer].pons):
            if pon[0]//4 in hand: # available chakan
                hand_index = hand.index(pon[0]//4)
                meld_tiles = pon + [self.players[iplayer].hand[hand_index]]
                meld_from_who = self.players[iplayer].pons_from_who[ipon]
                meld_called = self.players[iplayer].pons_called[ipon]
                chakan_availables.append((meld_from_who, meld_called, meld_tiles))
        return chakan_availables

    def compute_ankan_availables(self, iplayer):
        hand = [t//4 for t in self.players[iplayer].hand]
        tile_type_keep_nums = np.zeros((NUM_TILE_TYPE,), dtype='int32')
        for tile_type in hand:
            tile_type_keep_nums[tile_type] += 1
        ankan_available_bases = np.where(tile_type_keep_nums == 4)[0]
        ankan_availables = []
        for base in ankan_available_bases:
            meld_tiles = [4*base, 1+4*base, 2+4*base, 3+4*base]
            meld_from_who = iplayer
            meld_called = -1
            ankan_availables.append((meld_from_who, meld_called, meld_tiles))
        return ankan_availables

    def compute_reach_ankan_availables(self, iplayer, new_tile):
        assert(self.players[iplayer].is_reached)
        hand = [t//4 for t in self.players[iplayer].hand]
        tile_type_keep_nums = np.zeros((NUM_TILE_TYPE,), dtype='int32')
        for tile_type in hand:
            tile_type_keep_nums[tile_type] += 1
        ankan_availables = []
        if tile_type_keep_nums[new_tile//4] < 4:
            return ankan_availables

        tile_type_keep_nums[new_tile//4] -= 4
        for tile_type in self.players[iplayer].waiting_type:
            tile_type_keep_nums[tile_type] += 1
            if not self.check_agari_from_keep_nums(tile_type_keep_nums):
                return ankan_availables
            tile_type_keep_nums[tile_type] -= 1
        '''
        tile_type_keep_nums[new_tile//4] -= 1
        for tile_type in self.players[iplayer].waiting_type:
            tile_type_keep_nums[tile_type] += 1
            agaris = self.acquire_agaris_from_keep_nums(tile_type_keep_nums, multiple_return=True)
            assert(len(agaris) > 0)
            for agari in agaris:
                heads, ankous, minkous, schunzs = agari
                if new_tile//4 not in [kou[0] for kou in ankous]:
                    return ankan_availables
            tile_type_keep_nums[tile_type] -= 1
        '''

        base = new_tile//4
        meld_tiles = [4*base, 1+4*base, 2+4*base, 3+4*base]
        meld_from_who = iplayer
        meld_called = -1
        ankan_availables.append((meld_from_who, meld_called, meld_tiles))
        return ankan_availables

    def check_player_is_menzen(self, iplayer):
        return (len(self.players[iplayer].chis) == 0 and
            len(self.players[iplayer].pons) == 0 and
            len(self.players[iplayer].daiminkans) == 0 and
            len(self.players[iplayer].chakans) == 0)

    def check_player_is_not_called(self, iplayer):
        return (len(self.players[iplayer].chis) == 0 and
            len(self.players[iplayer].pons) == 0 and
            len(self.players[iplayer].daiminkans) == 0 and
            len(self.players[iplayer].chakans) == 0 and
            len(self.players[iplayer].ankans) == 0)

    def check_player_is_tenpai(self, iplayer):
        return len(self.players[iplayer].waiting_type) > 0

    def check_player_is_furiten(self, iplayer):
        if len(self.players[iplayer].waiting_type) == 0:
            return False
        for tile in self.players[iplayer].thrown:
            if tile//4 in self.players[iplayer].waiting_type:
                return True
        return False

    def remove_meld_tiles_from_player_hand(self, iplayer, meld_type, meld_from_who, meld_called, meld_tiles):
        assert(meld_type in ['chi', 'pon', 'daiminkan', 'chakan', 'ankan'])
        if meld_type != 'chakan':
            for itile, tile in enumerate(meld_tiles):
                if itile != meld_called:
                    assert(tile in self.players[iplayer].hand)
                    tile_index = self.players[iplayer].hand.index(tile)
                    del self.players[iplayer].hand[tile_index]
        else:
            assert(meld_tiles[-1] in self.players[iplayer].hand)
            tile_index = self.players[iplayer].hand.index(meld_tiles[-1])
            del self.players[iplayer].hand[tile_index]

    def add_meld_tiles_to_player(self, iplayer, meld_type, meld_from_who, meld_called, meld_tiles):
        assert(meld_type in ['chi', 'pon', 'daiminkan', 'chakan', 'ankan'])
        if meld_type == 'chakan': # remove pon
            chakan_index = -1
            for ipon, pon in enumerate(self.players[iplayer].pons):
                if pon == meld_tiles[:-1]:
                    chakan_index = ipon
                    break
            assert(chakan_index >= 0)
            assert(meld_from_who == self.players[iplayer].pons_from_who[chakan_index])
            assert(meld_called == self.players[iplayer].pons_called[chakan_index])
            del self.players[iplayer].pons[chakan_index]
            del self.players[iplayer].pons_from_who[chakan_index]
            del self.players[iplayer].pons_called[chakan_index]
        getattr(self.players[iplayer], meld_type + 's').append(meld_tiles)
        if meld_type != 'ankan':
            getattr(self.players[iplayer], meld_type + 's_from_who').append(meld_from_who)
            getattr(self.players[iplayer], meld_type + 's_called').append(meld_called)
        else:
            assert(meld_from_who == iplayer)
            assert(meld_called == -1)
        # cumulative infos
        self.players[iplayer].cumulative_melds.append(meld_tiles)
        self.players[iplayer].cumulative_melds_from_who.append(meld_from_who)
        self.players[iplayer].cumulative_melds_called.append(meld_called)
        self.players[iplayer].cumulative_melds_type.append(meld_type)

    def acquire_tile_info(self, tile, doras, akadoras, round_wind, player_wind, is_safe, num_remain, tile_num=None):
        if tile >= 0:
            tile_type = tile//4
        else: # None tile
            tile_type = NUM_TILE_TYPE
        num_dora_in_tile = 0
        is_previous_dora = False
        is_next_dora = False
        is_previous_previous_dora = False
        is_next_next_dora = False
        for dora in doras:
            if dora == tile_type:
                num_dora_in_tile += 1
            if dora < 27 and dora%9 > 0 and tile_type == dora-1:
                is_previous_dora = True
            if dora < 27 and dora%9 > 1 and tile_type == dora-2:
                is_previous_previous_dora = True
            if dora < 27 and dora%9 < 8 and tile_type == dora+1:
                is_next_dora = True
            if dora < 27 and dora%9 < 7 and tile_type == dora+2:
                is_next_next_dora = True
        is_akadora = tile in akadoras
        is_round_wind = tile_type == round_wind
        is_player_wind = tile_type == player_wind

        # tile_type, num_dora_in_tile, is_round_wind, is_player_wind, is_safe, num_remain
        # Now make to list
        if tile_num is None:
            tile_info = [0] * (1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 3 + 1) # 13, case of thrown
        else:
            tile_info = [0] * (1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 3 + 1 + 4) # 17, case of hand
        assert(tile_type < NUM_TILE_TYPE+1)
        tile_info[0] = tile_type
        tile_info[1] = num_dora_in_tile
        tile_info[2] = int(is_previous_dora)
        tile_info[3] = int(is_next_dora)
        tile_info[4] = int(is_previous_previous_dora)
        tile_info[5] = int(is_next_next_dora)
        tile_info[6] = int(is_akadora)
        tile_info[7] = int(is_round_wind)
        tile_info[8] = int(is_player_wind)
        for j in range(3):
            tile_info[9+j] = int(is_safe[j])
        tile_info[12] = num_remain
        if tile_num is not None:
            assert(tile_num < 4)
            tile_info[13+tile_num] = 1
        return tile_info

    def acquire_player_tile_info(self, iplayer, tile, tile_num=None): # all info can be collected from opponent
        if tile >= 0:
            tile_type = tile//4
        else: # None tile
            tile_type = NUM_TILE_TYPE
        round_wind = self.compute_round_wind_tile_type()
        player_wind = self.compute_player_wind_tile_type(iplayer)
        is_safe = [False] * (self.num_player-1)
        for j in range(self.num_player-1):
            jplayer = (iplayer+1+j)%self.num_player
            is_safe[j] = tile_type in (self.players[jplayer].safe_tile_types +
                [t//4 for t in self.players[jplayer].thrown]) # safe tile throwing
        if tile_type < NUM_TILE_TYPE:
            num_remain = self.remain_tile_type_nums[tile_type]
        else:
            num_remain = 0
        return self.acquire_tile_info(tile, self.doras, self.akadoras,
            round_wind, player_wind, is_safe, num_remain, tile_num)

    def acquire_meld_info(self, meld_tiles, meld_type, doras, akadoras, round_wind, player_wind, meld_num):
        # meld_type is either 'schunz', 'minkou', 'ankan', 'minkan'.
        meld_tiles = np.array(meld_tiles)
        tile_type = meld_tiles[0]//4
        if meld_type == 'schunz':
            meld_index = (tile_type//9) * 7 + (tile_type%9)
            assert(meld_index < 21)
        elif meld_type == 'minkou':
            meld_index = 21 + tile_type
        elif meld_type == 'minkan':
            meld_index = 55 + tile_type
        else:
            assert(meld_type == 'ankan')
            meld_index = 89 + tile_type
        num_dora_in_meld = 0
        for dora in doras:
            num_dora_in_meld += np.sum(meld_tiles//4 == dora)
        num_akadora_in_meld = 0
        for akadora in akadoras:
            num_akadora_in_meld += np.sum(meld_tiles == akadora)
        is_round_wind = tile_type == round_wind
        is_player_wind = tile_type == player_wind

        meld_info = [0] * (1 + 1 + 1 + 1 + 1 + 4) # 9
        meld_info[0] = meld_index
        meld_info[1] = num_dora_in_meld
        meld_info[2] = num_akadora_in_meld
        meld_info[3] = int(is_round_wind)
        meld_info[4] = int(is_player_wind)
        assert(meld_num < 4)
        meld_info[5+meld_num] = 1
        return meld_info

    def acquire_player_meld_info(self, iplayer, meld_tiles, meld_type, meld_num):
        round_wind = self.compute_round_wind_tile_type()
        player_wind = self.compute_player_wind_tile_type(iplayer)
        return self.acquire_meld_info(meld_tiles, meld_type, self.doras, self.akadoras,
            round_wind, player_wind, meld_num)

    def acquire_remain_info(self, iplayer): # these cannot be collected from opponent
        remain_tile_type_nums = self.remain_tile_type_nums[:] # 34
        for tile in self.players[iplayer].hand:
            remain_tile_type_nums[tile//4] -= 1
            assert(remain_tile_type_nums[tile//4] >= 0)
        return remain_tile_type_nums

    def acquire_player_furiten_info(self, iplayer, use_temp_furiten = True): # all info can be collected from opponent
        safe_tile_types = [0] * 34 # 34
        for tile in self.players[iplayer].thrown:
            safe_tile_types[tile//4] = 1
        if use_temp_furiten:
            for tile_type in self.players[iplayer].safe_tile_types:
                safe_tile_types[tile_type] = 1
        return safe_tile_types

    def acquire_round_info(self, iplayer): # fixed in round
        round = self.current_round
        combo = self.combo
        reach_stick = self.reach_stick
        round_wind = self.compute_round_wind_tile_type()
        player_wind = self.compute_player_wind_tile_type(iplayer)
        score_diffs = []
        for j in range(self.num_player-1):
            jplayer = (iplayer+1+j)%self.num_player
            score_diff = self.player_scores[jplayer] - self.player_scores[iplayer]
            score_diffs.append(score_diff)
        remain_tile_num = self.remain_tile_num
        num_throw = len(self.players[iplayer].thrown)
        is_menzen = self.check_player_is_menzen(iplayer)
        is_reached = self.players[iplayer].is_reached
        is_double_reached = self.players[iplayer].is_double_reached
        is_ready_to_reach = self.players[iplayer].is_ready_to_reach

        round_info = [0] * (1 + 1 + 1 + 1 + 1 + 4 + 4 + 3 + 1 + 34 + 1 + 1 + 1 + 1) # 55
        round_info[0] = self.num_round
        round_info[1] = self.max_num_round
        round_info[2] = round
        round_info[3] = combo
        round_info[4] = reach_stick
        assert((0<= (round_wind-27) < 4) and (0 <= (player_wind-27) < 4))
        round_info[5 + (round_wind-27)] = 1
        round_info[9 + (player_wind-27)] = 1
        round_info[13:16] = score_diffs
        round_info[16] = remain_tile_num
        for tile_type in self.doras:
            assert(tile_type < NUM_TILE_TYPE)
            round_info[17+tile_type] += 1
        round_info[51] = num_throw
        round_info[52] = int(is_menzen)
        round_info[53] = int(is_reached) + int(is_double_reached)
        round_info[54] = int(is_ready_to_reach)
        return round_info

    def acquire_player_melds_info(self, iplayer):
        melds_info = [] # 9 * num_mentsu
        tile_nums = defaultdict(int)
        for schunz in self.players[iplayer].chis:
            melds_info.append(self.acquire_player_meld_info(iplayer, schunz, 'schunz', tile_nums[schunz[0]//4]))
            tile_nums[schunz[0]//4] += 1
        for minkou in self.players[iplayer].pons:
            melds_info.append(self.acquire_player_meld_info(iplayer, minkou, 'minkou', 0))
        for minkan in self.players[iplayer].daiminkans + self.players[iplayer].chakans:
            melds_info.append(self.acquire_player_meld_info(iplayer, minkan, 'minkan', 0))
        for ankan in self.players[iplayer].ankans:
            melds_info.append(self.acquire_player_meld_info(iplayer, ankan, 'ankan', 0))
        assert(len(melds_info) <= 4)
        return melds_info

    def acquire_player_cum_melds_info(self, iplayer):
        melds_info = [] # 9 * (2 * num_mentsu)
        tile_nums = defaultdict(int)
        for (imeld, meld), meld_from_who, cum_meld_type in zip(enumerate(self.players[iplayer].cumulative_melds),
            self.players[iplayer].cumulative_melds_from_who, self.players[iplayer].cumulative_melds_type):
            if cum_meld_type == 'chi':
                melds_info.append(self.acquire_player_meld_info(iplayer, meld, 'schunz', tile_nums[meld[0]//4]))
                tile_nums[meld[0]//4] += 1
            elif cum_meld_type == 'pon':
                melds_info.append(self.acquire_player_meld_info(iplayer, meld, 'minkou', 0))
            elif cum_meld_type == 'chakan':
                melds_info.append(self.acquire_player_meld_info(iplayer, meld, 'minkan', 0))
            elif cum_meld_type == 'daiminkan':
                melds_info.append(self.acquire_player_meld_info(iplayer, meld, 'minkan', 0))
            else:
                assert(cum_meld_type == 'ankan')
                melds_info.append(self.acquire_player_meld_info(iplayer, meld, 'ankan', 0))
        assert(len(melds_info) <= 8)
        return melds_info

    def acquire_player_mask_melds_info(self, iplayer):
        mask_melds_info = [0] * len(self.players[iplayer].cumulative_melds) # (2 * num_mentsu)
        pon_dict = {}
        for (imeld, meld), meld_from_who, cum_meld_type in zip(enumerate(self.players[iplayer].cumulative_melds),
            self.players[iplayer].cumulative_melds_from_who, self.players[iplayer].cumulative_melds_type):
            if cum_meld_type == 'pon':
                pon_dict[meld[0]//4] = imeld
            elif cum_meld_type == 'chakan':
                mask_melds_info[pon_dict[meld[0]//4]] = 1
        assert(len(mask_melds_info) <= 8)
        return mask_melds_info

    def acquire_player_basic_hand_info(self, iplayer):
        basic_hand_info = [] # 22 * len(hand)
        tile_nums = defaultdict(int)
        thrown_type = np.array([t//4 for t in self.players[iplayer].thrown])
        for tile in self.players[iplayer].hand:
            tile_info = self.acquire_player_tile_info(iplayer, tile, tile_nums[tile//4])
            tile_nums[tile//4] += 1
            is_furiten = np.sum(thrown_type == tile//4) > 0
            is_previous_furiten = np.sum(np.logical_and.reduce([
                thrown_type < 27, thrown_type%9 > 0, thrown_type-1 == tile//4])) > 0
            is_previous_previous_furiten = np.sum(np.logical_and.reduce([
                thrown_type < 27, thrown_type%9 > 1, thrown_type-2 == tile//4])) > 0
            is_next_furiten = np.sum(np.logical_and.reduce([
                thrown_type < 27, thrown_type%9 < 8, thrown_type+1 == tile//4])) > 0
            is_next_next_furiten = np.sum(np.logical_and.reduce([
                thrown_type < 27, thrown_type%9 < 7, thrown_type+2 == tile//4])) > 0
            tile_info = (tile_info[:-4] + [int(is_furiten), int(is_previous_furiten), int(is_next_furiten),
                int(is_previous_previous_furiten), int(is_next_next_furiten)] + tile_info[-4:])
            basic_hand_info.append(tile_info)
        assert(len(basic_hand_info) <= 14)
        return basic_hand_info

    def acquire_player_mask_hand_info(self, iplayer):
        mask_hand_info = [0] * len(self.players[iplayer].hand)
        for tile in self.players[iplayer].tiles_cannot_be_thrown:
            if tile in self.players[iplayer].hand:
                tile_index = self.players[iplayer].hand.index(tile)
                mask_hand_info[tile_index] = 1
        return mask_hand_info

    def add_hand_info_to_player(self, iplayer, selected_tile=-1, hand_infos_to_add=None):

        HAND_TILE_INFO_DIMENSION = 22
        MELD_INFO_DIMENSION = 9
        ROUND_INFO_DIMENSION = 55
        HAND_INPUT_DIMENSION = 593
        HAND_DATA_DIMENSION = 879

        thrown_tile_index_infos = []
        for j in range(self.num_player):
            jplayer = (iplayer+j)%self.num_player
            thrown_tile_index_infos.append(len(self.players[jplayer].thrown))
        basic_hand_info = self.acquire_player_basic_hand_info(iplayer)
        mask_hand_info = self.acquire_player_mask_hand_info(iplayer)
        melds_info = self.acquire_player_melds_info(iplayer)
        round_info = self.acquire_round_info(iplayer) # 53
        remain_tile_type_nums = self.acquire_remain_info(iplayer)
        #remain_tile_type_nums = self.remain_tile_type_nums[:] # 34, for hand calc, use round remain
        safe_tile_type_infos = [] # 34 * 4
        safe_tile_type_infos.append(self.acquire_player_furiten_info(iplayer, use_temp_furiten=False))
        for j in range(self.num_player-1):
            jplayer = (iplayer+1+j)%self.num_player
            safe_tile_type_infos.append(self.acquire_player_furiten_info(jplayer, use_temp_furiten=True))

        ippatsu_infos = [int(self.players[(iplayer+j)%self.num_player].is_ippatsu) for j in range(self.num_player)]

        if selected_tile >= 0:
            tile_index = self.players[iplayer].hand.index(selected_tile)
        else:
            tile_index = -1

        tsumo_info = -1
        ron_infos = [[-1, -1, -1, -1, -1] for _ in range(14)]
        chi_infos = [[-1, -1, -1, -1] for _ in range(14)]
        pon_infos = [-1] * 14
        chakan_infos = [-1] * 14
        daiminkan_infos = [-1] * 14
        ankan_infos = [-1] * 14
        yao9_info = -1

        mean_score = -1

        reach_infos = [-1] * 14 # added later
        chi_by_other_infos = [[-1, -1, -1] for _ in range(14)] # added_later
        pon_by_other_infos = [[-1, -1, -1] for _ in range(14)] # added_later
        if selected_tile >= 0:
            hand_type = np.array([t//4 for t in self.players[iplayer].hand])
            for index in np.where(hand_type == selected_tile//4)[0]:
                chi_by_other_infos[index] = [0, 0, 0]
                pon_by_other_infos[index] = [0, 0, 0]

        if hand_infos_to_add is not None:
            if 'tsumo' in hand_infos_to_add:
                tsumo_info = 0
            if 'ron' in hand_infos_to_add:
                agari_tile = hand_infos_to_add['ron']
                agari_tile_type = agari_tile//4
                hand_type = np.array([t//4 for t in self.players[iplayer].hand])
                for index in np.where(hand_type == agari_tile_type)[0]:
                    ron_infos[index][2] = 0
                if agari_tile_type < 27 and agari_tile_type%9 > 0:
                    for index in np.where(hand_type == agari_tile_type-1)[0]:
                        ron_infos[index][3] = 0
                if agari_tile_type < 27 and agari_tile_type%9 > 1:
                    for index in np.where(hand_type == agari_tile_type-2)[0]:
                        ron_infos[index][4] = 0
                if agari_tile_type < 27 and agari_tile_type%9 < 8:
                    for index in np.where(hand_type == agari_tile_type+1)[0]:
                        ron_infos[index][1] = 0
                if agari_tile_type < 27 and agari_tile_type%9 < 7:
                    for index in np.where(hand_type == agari_tile_type+2)[0]:
                        ron_infos[index][0] = 0
            if 'chi' in hand_infos_to_add:
                chi_availables = hand_infos_to_add['chi']
                for meld_from_who, meld_called, meld_tiles in chi_availables:
                    if meld_called == 0: # [tile, tile1, tile2]
                        tile1 = meld_tiles[1]
                        tile2 = meld_tiles[2]
                        chi_infos[self.players[iplayer].hand.index(tile1)][1] = 0
                        chi_infos[self.players[iplayer].hand.index(tile2)][0] = 0
                    elif meld_called == 1: # [tile1, tile, tile2]
                        tile1 = meld_tiles[0]
                        tile2 = meld_tiles[2]
                        chi_infos[self.players[iplayer].hand.index(tile1)][2] = 0
                        chi_infos[self.players[iplayer].hand.index(tile2)][1] = 0
                    else:
                        assert(meld_called == 2) # [tile1, tile2, tile]
                        tile1 = meld_tiles[0]
                        tile2 = meld_tiles[1]
                        chi_infos[self.players[iplayer].hand.index(tile1)][3] = 0
                        chi_infos[self.players[iplayer].hand.index(tile2)][2] = 0
            if 'pon' in hand_infos_to_add:
                pon_availables = hand_infos_to_add['pon']
                for meld_from_who, meld_called, meld_tiles in pon_availables:
                    for i in range(3):
                        if i != meld_called:
                            pon_infos[self.players[iplayer].hand.index(meld_tiles[i])] = 0
            if 'chakan' in hand_infos_to_add:
                chakan_availables = hand_infos_to_add['chakan']
                for meld_from_who, meld_called, meld_tiles in chakan_availables:
                    chakan_infos[self.players[iplayer].hand.index(meld_tiles[-1])] = 0
            if 'daiminkan' in hand_infos_to_add:
                daiminkan_availables = hand_infos_to_add['daiminkan']
                assert(len(daiminkan_availables) == 1)
                meld_from_who, meld_called, meld_tiles = daiminkan_availables[0]
                for i in range(4):
                    if i != meld_called:
                        daiminkan_infos[self.players[iplayer].hand.index(meld_tiles[i])] = 0
            if 'ankan' in hand_infos_to_add:
                ankan_availables = hand_infos_to_add['ankan']
                for meld_from_who, meld_called, meld_tiles in ankan_availables:
                    for i in range(4):
                        ankan_infos[self.players[iplayer].hand.index(meld_tiles[i])] = 0
            if 'yao9' in hand_infos_to_add:
                yao9_info = 0
            if 'score' in hand_infos_to_add: # No late info!
                mean_score = hand_infos_to_add['score']

        next_tile = -1 # added later

        connection_info = int(self.players[iplayer].is_connected)

        hand_info = []
        hand_info += thrown_tile_index_infos
        hand_info.append(len(basic_hand_info))
        for it in range(len(basic_hand_info)):
            hand_info += basic_hand_info[it]
        hand_info += [-1] * ((14-len(basic_hand_info))*HAND_TILE_INFO_DIMENSION)
        hand_info.append(len(melds_info))
        for it in range(len(melds_info)):
            hand_info += melds_info[it]
        hand_info += [-1] * ((4-len(melds_info))*MELD_INFO_DIMENSION)
        hand_info += round_info
        hand_info += remain_tile_type_nums
        for j in range(self.num_player):
            hand_info += safe_tile_type_infos[j]
        hand_info += mask_hand_info
        hand_info += [0] * (14-len(mask_hand_info))
        hand_info += ippatsu_infos

        hand_info.append(tile_index)
        hand_info.append(tsumo_info)
        for it in range(14):
            hand_info += ron_infos[it]
        for it in range(14):
            hand_info += chi_infos[it]
        hand_info += pon_infos
        hand_info += chakan_infos
        hand_info += daiminkan_infos
        hand_info += ankan_infos
        hand_info.append(yao9_info)

        hand_info.append(mean_score)
        hand_info += reach_infos
        for it in range(14):
            hand_info += chi_by_other_infos[it]
        for it in range(14):
            hand_info += pon_by_other_infos[it]

        hand_info.append(next_tile)

        hand_info.append(connection_info)

        assert(len(hand_info) == (4 + 1 + 14*HAND_TILE_INFO_DIMENSION + 1 + 4*MELD_INFO_DIMENSION +
            ROUND_INFO_DIMENSION + NUM_TILE_TYPE + 4*NUM_TILE_TYPE + 14 + 4 +
            1 + 1 + 14*5 + 14*4 + 14 + 14 + 14 + 14 + 1 + 1 + 14 + 14*3 + 14*3 + 1 + 1))
        assert(len(hand_info) == HAND_DATA_DIMENSION)
        if np.all(np.array(hand_info[HAND_INPUT_DIMENSION:HAND_DATA_DIMENSION-1]) == -1): # kokushi, not append data
            assert('ron' in hand_infos_to_add and hand_infos_to_add['ron']//4 in
                [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33])
        if hand_infos_to_add is not None and 'score' in hand_infos_to_add:
            self.players[iplayer].hand_infos = (self.players[iplayer].hand_infos[:-1] +
                [hand_info, self.players[iplayer].hand_infos[-1]])
        else:
            self.players[iplayer].hand_infos.append(hand_info)

    def add_late_hand_info_to_player(self, iplayer, late_hand_infos_to_add=None):

        HAND_TILE_INFO_DIMENSION = 22
        MELD_INFO_DIMENSION = 9
        ROUND_INFO_DIMENSION = 55
        HAND_INPUT_DIMENSION = 593
        HAND_DATA_DIMENSION = 879
        assert(HAND_INPUT_DIMENSION == 4 + 1 + 14*HAND_TILE_INFO_DIMENSION + 1 + 4*MELD_INFO_DIMENSION +
            ROUND_INFO_DIMENSION + NUM_TILE_TYPE + 4*NUM_TILE_TYPE + 14 + 4)

        hand_info = self.players[iplayer].hand_infos[-1]
        hand_type = np.array(hand_info[4+1:4+1+14*HAND_TILE_INFO_DIMENSION:HAND_TILE_INFO_DIMENSION])
        assert(len(hand_type) == 14)

        it = HAND_INPUT_DIMENSION
        it += 1 # tile index cannot be changed.
        if 'tsumo_done' in late_hand_infos_to_add:
            assert(hand_info[it] == 0)
            hand_info[it] = 1
        it += 1
        if 'ron_done' in late_hand_infos_to_add:
            agari_tile = late_hand_infos_to_add['ron_done']
            agari_tile_type = agari_tile//4
            for index in np.where(hand_type == agari_tile_type)[0]:
                assert(hand_info[it+index*5+2] == 0)
                hand_info[it+index*5+2] = 1
            if agari_tile_type < 27 and agari_tile_type%9 > 0:
                for index in np.where(hand_type == agari_tile_type-1)[0]:
                    assert(hand_info[it+index*5+3] == 0)
                    hand_info[it+index*5+3] = 1
            if agari_tile_type < 27 and agari_tile_type%9 > 1:
                for index in np.where(hand_type == agari_tile_type-2)[0]:
                    assert(hand_info[it+index*5+4] == 0)
                    hand_info[it+index*5+4] = 1
            if agari_tile_type < 27 and agari_tile_type%9 < 8:
                for index in np.where(hand_type == agari_tile_type+1)[0]:
                    assert(hand_info[it+index*5+1] == 0)
                    hand_info[it+index*5+1] = 1
            if agari_tile_type < 27 and agari_tile_type%9 < 7:
                for index in np.where(hand_type == agari_tile_type+2)[0]:
                    assert(hand_info[it+index*5+0] == 0)
                    hand_info[it+index*5+0] = 1
        it += 14 * 5
        if 'chi_done' in late_hand_infos_to_add:
            meld_from_who, meld_called, meld_tiles = late_hand_infos_to_add['chi_done']
            if meld_called == 0: # [tile, tile1, tile2]
                tile1 = meld_tiles[1]
                tile2 = meld_tiles[2]
                index1 = 1
                index2 = 0
            elif meld_called == 1: # [tile1, tile, tile2]
                tile1 = meld_tiles[0]
                tile2 = meld_tiles[2]
                index1 = 2
                index2 = 1
            else:
                assert(meld_called == 2) # [tile1, tile2, tile]
                tile1 = meld_tiles[0]
                tile2 = meld_tiles[1]
                index1 = 3
                index2 = 2
            for index in np.where(hand_type == tile1//4)[0]:
                assert(hand_info[it+index*4+index1] == 0)
                hand_info[it+index*4+index1] = 1
            for index in np.where(hand_type == tile2//4)[0]:
                assert(hand_info[it+index*4+index2] == 0)
                hand_info[it+index*4+index2] = 1
        if 'chi_remove' in late_hand_infos_to_add:
            hand_info[it:it+14*4] = [-1] * (14*4)
        it += 14 * 4
        if 'pon_done' in late_hand_infos_to_add:
            meld_from_who, meld_called, meld_tiles = late_hand_infos_to_add['pon_done']
            for index in np.where(hand_type == meld_tiles[0]//4)[0]:
                assert(hand_info[it+index] == 0)
                hand_info[it+index] = 1
        if 'pon_remove' in late_hand_infos_to_add:
            hand_info[it:it+14] = [-1] * 14
        it += 14
        if 'chakan_done' in late_hand_infos_to_add:
            meld_from_who, meld_called, meld_tiles = late_hand_infos_to_add['chakan_done']
            for index in np.where(hand_type == meld_tiles[0]//4)[0]:
                assert(hand_info[it+index] == 0)
                hand_info[it+index] = 1
        it += 14
        if 'daiminkan_done' in late_hand_infos_to_add:
            meld_from_who, meld_called, meld_tiles = late_hand_infos_to_add['daiminkan_done']
            for index in np.where(hand_type == meld_tiles[0]//4)[0]:
                assert(hand_info[it+index] == 0)
                hand_info[it+index] = 1
        if 'daiminkan_remove' in late_hand_infos_to_add:
            hand_info[it:it+14] = [-1] * 14
        it += 14
        if 'ankan_done' in late_hand_infos_to_add:
            meld_from_who, meld_called, meld_tiles = late_hand_infos_to_add['ankan_done']
            for index in np.where(hand_type == meld_tiles[0]//4)[0]:
                assert(hand_info[it+index] == 0)
                hand_info[it+index] = 1
        it += 14
        if 'yao9_done' in late_hand_infos_to_add:
            assert(hand_info[it] == 0)
            hand_info[it] = 1
        it += 1
        #if 'score' in late_hand_infos_to_add:
        #    mean_score = late_hand_infos_to_add['score']
        #    assert(hand_info[it] == 0)
        #    hand_info[it] = mean_score
        it += 1 # No late data for score
        if 'reach' in late_hand_infos_to_add:
            selected_tile, do_reach = late_hand_infos_to_add['reach']
            for index in np.where(hand_type == selected_tile//4)[0]:
                assert(hand_info[it+index] == -1)
                hand_info[it+index] = int(do_reach)
        it += 14
        if 'chi_ed' in late_hand_infos_to_add:
            selected_tile, chi_type = late_hand_infos_to_add['chi_ed']
            for index in np.where(hand_type == selected_tile//4)[0]:
                assert(hand_info[it+index*3+chi_type] == 0)
                hand_info[it+index*3+chi_type] = 1
        it += 14*3
        if 'pon_ed' in late_hand_infos_to_add:
            selected_tile, pon_by_who = late_hand_infos_to_add['pon_ed']
            for index in np.where(hand_type == selected_tile//4)[0]:
                j = (pon_by_who - iplayer)%self.num_player - 1
                assert(j >= 0)
                assert(hand_info[it+index*3+j] == 0)
                hand_info[it+index*3+j] = 1
        it += 14*3
        if 'next_tile' in late_hand_infos_to_add:
            next_tile = late_hand_infos_to_add['next_tile']
            assert(hand_info[HAND_INPUT_DIMENSION] >= 0) # selected tile
            assert(next_tile // 4 < NUM_TILE_TYPE)
            hand_info[it] = next_tile // 4
        it += 1
        it += 1 # connection info
        assert(it == HAND_DATA_DIMENSION)

        if np.all(np.array(hand_info[HAND_INPUT_DIMENSION:HAND_DATA_DIMENSION-1]) == -1):
            del self.players[iplayer].hand_infos[-1]

    def add_thrown_info_to_player(self, iplayer, jplayer, thrown_tile):

        THROWN_TILE_INFO_DIMENSION = 15
        MELD_INFO_DIMENSION = 9
        ROUND_INFO_DIMENSION = 55
        THROWN_DATA_DIMENSION = 459

        tile_info = self.acquire_player_tile_info(iplayer, thrown_tile, tile_num=None)
        is_tsumo_giri = self.players[iplayer].is_tsumo_giri
        self.players[iplayer].last_tsumo_giri = is_tsumo_giri
        last_meld = 55
        if self.players[iplayer].last_chi >= 0:
            last_chi = self.players[iplayer].last_chi
            assert(self.players[iplayer].last_pon == -1)
            last_meld = (last_chi//9) * 7 + (last_chi%9)
            assert(last_meld < 21)
        if self.players[iplayer].last_pon >= 0:
            last_pon = self.players[iplayer].last_pon
            assert(self.players[iplayer].last_chi == -1)
            last_meld = 21 + last_pon
            assert(last_meld < 55)
        tile_info.append(int(is_tsumo_giri))
        tile_info.append(last_meld) # 13

        melds_info, mask_melds_info, round_info, safe_tile_type_info, is_tenpai, \
            waiting_type, scores, my_hand, remain_tile_type_infos, opponent_hand = (
            None, None, None, None, None, None, None, None, None, None)
        mask_melds_info = self.acquire_player_mask_melds_info(iplayer)
        if self.play_mode == 'log_play':
            melds_info = self.acquire_player_cum_melds_info(iplayer)
            round_info = self.acquire_round_info(iplayer) # 53
            if is_tsumo_giri:
                safe_tile_type_info = self.acquire_player_furiten_info(iplayer, use_temp_furiten=True)
            else:
                safe_tile_type_info = self.acquire_player_furiten_info(iplayer, use_temp_furiten=False)

            is_tenpai = self.check_player_is_tenpai(iplayer)
            if self.players[iplayer].is_ready_to_reach:
                assert(is_tenpai)
            waiting_type = [0] * NUM_TILE_TYPE # 34
            scores = [-1] * NUM_TILE_TYPE # 34
            for tile_type, score in zip(self.players[iplayer].waiting_type, self.players[iplayer].waiting_score):
                waiting_type[tile_type] = 1
                scores[tile_type] = score

            my_hand = [0] * NUM_TILE_TYPE
            for tile in self.players[iplayer].hand[:]:
                my_hand[tile//4] += 1

            remain_tile_type_infos = [] # 34 * 4
            for kplayer in range(self.num_player):
                remain_tile_type_infos.append(self.acquire_remain_info(kplayer))
            opponent_hand = [0] * NUM_TILE_TYPE
            for tile in self.players[jplayer].hand[:]:
                opponent_hand[tile//4] += 1

        thrown_info = []
        thrown_info += tile_info
        if self.play_mode == 'log_play':
            thrown_info.append(len(melds_info))
            for meld_info in melds_info:
                thrown_info += meld_info
            thrown_info += [0] * ((8-len(melds_info))*MELD_INFO_DIMENSION)
            thrown_info += mask_melds_info
            thrown_info += [0] * (8-len(mask_melds_info))
            thrown_info += round_info
            thrown_info += safe_tile_type_info
            thrown_info.append(int(is_tenpai))
            thrown_info += waiting_type
            thrown_info += scores
            thrown_info += my_hand
            thrown_info.append(jplayer)
            for remain_tile_type_info in remain_tile_type_infos:
                thrown_info += remain_tile_type_info
            assert(len(remain_tile_type_infos) == 4)
            thrown_info += opponent_hand
        else:
            thrown_info.append(len(mask_melds_info))
            thrown_info += mask_melds_info
            thrown_info += [0] * (8-len(mask_melds_info))

        if self.play_mode == 'log_play':
            assert(len(thrown_info) == (THROWN_TILE_INFO_DIMENSION + 1 + 8*MELD_INFO_DIMENSION + 8 +
                ROUND_INFO_DIMENSION + NUM_TILE_TYPE +
                1 + NUM_TILE_TYPE + NUM_TILE_TYPE + NUM_TILE_TYPE +
                1 + 4*NUM_TILE_TYPE + NUM_TILE_TYPE))
            assert(len(thrown_info) == THROWN_DATA_DIMENSION)
        else:
            assert(len(thrown_info) == THROWN_TILE_INFO_DIMENSION + 1 + 8)

        self.players[iplayer].thrown_infos.append(np.copy(thrown_info))

    def add_agari_info(self, iplayer):
        agari_info = []
        agari_info.append(int(self.players[iplayer].is_agari and self.players[iplayer].is_agari_by_who)) # tsumo
        agari_info.append(int(self.players[iplayer].is_agari and not self.players[iplayer].is_agari_by_who)) # ron
        agari_info.append(int(not self.players[iplayer].is_agari and self.players[iplayer].is_agari_by_who)) # lose
        agari_info.append(self.players[iplayer].agari_score) # score
        agari_info.append(len(self.players[iplayer].thrown))

        argsort = np.argsort(-self.player_scores) # argsort[0] is top
        assert(len(np.where(argsort == iplayer)[0]) == 1)
        agari_info.append(np.where(argsort == iplayer)[0][0]) # rank, later added
        assert(len(agari_info) == 6)
        assert(len(self.players[iplayer].agari_infos) == 0)
        self.players[iplayer].agari_infos.append(agari_info)

    def add_tile_to_player(self, iplayer, new_tile):
        self.players[iplayer].hand.append(new_tile)

    def throw_tile_from_player(self, iplayer, thrown_tile):
        reach_available = self.players[iplayer].is_reach_available
        # reach_available = self.check_reach_available(iplayer) # computed before throw.

        # 0. Add hand info
        if self.play_mode == 'log_play': # data acquire
            self.add_hand_info_to_player(iplayer, thrown_tile)

        # 1. Add tile to thrown tiles
        self.players[iplayer].thrown.append(thrown_tile)
        self.players[iplayer].thrown_is_called.append(False)
        self.players[iplayer].thrown_is_reached.append(self.players[iplayer].is_ready_to_reach)
        # TEST for virtual_play is this needed?
        self.remain_tile_type_nums[thrown_tile//4] -= 1
        assert(self.remain_tile_type_nums[thrown_tile//4] >= 0)

        if self.play_mode not in ['blind_play', 'virtual_play'] or iplayer == self.my_player:
            # 2. Remove tile from hand
            assert(thrown_tile in self.players[iplayer].hand)
            tile_index = self.players[iplayer].hand.index(thrown_tile)
            del self.players[iplayer].hand[tile_index]
            self.players[iplayer].hand = sorted(self.players[iplayer].hand) # sort

            # 3. Set waiting tile types
            if (self.players[iplayer].is_reached and
                not self.players[iplayer].is_ready_to_reach): # Player is reached previous
                assert(self.players[iplayer].is_tsumo_giri)
            if not self.players[iplayer].is_tsumo_giri:
                self.acquire_waiting_tile_types(iplayer)
                self.players[iplayer].is_furiten = self.check_player_is_furiten(iplayer)
            if self.players[iplayer].is_ready_to_reach:
                assert(self.check_player_is_tenpai(iplayer))

        if self.play_mode == 'log_play': # data acquire
            # 4. Add mean score info, reach info
            if self.check_player_is_tenpai(iplayer):
                do_reach = self.players[iplayer].is_ready_to_reach
                assert(reach_available or (not do_reach))
                if reach_available:
                    late_hand_infos_to_add = {}
                    late_hand_infos_to_add['reach'] = (thrown_tile, do_reach)
                    self.add_late_hand_info_to_player(iplayer, late_hand_infos_to_add)

                waiting_score_np = np.array(self.players[iplayer].waiting_score)
                if np.all(waiting_score_np == 0):
                    mean_score = 0
                else:
                    mean_score = int(np.round(np.mean(waiting_score_np[waiting_score_np > 0])))
                hand_infos_to_add = {}
                hand_infos_to_add['score'] = mean_score
                self.add_hand_info_to_player(iplayer, hand_infos_to_add=hand_infos_to_add)

        # 5. Add thrown info
        if self.play_mode in ['self_play', 'blind_play', 'virtual_play', 'log_play', 'log_self_play']:
            jplayer = self.players[iplayer].opponent
            self.add_thrown_info_to_player(iplayer, jplayer, thrown_tile)
            #assert(len(self.players[iplayer].thrown_infos) == len(self.players[iplayer].thrown)+1)

    def arrange_player_tiles(self, iplayer):
        self.players[iplayer].hand = sorted(self.players[iplayer].hand)

    def compute_round_wind_tile_type(self):
        return 27 + self.current_round // self.num_player

    def compute_player_wind_tile_type(self, iplayer):
        return 27 + (iplayer - self.oya) % self.num_player

    def compute_dora_tile_type(self, tile):
        tile_type = tile // 4
        if tile_type < 9:
            return (tile_type + 1) % 9
        elif tile_type < 2 * 9:
            return (tile_type + 1) % 9 + 9
        elif tile_type < 3 * 9:
            return (tile_type + 1) % 9 + 2 * 9
        elif tile_type < 31:
            return (tile_type - 26) % 4 + 27
        else:
            return (tile_type - 30) % 3 + 31

    def compute_player_dora_num(self, iplayer):
        hand = np.array([t for t in self.players[iplayer].hand])
        flatten_meld = np.array([t for meld in
            self.players[iplayer].chis + self.players[iplayer].pons +
            self.players[iplayer].daiminkans + self.players[iplayer].chakans +
            self.players[iplayer].ankans for t in meld])
        num_dora = 0
        for dora in self.doras:
            num_dora += np.sum(hand//4 == dora) + np.sum(flatten_meld//4 == dora)
        num_uradora = 0
        for uradora in self.uradoras:
            num_uradora += np.sum(hand//4 == uradora) + np.sum(flatten_meld//4 == uradora)
        num_akadora = 0
        for akadora in self.akadoras:
            num_akadora += np.sum(hand == akadora) + np.sum(flatten_meld == akadora)
        return num_dora, num_uradora, num_akadora

    def compute_value(self, yaku_style, yaku_values):
        if yaku_style == 'yaku':
            return min(int(np.sum(yaku_values)), 13)
        else:
            assert(yaku_style == 'yakuman')
            return int(np.sum(yaku_values))

    def compute_best_agari(self, iplayer, agari_tile, agari_type):
        best_value = 0
        best_value_sum = 0
        best_fu = 0
        best_yaku_style = 'yaku'
        best_agari, best_yaku_values, best_yaku_types = (None, None, None)

        agaris = self.acquire_agaris(iplayer)
        assert(len(agaris) > 0)
        for agari_init in agaris:
            yaku_style_basic, yaku_values_basic, yaku_types_basic = (None, None, None)
            yaku_style_pinfu, yaku_values_pinfu, yaku_types_pinfu = (None, None, None)
            machi_types, machi_indices = self.compute_machi_types(iplayer, agari_tile, agari_init)
            for machi_type, machi_index in zip(machi_types, machi_indices):
                if machi_type == 1 and agari_type in ['ron', 'chakan']: # move ankou to minkou
                    heads, ankous, minkous, schunzs = agari_init
                    ankous = ankous[:]
                    minkous = [ankous[machi_index]]
                    del ankous[machi_index]
                    agari = (heads, ankous, minkous, schunzs)
                else:
                    agari = agari_init

                fu = self.compute_fu(iplayer, agari_type, agari, machi_type)

                if (self.check_player_is_menzen(iplayer) and
                    ((agari_type in ['tsumo', 'rinshan_tsumo'] and fu == 20) or
                    (agari_type in ['ron', 'chakan'] and fu == 30))): # pinfu
                    assert(machi_type == 0)
                    if yaku_style_pinfu is None: # compute value for pinfu
                        yaku_style_pinfu, yaku_values_pinfu, yaku_types_pinfu = self.compute_yakus(iplayer,
                            agari_tile, agari_type, agari, machi_type)
                        assert(yaku_style_pinfu == 'yakuman' or 7 in yaku_types_pinfu)
                    yaku_style, yaku_values, yaku_types = (yaku_style_pinfu, yaku_values_pinfu, yaku_types_pinfu)
                elif machi_type == 1 and agari_type in ['ron', 'chakan']: # move ankou to minkou
                    yaku_style, yaku_values, yaku_types = self.compute_yakus(iplayer,
                        agari_tile, agari_type, agari, machi_type) # should re-compute value
                else: # no effect of machi type
                    if yaku_style_basic is None: # compute value
                        yaku_style_basic, yaku_values_basic, yaku_types_basic = self.compute_yakus(iplayer,
                            agari_tile, agari_type, agari, machi_type)
                    yaku_style, yaku_values, yaku_types = (yaku_style_basic, yaku_values_basic, yaku_types_basic)

                value = self.compute_value(yaku_style, yaku_values)
                if yaku_style == 'yakuman':
                    assert(value >= best_value)
                    if best_yaku_style == 'yaku' or (best_yaku_style == 'yakuman' and
                        (value > best_value or (value == best_value and fu >= best_fu))):
                        best_yaku_style = 'yakuman'
                        best_value, best_fu, best_agari, best_yaku_values, best_yaku_types = \
                            (value, fu, agari, yaku_values, yaku_types)
                elif best_yaku_style == 'yaku':
                    value_sum = np.sum(yaku_values)
                    if (best_yaku_style == 'yaku' and
                        (value_sum > best_value_sum or (value_sum == best_value_sum and fu >= best_fu))):
                        best_value, best_fu, best_agari, best_yaku_values, best_yaku_types = \
                            (value, fu, agari, yaku_values, yaku_types)
                        best_value_sum = np.sum(best_yaku_values)

        return best_value, best_fu, best_agari, best_yaku_style, best_yaku_values, best_yaku_types

    def compute_basic_score(self, value, fu):
        if value == 0: # cannot agari
            basic_score = 0
            limit = -1
        elif value <= 4:
            basic_score = fu * (2 ** (value + 2)) / 100.
            limit = 0
            if basic_score > 20.:
                basic_score = 20.
                limit = 1
        else:
            if value <= 5:
                basic_score = 20.
                limit = 1
            elif value <= 7:
                basic_score = 30.
                limit = 2
            elif value <= 10:
                basic_score = 40.
                limit = 3
            elif value <= 12:
                basic_score = 60.
                limit = 4
            else:
                assert(value%13 == 0)
                basic_score = 80. * (value//13)
                limit = 5
        return basic_score, limit

    def compute_score(self, iplayer, agari_from_who, basic_score, use_sticks):
        # agari_from_who : agari from which player?
        is_chin = iplayer == self.oya
        chin_player = self.oya

        multipliers = np.zeros((self.num_player,), dtype='float')
        if is_chin:
            multipliers[:] = 2.
            multipliers[iplayer] = 0.
        else:
            multipliers[:] = 1.
            multipliers[chin_player] = 2.
            multipliers[iplayer] = 0.
        responsibility_players = set()
        if agari_from_who != iplayer:
            responsibility_players.add(agari_from_who)
        if self.players[iplayer].pao_who >= 0:
            responsibility_players.add(self.players[iplayer].pao_who)
        if len(responsibility_players) > 0:
            assert(len(responsibility_players) <= 2)
            s = np.sum(multipliers)
            multipliers[:] = 0.
            for jplayer in responsibility_players:
                multipliers[jplayer] = s // len(responsibility_players)
        scores = np.ceil(basic_score * multipliers).astype('int')

        if use_sticks:
            combo_scores = np.zeros((self.num_player,), dtype='int')
            combo_scores[:] = self.combo
            combo_scores[iplayer] = 0
            if self.players[iplayer].pao_who >= 0:
                s = np.sum(combo_scores)
                combo_scores[:] = 0
                combo_scores[self.players[iplayer].pao_who] = s
            elif agari_from_who != iplayer:
                s = np.sum(combo_scores)
                combo_scores[:] = 0
                combo_scores[agari_from_who] = s
            scores += combo_scores

        scores = -scores
        scores[iplayer] = -np.sum(scores)
        if use_sticks:
            scores[iplayer] += 10 * self.reach_stick

        return scores

    def compute_machi_types(self, iplayer, agari_tile, agari):
        # machi_type : ryanmen, shanpon, penchan, kanchan, tanki
        # machi index means which shape is filled.

        agari_tile_type = agari_tile // 4

        machi_types = []
        machi_indices = []

        heads, ankous, minkous, schunzs = agari
        assert(len(minkous) == 0)

        if len(heads) == 7 or len(heads) == 13:
            machi_types.append(4)
            machi_indices.append(0)
            return machi_types, machi_indices

        if heads[0][0] == agari_tile_type:
            machi_types.append(4)
            machi_indices.append(0)

        for iankou, ankou in enumerate(ankous):
            if ankou[0] == agari_tile_type:
                machi_types.append(1)
                machi_indices.append(iankou)
                # In this case with agari_type == 'ron', we must temporarily remove ankou.

        for ischunz, schunz in enumerate(schunzs):
            if schunz[0] == agari_tile_type:
                if agari_tile_type % 9 != 6:
                    machi_types.append(0)
                    machi_indices.append(ischunz)
                else:
                    machi_types.append(2)
                    machi_indices.append(ischunz)
            elif schunz[2] == agari_tile_type:
                if agari_tile_type % 9 != 2:
                    machi_types.append(0)
                    machi_indices.append(ischunz)
                else:
                    machi_types.append(2)
                    machi_indices.append(ischunz)
            elif schunz[1] == agari_tile_type:
                machi_types.append(3)
                machi_indices.append(ischunz)

        return machi_types, machi_indices

    def compute_fu(self, iplayer, agari_type, agari, machi_type):
        # agari_type = 'ron', 'tsumo', 'rinshan_tsumo', or 'chakan'
        # machi_type : ryanmen, shanpon, penchan, kanchan, tanki
        pons = [[t//4 for t in pon] for pon in self.players[iplayer].pons]
        minkans = [[t//4 for t in minkan] for minkan in
            self.players[iplayer].daiminkans + self.players[iplayer].chakans]
        ankans = [[t//4 for t in ankan] for ankan in self.players[iplayer].ankans]

        heads, ankous, minkous, schunzs = agari # schunzs not used
        minkous = minkous + pons

        round_wind_tile_type = self.compute_round_wind_tile_type()
        player_wind_tile_type = self.compute_player_wind_tile_type(iplayer)

        is_menzen = self.check_player_is_menzen(iplayer)

        fu = 20
        if len(heads) == 13: # kokushi
            fu = 0
            return fu
        if len(heads) == 7: # chiitoi
            fu = 25
            return fu
        if machi_type >= 2:
            fu += 2

        if heads[0][0] >= 31:
            fu += 2
        if heads[0][0] == round_wind_tile_type:
            fu += 2
        if heads[0][0] == player_wind_tile_type:
            fu += 2

        for minkou in minkous:
            tile_type = minkou[0]
            if tile_type < 3 * 9 and tile_type % 9 >= 1 and tile_type % 9 <= 7:
                fu += 2
            else:
                fu += 4
        for ankou in ankous:
            tile_type = ankou[0]
            if tile_type < 3 * 9 and tile_type % 9 >= 1 and tile_type % 9 <= 7:
                fu += 4
            else:
                fu += 8
        for minkan in minkans:
            tile_type = minkan[0]
            if tile_type < 3 * 9 and tile_type % 9 >= 1 and tile_type % 9 <= 7:
                fu += 8
            else:
                fu += 16
        for ankan in ankans:
            tile_type = ankan[0]
            if tile_type < 3 * 9 and tile_type % 9 >= 1 and tile_type % 9 <= 7:
                fu += 16
            else:
                fu += 32

        if agari_type in ['tsumo', 'rinshan_tsumo'] and fu > 20: # pinfu, case of tenhou?
            fu += 2
        elif agari_type in ['ron', 'chakan'] and is_menzen: # menzen ron
            fu += 10

        fu = int(10 * np.ceil(float(fu) / 10.))

        if not is_menzen and fu == 20: # kuipin
            fu = 30
        return fu

    def compute_yakus(self, iplayer, agari_tile, agari_type, agari, machi_type):
        '''
        # value 1
        'mentsumo', 'riichi', 'ippatsu', 'chakan', 'rinshan kaihou',
        'haitei raoyue', 'houtei raoyui', 'pinfu', 'tanyao', 'iipeiko',
        # seat winds
        'ton', 'nan', 'xia', 'pei',
        # round winds
        'ton', 'nan', 'xia', 'pei', 'haku', 'hatsu', 'chun',
        # value 2
        'daburu riichi', 'chiitoitsu', 'chanta', 'ittsu', 'sanshoku doujun', 'sanshoku doukou',
        'sankantsu', 'toitoi', 'sanankou', 'shousangen', 'honroutou',
        # value 3
        'ryanpeikou', 'junchan', 'honitsu',
        # value 6
        'chinitsu',
        # mangan
        'renhou',
        # yakuman
        'tenhou', 'chihou', 'daisangen', 'suuankou', 'suuankou tanki', 'tsuuiisou', 'ryuuiisou',
        'chinroutou', 'chuuren pouto', 'chuuren pouto 9-wait', 'kokushi musou', 'kokushi musou 13-wait',
        'daisuushi', 'shousuushi', 'suukantsu',
        # dora
        'dora', 'uradora', 'akadora',
        '''

        #hand = [t//4 for t in self.players[iplayer].hand]
        chis = [[t//4 for t in chi] for chi in self.players[iplayer].chis]
        pons = [[t//4 for t in pon] for pon in self.players[iplayer].pons]
        minkans = [[t//4 for t in minkan] for minkan in
            self.players[iplayer].daiminkans + self.players[iplayer].chakans]
        ankans = [[t//4 for t in ankan] for ankan in self.players[iplayer].ankans]

        heads, ankous, minkous, schunzs = agari
        schunzs = schunzs + chis
        minkous = minkous + pons

        heads_first = [head[0] for head in heads]
        schunzs_first = sorted([schunz[0] for schunz in schunzs])
        kous_first = [kou[0] for kou in minkous + ankous]
        kans_first = [kan[0] for kan in minkans + ankans]

        tile_type_keep_nums = np.zeros((NUM_TILE_TYPE,), dtype='int32')
        if len(heads) != 13: # kokushi is exception
            for tile_type in heads_first:
                tile_type_keep_nums[tile_type] += 2
            for tile_type in schunzs_first:
                tile_type_keep_nums[tile_type:tile_type+3] += 1
            for tile_type in kous_first:
                tile_type_keep_nums[tile_type] += 3
            for tile_type in kans_first:
                tile_type_keep_nums[tile_type] += 4
        else:
            for head in heads:
                tile_type_keep_nums[head[0]] += len(head)
        player_tile_num = np.sum(tile_type_keep_nums)

        # Last tile locates at the end.
        agari_tile_type = agari_tile // 4

        is_tsumo = agari_type in ['tsumo', 'rinshan_tsumo']
        is_rinshan = agari_type == 'rinshan_tsumo'
        is_chakan = agari_type == 'chakan'
        is_chin = iplayer == self.oya
        is_first = self.players[iplayer].is_first
        is_menzen = self.check_player_is_menzen(iplayer)
        is_not_called = self.check_player_is_not_called(iplayer)
        is_reached = self.players[iplayer].is_reached
        is_double_reached = self.players[iplayer].is_double_reached
        is_ippatsu = self.players[iplayer].is_ippatsu
        is_haitei = self.remain_tile_num == 0

        round_wind_tile_type = self.compute_round_wind_tile_type()
        player_wind_tile_type = self.compute_player_wind_tile_type(iplayer)

        yaku_values = []
        yaku_types = []

        # yakuman
        #'tenhou', 'chihou', 'daisangen', 'suuankou', 'suuankou tanki', 'tsuuiisou', 'ryuuiisou',
        #'chinroutou', 'chuuren pouto', 'chuuren pouto 9-wait', 'kokushi musou', 'kokushi musou 13-wait',
        #'daisuushi', 'shousuushi', 'suukantsu',

        if is_chin and is_first and is_tsumo: # tenhou
            yaku_values.append(13)
            yaku_types.append(37)

        if not is_chin and is_first and is_tsumo: # chihou
            yaku_values.append(13)
            yaku_types.append(38)

        if tile_type_keep_nums[31] >= 3 and tile_type_keep_nums[32] >= 3 and tile_type_keep_nums[33] >= 3: # daisangen
            yaku_values.append(13)
            yaku_types.append(39)

        if len(ankous) + len(ankans) == 4: #suuankou
            if tile_type_keep_nums[agari_tile_type] == 2: # suuankou tanki
                yaku_values.append(13) # not double yakuman in tenhou
                yaku_types.append(41)
            else:
                yaku_values.append(13) # suuankou
                yaku_types.append(40)

        if np.sum(tile_type_keep_nums[27:34]) == player_tile_num: # tsuuiisou
            yaku_values.append(13)
            yaku_types.append(42)

        if (tile_type_keep_nums[19] + tile_type_keep_nums[20] + tile_type_keep_nums[21] + tile_type_keep_nums[23] +
            tile_type_keep_nums[25] + tile_type_keep_nums[32] == player_tile_num): # ryuuiisou, allow hatsu
            yaku_values.append(13)
            yaku_types.append(43)

        if (tile_type_keep_nums[0] + tile_type_keep_nums[8] + tile_type_keep_nums[9] + tile_type_keep_nums[17] +
            tile_type_keep_nums[18] + tile_type_keep_nums[26] == player_tile_num): # chinroutou
            yaku_values.append(13)
            yaku_types.append(44)

        for it in range(3):
            if (is_not_called and np.all(tile_type_keep_nums[9*it+1:9*(it+1)-1] >= 1) and
                tile_type_keep_nums[9*it] >= 3 and tile_type_keep_nums[9*(it+1)-1] >= 3):
                if ( ( (agari_tile_type == 9*it or agari_tile_type == 9*(it+1)-1) and tile_type_keep_nums[agari_tile_type] == 4) or
                    ( (9*it+1 <= agari_tile_type < 9*(it+1)-1) and tile_type_keep_nums[agari_tile_type] == 2) ): # chuuren pouto 9-wait
                    yaku_values.append(13) # not double yakuman in tenhou
                    yaku_types.append(46)
                else: # chuuren pouto
                    yaku_values.append(13)
                    yaku_types.append(45)

        if len(heads) == 13:
            if tile_type_keep_nums[agari_tile_type] == 2: # kokushi musou 13-wait
                yaku_values.append(13) # not double yakuman in tenhou
                yaku_types.append(48)
            else: # kokushi musou
                yaku_values.append(13)
                yaku_types.append(47)

        if np.all(tile_type_keep_nums[27:31] >= 3): # daisuushi
            yaku_values.append(13) # not double yakuman in tenhou
            yaku_types.append(49)
        elif len(heads) == 1 and np.all(tile_type_keep_nums[27:31] >= 2): # shousuushi
            yaku_values.append(13)
            yaku_types.append(50)

        if len(ankans) + len(minkans) == 4: # suukantsu
            yaku_values.append(13)
            yaku_types.append(51)

        if len(yaku_values) > 0: # yakuman
            # print 'value : ', yaku_values, yaku_types
            yaku_style = 'yakuman'
            return yaku_style, yaku_values, yaku_types

        # value 1
        #'mentsumo', 'riichi', 'ippatsu', 'chakan', 'rinshan kaihou',
        #'haitei raoyue', 'houtei raoyui', 'pinfu', 'tanyao', 'iipeiko',

        if is_menzen and is_tsumo: # mentsumo
            yaku_values.append(1)
            yaku_types.append(0)

        if is_reached and not is_double_reached: # riichi
            yaku_values.append(1)
            yaku_types.append(1)

        if is_reached and is_ippatsu: # ippatsu
            yaku_values.append(1)
            yaku_types.append(2)

        if is_chakan: # chakan
            yaku_values.append(1)
            yaku_types.append(3)

        if is_rinshan: # rinshan kaihou
            yaku_values.append(1)
            yaku_types.append(4)

        if is_haitei and is_tsumo and not is_rinshan: # haitei raoyue
            yaku_values.append(1)
            yaku_types.append(5)

        if is_haitei and not is_tsumo: # haitei raoyui
            yaku_values.append(1)
            yaku_types.append(6)

        if (is_menzen and len(schunzs) == 4 and
            heads[0][0] <= 30 and
            heads[0][0] != round_wind_tile_type and
            heads[0][0] != player_wind_tile_type and
            machi_type == 0): # pinfu
            yaku_values.append(1)
            yaku_types.append(7)

        if (np.sum(tile_type_keep_nums[1:8]) + np.sum(tile_type_keep_nums[10:17]) +
            np.sum(tile_type_keep_nums[19:26]) == player_tile_num): # tanyao
            if self.allow_kuitan or is_menzen:
                yaku_values.append(1)
                yaku_types.append(8)

        # iipeiko for later

        # seat winds
        #'ton', 'nan', 'xia', 'pei',
        # round winds
        #'ton', 'nan', 'xia', 'pei', 'haku', 'hatsu', 'chun',

        for tile_type in kous_first + kans_first:
            if tile_type == player_wind_tile_type: # seat winds
                yaku_values.append(1)
                yaku_types.append(10 + tile_type - 27)
            if tile_type == round_wind_tile_type: # round winds
                yaku_values.append(1)
                yaku_types.append(14 + tile_type - 27)
            if tile_type >= 31: # dragon tiles
                yaku_values.append(1)
                yaku_types.append(18 + tile_type - 31)

        # value 2
        #'daburu riichi', 'chiitoitsu', 'chanta', 'ittsu', 'sanshoku doujun', 'sanshoku doukou',
        #'sankantsu', 'toitoi', 'sanankou', 'shousangen', 'honroutou',

        if is_double_reached: # daburu riichi
            yaku_values.append(2)
            yaku_types.append(21)

        if len(heads) == 7: # chiitoitsu
            yaku_values.append(2)
            yaku_types.append(22)

        # chanta for later

        for it in range(3):
            if (9 * it in schunzs_first and 9 * it + 3 in schunzs_first and 9 * it + 6 in schunzs_first): #ittsu
                if is_menzen:
                    yaku_values.append(2)
                    yaku_types.append(24)
                else:
                    yaku_values.append(1)
                    yaku_types.append(24)

        for i in range(7):
            if (i in schunzs_first and i + 9 in schunzs_first and i + 2 * 9 in schunzs_first): # sanshoku doujun
                if is_menzen:
                    yaku_values.append(2)
                    yaku_types.append(25)
                else:
                    yaku_values.append(1)
                    yaku_types.append(25)

        for i in range(9):
            if (i in kous_first + kans_first and i + 9 in kous_first + kans_first and
                i + 2 * 9 in kous_first + kans_first): # sanshoku doukou
                yaku_values.append(2)
                yaku_types.append(26)

        if len(ankans) + len(minkans) == 3: # sankantsu
            yaku_values.append(2)
            yaku_types.append(27)

        if len(ankous) + len(minkous) + len(ankans) + len(minkans) == 4: # toitoi
            yaku_values.append(2)
            yaku_types.append(28)

        if len(ankous) + len(ankans) == 3: # sanankou
            yaku_values.append(2)
            yaku_types.append(29)

        if len(heads) == 1 and np.all(tile_type_keep_nums[31:34] >= 2): # shousangen
            yaku_values.append(2)
            yaku_types.append(30)

        honroutou_flag = False
        if (tile_type_keep_nums[0] + tile_type_keep_nums[8] + tile_type_keep_nums[9] +
            tile_type_keep_nums[17] + tile_type_keep_nums[18] +
            tile_type_keep_nums[26] + sum(tile_type_keep_nums[27:34]) == player_tile_num): # honroutou
            honroutou_flag = True
            yaku_values.append(2)
            yaku_types.append(31)

        # value 3
        #'ryanpeikou', 'junchan', 'honitsu',

        ryanpeikou_flag = False
        if is_menzen and (len(schunzs) == 4 and
            schunzs_first[0] == schunzs_first[1] and schunzs_first[2] == schunzs_first[3]): # ryanpeikou
            ryanpeikou_flag = True
            yaku_values.append(3)
            yaku_types.append(32)

        if is_menzen and (
            (len(schunzs) >= 2 and schunzs_first[0] == schunzs_first[1]) or
            (len(schunzs) >= 3 and schunzs_first[1] == schunzs_first[2]) or
            (len(schunzs) == 4 and schunzs_first[2] == schunzs_first[3])):
            if not ryanpeikou_flag: #iipeiko
                yaku_values.append(1)
                yaku_types.append(9)

        junchan_flag = True
        chanta_flag = True
        for i in heads_first:
            if i >= 27 or (i % 9 != 0 and i % 9 != 8): junchan_flag = False
            if i < 27 and (i % 9 != 0 and i % 9 != 8): chanta_flag = False
        for i in schunzs_first:
            if i % 9 != 0 and i % 9 != 6:
                junchan_flag = False
                chanta_flag = False
        for i in kous_first + kans_first:
            if i >= 27 or (i % 9 != 0 and i % 9 != 8): junchan_flag = False
            if i < 27 and (i % 9 != 0 and i % 9 != 8): chanta_flag = False

        if junchan_flag: # junchan
            if is_menzen:
                yaku_values.append(3)
                yaku_types.append(33)
            else:
                yaku_values.append(2)
                yaku_types.append(33)

        if chanta_flag and not honroutou_flag and not junchan_flag: # chanta
            if is_menzen:
                yaku_values.append(2)
                yaku_types.append(23)
            else:
                yaku_values.append(1)
                yaku_types.append(23)

        # honitsu for later

        # value 6
        #'chinitsu',

        if (np.sum(tile_type_keep_nums[0:9]) == player_tile_num or
            np.sum(tile_type_keep_nums[9:18]) == player_tile_num or
            np.sum(tile_type_keep_nums[18:27]) == player_tile_num):
            if is_menzen:
                yaku_values.append(6)
                yaku_types.append(35)
            else:
                yaku_values.append(5)
                yaku_types.append(35)

        elif (np.sum(tile_type_keep_nums[0:9]) + np.sum(tile_type_keep_nums[27:34]) == player_tile_num or
            np.sum(tile_type_keep_nums[9:18]) + np.sum(tile_type_keep_nums[27:34]) == player_tile_num or
            np.sum(tile_type_keep_nums[18:27]) + np.sum(tile_type_keep_nums[27:34]) == player_tile_num): # honitsu
            if is_menzen:
                yaku_values.append(3)
                yaku_types.append(34)
            else:
                yaku_values.append(2)
                yaku_types.append(34)

        # 満貫
        #'renhou',
        # renhou not allowed in tenhou

        # dora
        #'dora', 'uradora', 'akadora',

        if len(yaku_values) > 0:
            num_dora, num_uradora, num_akadora = self.compute_player_dora_num(iplayer)
            if num_dora > 0: # dora
                yaku_values.append(num_dora)
                yaku_types.append(52)
            if is_reached and num_uradora > 0:
                yaku_values.append(num_uradora)
                yaku_types.append(53)
            if num_akadora > 0: # akadora
                yaku_values.append(num_akadora)
                yaku_types.append(54)

        # print 'yakus : ', yaku_values, yaku_types

        yaku_style = 'yaku'
        return yaku_style, yaku_values, yaku_types

    def compute_player_is_tenpai(self, iplayer):
        return self.compute_hand_is_tenpai(self.players[iplayer].hand)

    def compute_hand_is_tenpai(self, hand):
        tile_type_keep_nums = np.zeros((NUM_TILE_TYPE,), dtype='int32')
        for tile in hand:
            tile_type_keep_nums[tile//4] += 1
        waiting_type = self.acquire_waiting_tile_types_from_keep_nums(
            tile_type_keep_nums, multiple_return=False)
        return len(waiting_type) > 0

    def acquire_waiting_tile_types(self, iplayer):
        hand = [t//4 for t in self.players[iplayer].hand]
        tile_type_keep_nums = np.zeros((NUM_TILE_TYPE,), dtype='int32')
        for tile_type in hand:
            tile_type_keep_nums[tile_type] += 1
        self.players[iplayer].waiting_type = self.acquire_waiting_tile_types_from_keep_nums(
            tile_type_keep_nums, multiple_return=True)

        self.players[iplayer].waiting_score = []
        self.players[iplayer].waiting_limit = []
        for tile_type in self.players[iplayer].waiting_type:
            agari_type = 'ron'
            agari_tile = tile_type*4+1 # avoid akadora
            self.players[iplayer].hand.append(agari_tile) # temporal hand
            value, fu, agari, yaku_style, yaku_values, yaku_types = \
                self.compute_best_agari(iplayer, agari_tile, agari_type)
            basic_score, limit = self.compute_basic_score(value, fu)
            score = int(np.ceil(basic_score * 4)) # compute ron score
            del self.players[iplayer].hand[-1]
            self.players[iplayer].waiting_score.append(score)
            self.players[iplayer].waiting_limit.append(limit)

        return self.players[iplayer].waiting_type

    def acquire_waiting_tile_types_from_keep_nums(self, tile_type_keep_nums, multiple_return=True):
        return list(acquire_waiting_tile_types_from_keep_nums_cython(tile_type_keep_nums, multiple_return))

    def check_agari_from_keep_nums(self, tile_type_keep_nums):
        return check_agari_from_keep_nums_cython(tile_type_keep_nums)

    def acquire_agaris(self, iplayer):
        hand = [t//4 for t in self.players[iplayer].hand]
        # This function is called only for tsumo and ron case.
        # Check agari
        tile_type_keep_nums = np.zeros((NUM_TILE_TYPE,), dtype='int32')
        for tile_type in hand:
            tile_type_keep_nums[tile_type] += 1
        agaris = self.acquire_agaris_from_keep_nums(tile_type_keep_nums)
        #if len(agaris) == 0:
        #    print(self.players[iplayer].hand)
        #    print(self.players[iplayer].waiting_type)
        assert(len(agaris) > 0)
        return agaris

    def acquire_agaris_from_keep_nums(self, tile_type_keep_nums, multiple_return = True):
        tile_type_keep_nums = np.copy(tile_type_keep_nums)
        agaris = []

        minkous = [] # temp minkous
        # 1. Check for kokushi, chiitoi
        if np.sum(tile_type_keep_nums) == NUM_STARTING_HAND_TILE + 1: # No call
            # chiitoi
            flag = True
            heads = []
            for i, tile_type_keep_num in enumerate(tile_type_keep_nums):
                if tile_type_keep_num != 0 and tile_type_keep_num != 2:
                    flag = False
                    break
                elif tile_type_keep_num == 2:
                    heads.append([i, i])
            if flag:
                ankous = []
                schunzs = []
                agaris.append((heads, ankous, minkous, schunzs))

            # kokushi
            kokushi_list = [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33]
            heads = []
            flag = True
            for i in kokushi_list:
                if tile_type_keep_nums[i] == 0:
                    flag = False
                    break
                elif tile_type_keep_nums[i] == 2:
                    heads.append([i, i])
                elif tile_type_keep_nums[i] == 1:
                    heads.append([i])
                else:
                    flag = False
                    break
            if flag and len(heads) == 13:
                ankous = []
                schunzs = []
                agaris.append((heads, ankous, minkous, schunzs))
                return agaris # No more agaris can exist

        # 2. Find candidates of head, ankous.
        head_must = None
        ankous_must = []
        for it in range(3):
            tile_walls = np.ones((13,), dtype = 'int')
            tile_walls[2:11] = (tile_type_keep_nums[9*it:9*(it+1)] == 0).astype('int')
            for i in range(9*it, 9*(it+1)):
                if (tile_type_keep_nums[i] > 0 and
                    np.sum(tile_walls[i-9*it:i-9*it+3]) > 0 and np.sum(tile_walls[i-9*it+1:i-9*it+4]) > 0 and
                    np.sum(tile_walls[i-9*it+2:i-9*it+5]) > 0): # No schunz can be made.
                    if tile_type_keep_nums[i] == 1 or tile_type_keep_nums[i] >= 4:
                        return agaris
                    elif tile_type_keep_nums[i] == 2:
                        if head_must is not None:
                            return agaris
                        head_must = [i, i]
                    elif tile_type_keep_nums[i] == 3:
                        ankous_must.append([i, i, i])
        for i in range(27, NUM_TILE_TYPE):
            if tile_type_keep_nums[i] > 0:
                if tile_type_keep_nums[i] == 1 or tile_type_keep_nums[i] >= 4:
                    return agaris
                elif tile_type_keep_nums[i] == 2:
                    if head_must is not None:
                        return agaris
                    head_must = [i, i]
                elif tile_type_keep_nums[i] == 3:
                    ankous_must.append([i, i, i])

        # 3. Process for confirmed head, ankous.
        ankous_basic = []
        for ankou in ankous_must:
            ankous_basic.append(ankou)
            tile_type_keep_nums[ankou[0]] -= 3
        if head_must is not None:
            heads_available = [head_must]
        else:
            heads_available = [[i, i] for i in range(NUM_TILE_TYPE) if tile_type_keep_nums[i] >= 2]

        # 4. Process for other candidates.
        for head in heads_available:
            heads = [head]
            tile_type_keep_nums[head[0]] -= 2
            assert(tile_type_keep_nums[head[0]] >= 0)

            ankous_available = [[i,i,i] for i in range(NUM_TILE_TYPE) if tile_type_keep_nums[i] >= 3]
            for iankou in range(2**len(ankous_available)):
                flag = True
                ankous = ankous_basic[:]
                tile_type_keep_nums_temp = np.copy(tile_type_keep_nums)
                for i, ankou in enumerate(ankous_available):
                    if (iankou // 2**i) % 2 == 0:
                        ankous.append(ankou)
                        tile_type_keep_nums_temp[ankou[0]] -= 3
                        assert(tile_type_keep_nums_temp[ankou[0]] >= 0)

                schunzs = []
                for it in range(3):
                    idx = 9 * it
                    while idx < 9 * (it + 1) - 2:
                        if tile_type_keep_nums_temp[idx] == 0: idx += 1
                        elif tile_type_keep_nums_temp[idx + 1] == 0 or tile_type_keep_nums_temp[idx + 2] == 0:
                            break
                        else:
                            schunz = [idx, idx + 1, idx + 2]
                            schunzs.append(schunz)
                            tile_type_keep_nums_temp[schunz[0]] -= 1
                            tile_type_keep_nums_temp[schunz[1]] -= 1
                            tile_type_keep_nums_temp[schunz[2]] -= 1
                    if np.sum(tile_type_keep_nums_temp[9*it : 9*(it+1)]) > 0:
                        flag = False
                        break
                if not flag:
                    continue
                assert(np.sum(tile_type_keep_nums_temp) == 0)
                agaris.append((heads, ankous, minkous, schunzs))
                if not multiple_return:
                    return agaris

            tile_type_keep_nums[head[0]] += 2

        return agaris


    '''
    def acquire_waiting_tile_types_from_keep_nums_old(self, tile_type_keep_nums, multiple_return=True):
        tiles = tile_type_keep_nums
        kokushi_list = [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33]
        kokushi_num = (tiles[0] + tiles[8] + tiles[9] + tiles[17] + tiles[18] + tiles[26] +
            tiles[27] + tiles[28] + tiles[29] + tiles[30] + tiles[31] + tiles[32] + tiles[33])
        if kokushi_num == 13:
            kokushi_kind = np.sum(tiles[kokushi_list] > 0)
            if kokushi_kind == 13:
                return kokushi_list
            elif kokushi_kind == 12:
                kokushi_tile_type = kokushi_list[np.where(tiles[kokushi_list] == 0)[0][0]]
                return [kokushi_tile_type]
        # Now no kokushi
        waiting_type_candidates = []
        for it in range(3):
            tile_walls = np.zeros((13,), dtype = 'bool')
            tile_walls[2:11] = (tiles[9*it:9*(it+1)] > 0)
            if np.sum(tile_walls) == 0:
                continue
            for tile_type in range(9*it, 9*(it+1)):
                if tiles[tile_type] == 4:
                    continue
                elif tiles[tile_type] > 0:
                    waiting_type_candidates.append(tile_type)
                else: # tiles[tile_type] == 0
                    i = tile_type - 9 * it + 2
                    if ((tile_walls[i-2] and tile_walls[i-1]) or (tile_walls[i-1] and tile_walls[i+1]) or
                        (tile_walls[i+1] and tile_walls[i+2])):
                        waiting_type_candidates.append(tile_type)
        for tile_type in range(27, 34):
            if 0 < tiles[tile_type] < 4:
                waiting_type_candidates.append(tile_type)
        waiting_type = []
        for tile_type in waiting_type_candidates:
            tiles[tile_type] += 1
            if self.check_agari_from_keep_nums_old(tiles):
                waiting_type.append(tile_type)
                if not multiple_return:
                    return waiting_type
            tiles[tile_type] -= 1
        return waiting_type

    def check_agari_from_keep_nums_old(self, tile_type_keep_nums):
        def _is_mentsu(m):
            a = m & 7
            b = 0
            c = 0
            if a == 1 or a == 4:
                b = c = 1
            elif a == 2:
                b = c = 2
            m >>= 3
            a = (m & 7) - b
            if a < 0:
                return False
            is_not_mentsu = False
            for _ in range(0, 6):
                b = c
                c = 0
                if a == 1 or a == 4:
                    b += 1
                    c += 1
                elif a == 2:
                    b += 2
                    c += 2
                m >>= 3
                a = (m & 7) - b
                if a < 0:
                    is_not_mentsu = True
                    break
            if is_not_mentsu:
                return False
            m >>= 3
            a = (m & 7) - c
            return a == 0 or a == 3
        def _is_atama_mentsu(nn, m):
            if nn == 0:
                if (m & (7 << 6)) >= (2 << 6) and _is_mentsu(m - (2 << 6)):
                    return True
                if (m & (7 << 15)) >= (2 << 15) and _is_mentsu(m - (2 << 15)):
                    return True
                if (m & (7 << 24)) >= (2 << 24) and _is_mentsu(m - (2 << 24)):
                    return True
            elif nn == 1:
                if (m & (7 << 3)) >= (2 << 3) and _is_mentsu(m - (2 << 3)):
                    return True
                if (m & (7 << 12)) >= (2 << 12) and _is_mentsu(m - (2 << 12)):
                    return True
                if (m & (7 << 21)) >= (2 << 21) and _is_mentsu(m - (2 << 21)):
                    return True
            elif nn == 2:
                if (m & (7 << 0)) >= (2 << 0) and _is_mentsu(m - (2 << 0)):
                    return True
                if (m & (7 << 9)) >= (2 << 9) and _is_mentsu(m - (2 << 9)):
                    return True
                if (m & (7 << 18)) >= (2 << 18) and _is_mentsu(m - (2 << 18)):
                    return True
            return False
        def _to_meld(tiles, d):
            result = 0
            for i in range(0, 9):
                result |= (tiles[d + i] << i * 3)
            return result

        tiles = tile_type_keep_nums
        j = (1 << tiles[27]) | (1 << tiles[28]) | (1 << tiles[29]) | (1 << tiles[30]) | \
            (1 << tiles[31]) | (1 << tiles[32]) | (1 << tiles[33])
        if j >= 0x10:
            return False
        # 13 orphans
        if ((j & 3) == 2) and (tiles[0] * tiles[8] * tiles[9] * tiles[17] * tiles[18] *
            tiles[26] * tiles[27] * tiles[28] * tiles[29] * tiles[30] *
            tiles[31] * tiles[32] * tiles[33] == 2):
            return True
        # seven pairs
        if not (j & 10) and np.sum(tiles == 2) == 7:
            return True
        if j & 2:
            return False
        n00 = tiles[0] + tiles[3] + tiles[6]
        n01 = tiles[1] + tiles[4] + tiles[7]
        n02 = tiles[2] + tiles[5] + tiles[8]
        n10 = tiles[9] + tiles[12] + tiles[15]
        n11 = tiles[10] + tiles[13] + tiles[16]
        n12 = tiles[11] + tiles[14] + tiles[17]
        n20 = tiles[18] + tiles[21] + tiles[24]
        n21 = tiles[19] + tiles[22] + tiles[25]
        n22 = tiles[20] + tiles[23] + tiles[26]
        n0 = (n00 + n01 + n02) % 3
        if n0 == 1:
            return False
        n1 = (n10 + n11 + n12) % 3
        if n1 == 1:
            return False
        n2 = (n20 + n21 + n22) % 3
        if n2 == 1:
            return False
        if (int(n0 == 2) + int(n1 == 2) + int(n2 == 2) + int(tiles[27] == 2) + int(tiles[28] == 2) +
            int(tiles[29] == 2) + int(tiles[30] == 2) + int(tiles[31] == 2) + int(tiles[32] == 2) +
            int(tiles[33] == 2) != 1):
            return False
        nn0 = (n00 * 1 + n01 * 2) % 3
        m0 = _to_meld(tiles, 0)
        nn1 = (n10 * 1 + n11 * 2) % 3
        m1 = _to_meld(tiles, 9)
        nn2 = (n20 * 1 + n21 * 2) % 3
        m2 = _to_meld(tiles, 18)
        if j & 4:
            return not (n0 | nn0 | n1 | nn1 | n2 | nn2) and _is_mentsu(m0) \
                and _is_mentsu(m1) and _is_mentsu(m2)
        if n0 == 2:
            return not (n1 | nn1 | n2 | nn2) and _is_mentsu(m1) and _is_mentsu(m2) \
                and _is_atama_mentsu(nn0, m0)
        if n1 == 2:
            return not (n2 | nn2 | n0 | nn0) and _is_mentsu(m2) and _is_mentsu(m0) \
                and _is_atama_mentsu(nn1, m1)
        if n2 == 2:
            return not (n0 | nn0 | n1 | nn1) and _is_mentsu(m0) and _is_mentsu(m1) \
                and _is_atama_mentsu(nn2, m2)
        return False
    '''

