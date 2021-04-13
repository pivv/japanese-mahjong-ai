''' Function to decode tenhou event to tfrecord format.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import numpy as np

from urlparse import unquote

from .mahjong_log_pb2 import *

TENHOU_NAMES = ['n0', 'n1', 'n2', 'n3']
TENHOU_HANDS = ['hai0', 'hai1', 'hai2', 'hai3']
TENHOU_THROWNS = ['kawa0', 'kawa1', 'kawa2', 'kawa3']
TENHOU_MELDS = ['m0', 'm1', 'm2', 'm3']

def decode_meld(data):
    relative_from_who = data & 0x3

    def decode_chi():
        assert(relative_from_who == 3)
        meld_type = 'chi'
        t0, t1, t2 = (data >> 3) & 0x3, (data >> 5) & 0x3, (data >> 7) & 0x3
        base_and_called = data >> 10
        meld_called = base_and_called % 3
        base = base_and_called // 3
        base = (base // 7) * 9 + base % 7
        meld_tiles = [t0+4*(base+0), t1+4*(base+1), t2+4*(base+2)]
        return meld_type, relative_from_who, meld_called, meld_tiles
    def decode_pon():
        assert(relative_from_who != 0)
        t4 = (data >> 5) & 0x3
        t0, t1, t2 = ((1,2,3),(0,2,3),(0,1,3),(0,1,2))[t4]
        base_and_called = data >> 9
        meld_called = base_and_called % 3
        base = base_and_called // 3
        if data & 0x8:
            meld_type = 'pon'
            meld_tiles = [t0+4*base, t1+4*base, t2+4*base]
        else:
            meld_type = 'chakan'
            meld_tiles = [t0+4*base, t1+4*base, t2+4*base, t4+4*base]
        return meld_type, relative_from_who, meld_called, meld_tiles
    def decode_nuki():
        assert(relative_from_who == 0)
        meld_called = -1
        meld_type = 'nuki'
        meld_tiles = [data >> 8]
        return meld_type, relative_from_who, meld_called, meld_tiles
    def decode_kan():
        base_and_called = data >> 8
        if relative_from_who != 0:
            meld_type = 'daiminkan'
            meld_called = base_and_called % 4
        else:
            meld_type = 'ankan'
            meld_called = -1
        base = base_and_called // 4
        meld_tiles = [4*base, 1+4*base, 2+4*base, 3+4*base]
        return meld_type, relative_from_who, meld_called, meld_tiles

    if data & 0x4:
        return decode_chi()
    elif data & 0x18:
        return decode_pon()
    elif data & 0x20:
        return decode_nuki()
    else:
        return decode_kan()

def event_to_process(event_tf_base):
    event_type = event_tf_base.type
    if event_type == 'reinit':
        event_type = 'init'
    event_tf = getattr(event_tf_base, 'event_' + event_type)
    if event_type == 'un':
        return (event_type, event_tf.who, list(event_tf.names))
    elif event_type == 'bye':
        return (event_type, event_tf.who)
    elif event_type == 'init':
        hands = [list(hand_tf.values) for hand_tf in event_tf.hands]
        throwns = [list(thrown_tf.values) for thrown_tf in event_tf.throwns]
        melds = [[(meld_one_tf.type, meld_one_tf.from_who, meld_one_tf.called, list(meld_one_tf.tiles)) for
            meld_one_tf in meld_tf.values] for meld_tf in event_tf.melds]
        return (event_type, event_tf.is_reinit, list(event_tf.dora_indicators), hands, throwns, melds,
            event_tf.oya, event_tf.round, event_tf.combo, event_tf.reach_stick, list(event_tf.total_scores))
    elif event_type == 'dora':
        return (event_type, event_tf.dora_indicator)
    elif event_type == 'tuvw':
        return (event_type, event_tf.who, event_tf.tile)
    elif event_type == 'defg':
        return (event_type, event_tf.who, event_tf.tile, event_tf.is_tsumo_giri)
    elif event_type == 'n':
        return (event_type, event_tf.type, event_tf.who,
            (event_tf.from_who, event_tf.called, list(event_tf.tiles)))
    elif event_type == 'reach':
        return (event_type, event_tf.who, event_tf.step)
    elif event_type == 'agari':
        return (event_type, event_tf.who,
            (event_tf.from_who, -1, event_tf.fu,
            str(event_tf.yaku_style), list(event_tf.yaku_types), list(event_tf.yaku_values)),
            list(event_tf.uradora_indicators)) # not use basic_score
    elif event_type == 'ryuukyoku':
        return (event_type, event_tf.type, list(event_tf.round_scores), True)
    else:
        raise ValueError('Invalid event type: %s'%(event_type))

class EventDecoder():
    @staticmethod
    def check_decode_available(event):
        print(event.tag)
        if event.tag.lower() in ['un', 'bye', 'init', 'reinit', 'dora', 'n', 'reach', 'agari', 'ryuukyoku']:
            return True
        elif event.tag.lower() == 'furiten':
            return False
        elif (event.tag[0].lower() in ['d', 'e', 'f', 'g', 't', 'u', 'v', 'w'] and
            (event.tag[1:] == '' or 0<=int(event.tag[1:])<136)):
            return True
        else:
            return False

    @staticmethod
    def decode(event, event_tf_base=None, mode='log', my_player=0):
        assert(mode in ['log', 'socket'])
        if event_tf_base is None:
            event_tf_base = MahjongEvent()
        if event.tag.lower() in ['un', 'bye', 'init', 'reinit', 'dora', 'n', 'reach', 'agari', 'ryuukyoku']:
            event_tf_base.type = event.tag.lower()
            event_type = event_tf_base.type
            if event_type == 'reinit':
                event_type = 'init'
            event_tf = getattr(event_tf_base, 'event_' + event_type)
            decode_fn = getattr(EventDecoder, 'decode_' + event_type)
            decode_fn(event, event_tf, mode, my_player)
        else:
            assert(event.tag[0].lower() in ['d', 'e', 'f', 'g', 't', 'u', 'v', 'w'] and
                (event.tag[1:] == '' or 0<=int(event.tag[1:])<136))
            if event.tag[0].lower() in ['d', 'e', 'f', 'g']:
                event_tf_base.type = 'defg'
                event_tf = event_tf_base.event_defg
                EventDecoder.decode_defg(event, event_tf, mode, my_player)
            else:
                event_tf_base.type = 'tuvw'
                event_tf = event_tf_base.event_tuvw
                EventDecoder.decode_tuvw(event, event_tf, mode, my_player)
        return event_tf_base

    @staticmethod
    def decode_un(event, event_tf, mode='log', my_player=0):
        assert(event.tag.lower() == 'un')
        if 'dan' in event.attrib:
            names_init = [unquote(rawname).strip() for rawname in [event.attrib[name] for name in TENHOU_NAMES]]
            dans_init = [int(dan) for dan in event.attrib['dan'].split(',')]
            rates_init = [float(rate) for rate in event.attrib['rate'].split(',')]
            sexes_init = [str(sex) for sex in event.attrib['sx'].split(',')]
            event_tf.names.extend(names_init[4-my_player:] + names_init[:4-my_player])
            event_tf.dans.extend(dans_init[4-my_player:] + dans_init[:4-my_player])
            event_tf.rates.extend(rates_init[4-my_player:] + rates_init[:4-my_player])
            event_tf.sexes.extend(sexes_init[4-my_player:] + sexes_init[:4-my_player])
            event_tf.who = -1
        else:
            event_tf.who = -1
            for iname, name in enumerate(TENHOU_NAMES):
                if name in event.attrib:
                    event_tf.who = (iname + my_player)%4
                    break
            assert(event_tf.who >= 0)

    @staticmethod
    def decode_bye(event, event_tf, mode='log', my_player=0):
        assert(event.tag.lower() == 'bye')
        event_tf.who = (int(event.attrib['who']) + my_player)%4

    @staticmethod
    def decode_init(event, event_tf, mode='log', my_player=0):
        assert(event.tag.lower() in ['init', 'reinit'])
        event_tf.is_reinit = (event.tag.lower() == 'reinit')
        total_scores_init = [int(s) for s in event.attrib['ten'].split(',')]
        event_tf.total_scores.extend(total_scores_init[4-my_player:] + total_scores_init[:4-my_player])
        seed = [int(s) for s in event.attrib['seed'].split(',')]
        if not event_tf.is_reinit:
            assert(len(seed) == 6)
        event_tf.round = seed[0]
        event_tf.combo = seed[1]
        event_tf.reach_stick = seed[2]
        event_tf.dora_indicators.extend(seed[5:])
        event_tf.oya = (int(event.attrib['oya']) + my_player)%4
        if mode == 'log':
            assert(not event_tf.is_reinit)
            for i, hand in enumerate(TENHOU_HANDS[4-my_player:] + TENHOU_HANDS[:4-my_player]):
                hand_tf = event_tf.hands.add()
                if event.attrib[hand] != '':
                    hand_tf.values.extend([int(s) for s in event.attrib[hand].split(',')])
                else: # 3-man
                    assert(i == 3)
        else:
            assert(mode == 'socket')
            for i in range(4):
                hand_tf = event_tf.hands.add()
                if i == my_player:
                    hand_tf.values.extend([int(s) for s in event.attrib['hai'].split(',')])
            if event_tf.is_reinit:
                for i, thrown in enumerate(TENHOU_THROWNS[4-my_player:] + TENHOU_THROWNS[:4-my_player]):
                    thrown_tf = event_tf.throwns.add()
                    thrown_tf.values.extend([int(s) for s in event.attrib[thrown].split(',')])
                for i, meld in enumerate(TENHOU_MELDS[4-my_player:] + TENHOU_MELDS[:4-my_player]):
                    meld_tf = event_tf.melds.add()
                    if meld in event.attrib:
                        for data in [int(s) for s in event.attrib[meld].split(',')]:
                            meld_type, relative_from_who, meld_called, meld_tiles = decode_meld(data)
                            meld_one_tf = meld_tf.values.add()
                            meld_one_tf.type = meld_type
                            meld_one_tf.from_who = (i + relative_from_who)%4
                            meld_one_tf.called = meld_called
                            meld_one_tf.tiles.extend(meld_tiles)

    @staticmethod
    def decode_dora(event, event_tf, mode='log', my_player=0):
        assert(event.tag.lower() == 'dora')
        event_tf.dora_indicator = int(event.attrib['hai'])

    @staticmethod
    def decode_tuvw(event, event_tf, mode='log', my_player=0):
        assert(event.tag[0].lower() in ['t', 'u', 'v', 'w'] and (event.tag[1:] == '' or 0<=int(event.tag[1:])<136))
        event_tf.who = ('tuvw'.index(event.tag[0].lower()) + my_player)%4
        if event.tag[1:] == '':
            event_tf.tile = -1
        else:
            event_tf.tile = int(event.tag[1:])

    @staticmethod
    def decode_defg(event, event_tf, mode='log', my_player=0):
        assert(event.tag[0].lower() in ['d', 'e', 'f', 'g'] and (event.tag[1:] == '' or 0<=int(event.tag[1:])<136))
        event_tf.who = ('defg'.index(event.tag[0].lower()) + my_player)%4
        if event.tag[1:] == '':
            event_tf.tile = -1
        else:
            event_tf.tile = int(event.tag[1:])
        event_tf.is_tsumo_giri = event.tag[0].islower()

    @staticmethod
    def decode_n(event, event_tf, mode='log', my_player=0):
        assert(event.tag.lower() == 'n')
        event_tf.who = (int(event.attrib['who']) + my_player)%4

        data = int(event.attrib['m'])
        meld_type, relative_from_who, meld_called, meld_tiles = decode_meld(data)

        event_tf.type = meld_type
        event_tf.from_who = (event_tf.who + relative_from_who)%4
        event_tf.called = meld_called
        event_tf.tiles.extend(meld_tiles)

    @staticmethod
    def decode_reach(event, event_tf, mode='log', my_player=0):
        assert(event.tag.lower() == 'reach')
        event_tf.who = (int(event.attrib['who']) + my_player)%4
        event_tf.step = int(event.attrib['step'])

    @staticmethod
    def decode_agari(event, event_tf, mode='log', my_player=0):
        assert(event.tag.lower() == 'agari')
        event_tf.who = (int(event.attrib['who']) + my_player)%4
        if 'fromwho' in event.attrib:
            event_tf.from_who = (int(event.attrib['fromwho']) + my_player)%4
        else:
            event_tf.from_who = (int(event.attrib['fromWho']) + my_player)%4
        if 'paowho' in event.attrib:
            event_tf.pao_who = (int(event.attrib['paowho']) + my_player)%4
        elif 'paoWho' in event.attrib:
            event_tf.pao_who = (int(event.attrib['paoWho']) + my_player)%4
        else:
            event_tf.pao_who = -1
        ten = [int(s) for s in event.attrib['ten'].split(',')]
        assert(len(ten) == 3)
        event_tf.round_score = ten[1]
        event_tf.fu = ten[0]
        event_tf.limit = ten[2]
        if 'dorahai' in event.attrib:
            event_tf.dora_indicators.extend([int(s) for s in event.attrib['dorahai'].split(',')])
        else:
            event_tf.dora_indicators.extend([int(s) for s in event.attrib['doraHai'].split(',')])
        if 'dorahaiura' in event.attrib:
            event_tf.uradora_indicators.extend([int(s) for s in event.attrib['dorahaiura'].split(',')])
        elif 'doraHaiUra' in event.attrib:
            event_tf.uradora_indicators.extend([int(s) for s in event.attrib['doraHaiUra'].split(',')])
        event_tf.hand.values.extend([int(s) for s in event.attrib['hai'].split(',')])
        if 'm' in event.attrib:
            for data in [int(s) for s in event.attrib['m'].split(',')]:
                meld_type, relative_from_who, meld_called, meld_tiles = decode_meld(data)
                meld_one_tf = event_tf.meld.values.add()
                meld_one_tf.type = meld_type
                meld_one_tf.from_who = (event_tf.who + relative_from_who)%4
                meld_one_tf.called = meld_called
                meld_one_tf.tiles.extend(meld_tiles)
        else:
            assert(len(event_tf.hand.values) == 14)
        if 'yaku' in event.attrib:
            event_tf.yaku_style = 'yaku'
            yakus = [int(s) for s in event.attrib['yaku'].split(',')]
            event_tf.yaku_values.extend(yakus[1::2])
            event_tf.yaku_types.extend(yakus[0::2])
        else:
            assert('yakuman' in event.attrib)
            event_tf.yaku_style = 'yakuman'
            yakuman_types = [int(s) for s in event.attrib['yakuman'].split(',')]
            yakuman_values = [13 for _ in yakuman_types]
            event_tf.yaku_values.extend(yakuman_values)
            event_tf.yaku_types.extend(yakuman_types)
        ba = [int(s) for s in event.attrib['ba'].split(',')]
        assert(len(ba) == 2)
        event_tf.combo = ba[0]
        event_tf.reach_stick = ba[1]
        scores = [int(s) for s in event.attrib['sc'].split(',')]
        assert(len(scores) == 8)
        round_scores_init = scores[1::2]
        total_scores_init = scores[0::2]
        event_tf.round_scores.extend(round_scores_init[4-my_player:] + round_scores_init[:4-my_player])
        event_tf.total_scores.extend(total_scores_init[4-my_player:] + total_scores_init[:4-my_player])
        if 'owari' in event.attrib:
            scores = [int(float(s)) for s in event.attrib['owari'].split(',')]
            assert(len(scores) == 8)
            owari_scores_init = scores[0::2]
            owari_points_init = scores[1::2]
            event_tf.owari_scores.extend(owari_scores_init[4-my_player:] + owari_scores_init[:4-my_player])
            event_tf.owari_points.extend(owari_points_init[4-my_player:] + owari_points_init[:4-my_player])

    @staticmethod
    def decode_ryuukyoku(event, event_tf, mode='log', my_player=0):
        assert(event.tag.lower() == 'ryuukyoku')
        if 'type' in event.attrib:
            event_tf.type = str(event.attrib['type'])
            assert(event.attrib['type'] in ['nm', 'reach4', 'yao9', 'kaze4', 'ron3', 'kan4'])
        else:
            event_tf.type = ''
        for i, hand in enumerate(TENHOU_HANDS[4-my_player:] + TENHOU_HANDS[:4-my_player]):
            hand_tf = event_tf.hands.add()
            if hand in event.attrib:
                hand_tf.values.extend([int(s) for s in event.attrib[hand].split(',')])
        ba = [int(s) for s in event.attrib['ba'].split(',')]
        assert(len(ba) == 2)
        event_tf.combo = ba[0]
        event_tf.reach_stick = ba[1]
        scores = [int(s) for s in event.attrib['sc'].split(',')]
        assert(len(scores) == 8)
        round_scores_init = scores[1::2]
        total_scores_init = scores[0::2]
        event_tf.round_scores.extend(round_scores_init[4-my_player:] + round_scores_init[:4-my_player])
        event_tf.total_scores.extend(total_scores_init[4-my_player:] + total_scores_init[:4-my_player])
        if 'owari' in event.attrib:
            scores = [int(float(s)) for s in event.attrib['owari'].split(',')]
            assert(len(scores) == 8)
            owari_scores_init = scores[0::2]
            owari_points_init = scores[1::2]
            event_tf.owari_scores.extend(owari_scores_init[4-my_player:] + owari_scores_init[:4-my_player])
            event_tf.owari_points.extend(owari_points_init[4-my_player:] + owari_points_init[:4-my_player])

