''' Tenhou client
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time

import numpy as np
from copy import copy, deepcopy

from collections import defaultdict

import re
import datetime

import socket
from threading import Thread
import six.moves.urllib as urllib

import xml.etree.ElementTree as etree

from .socket_constant import *
from ..event.event_decoder import event_to_process, EventDecoder

class TenhouClient(object):
    def __init__(self, logger=None):
        self.socket = None
        self.logger = logger
        self.id = 'NoName'
        self.lobby = 0
        self.is_tournament = False
        self.game_is_continue = True
        self.keep_alive_thread = None
        self.reconnected_messages = None
        self.rating_string = None

    def send_message(self, message):
        if self.logger is not None:
            self.logger.debug('Send: {}'.format(message))
        message += '\0'
        self.socket.sendall(message.encode())

    def read_messages(self):
        message = self.socket.recv(2048).decode('utf-8')
        # to use etree & should be changed to &amp.
        messages = re.sub(r'&', '&amp;', message)
        messages = messages.split('\x00')
        # last message is always empty after split.
        messages = messages[0:-1]
        if len(messages) > 0:
            if self.logger is not None:
                self.logger.debug('Get: {}'.format(message.decode('utf-8').replace('\x00', ' ')))
        return [etree.fromstring(message) for message in messages]

    def read_messages_loop(self, max_count_of_empty_messages=120):
        count_of_empty_messages = 0
        messages = []
        while len(messages) == 0:
            time.sleep(SLEEP_BETWEEN_ACTIONS)
            messages = self.read_messages()
            if len(messages) == 0:
                count_of_empty_messages += 1
            # socket was closed by tenhou
            if count_of_empty_messages >= max_count_of_empty_messages:
                if self.logger is not None:
                    self.logger.error('We are getting empty messages from socket. Probably socket connection was closed')
                return []
        return messages

    def connect(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((TENHOU_HOST, TENHOU_PORT))

    def authenticate(self, id='NoName', lobby=0, is_tournament=False):
        self.id = id
        self.lobby = lobby
        self.is_tournament = is_tournament
        self.send_message('<HELO name="{}" tid="f0" sx="M" />'.format(urllib.parse.quote(self.id)))
        messages = self.read_messages()
        if len(messages) == 0: # Auth message is not received.
            if self.logger is not None:
                self.logger.info("Auth message wasn't received.")
            return False
        auth_message = messages[0]
        # reconnection
        if auth_message.tag.lower() == 'go': # No auth message, but go message.
            if self.logger is not None:
                self.logger.info("Successfully reconnected.")
            self.reconnected_messages = messages[1:]
            #selected_game_type = int(auth_message.attrib['type'])
            authenticated = True
        else:
            assert(auth_message.tag.lower() == 'helo')
            if 'auth' not in auth_message.attrib: # Didn't obtain auth string.
                if self.logger is not None:
                    self.logger.info("We didn't obtain auth string.")
                return False
            auth_string = auth_message.attrib['auth']
            if 'PF4' in auth_message.attrib:
                self.rating_string = auth_message.attrib['PF4']
            if 'nintei' in auth_message.attrib:
                new_rank_string = auth_message.attrib['nintei']
                if self.logger is not None:
                    self.logger.info("Achieved a new rank! \n {}.".format(new_rank_string))
            auth_token = self._generate_auth_token(auth_string)
            self.send_message('<AUTH val="{}"/>'.format(auth_token))
            self.send_message(self._pxr_tag())
            # sometimes tenhou send an empty tag after authentication (in tournament mode)
            # and bot thinks that he was not auth
            # to prevent it lets wait a little bit
            # and lets read a group of tags
            continue_reading = True
            counter = 0
            authenticated = False
            while continue_reading:
                messages = self.read_messages()
                for message in messages:
                    if message.tag.lower() == 'ln':
                        authenticated = True
                        continue_reading = False
                counter += 1
                # to avoid infinity loop
                if counter > 10:
                    continue_reading = False

        if authenticated:
            self._send_keep_alive_ping()
            if self.logger is not None:
                self.logger.info("Successfully authenticated.")
            return True
        else:
            if self.logger is not None:
                self.logger.info("Failed to authenticate.")
            return False

    def acquire_rating(self):
        if not self.rating_string:
            if self.logger is not None:
                self.logger.error("For NoName no rating exists")
            return None

        temp = self.rating_string.split(',')
        dan = int(temp[0])
        rate = float(temp[2])
        if self.logger is not None:
            self.logger.info('Player has {} rank and {} rate'.format(RANKS[dan].encode('utf-8'), rate))

        return dan, rate

    def build_game_type(self, game_type=None):
        if game_type is not None:
            return game_type

        # kyu lobby, hanchan ari-ari
        default_game_type = 9

        if self.lobby != 0:
            if self.logger is not None:
                self.logger.error("We can't use dynamic game type and custom lobby. Default game type was set")
            return default_game_type

        if not self.rating_string:
            if self.logger is not None:
                self.logger.error("For NoName dynamic game type is not available. Default game type was set")
            return default_game_type

        dan, rate = self.acquire_rating()

        game_type = default_game_type
        # dan lobby, we can play here from 1 kyu
        if dan >= 9:
            game_type = 137

        # upperdan lobby, we can play here from 4 dan and with 1800+ rate
        if dan >= 13 and rate >= 1800:
            game_type = 41

        # phoenix lobby, we can play here from 7 dan and with 2000+ rate
        if dan >= 16 and rate >= 2000:
            game_type = 169

        return game_type

    def start_game(self, mahjong_board):
        log_link = ''
        looking_for_game = True

        # play in private or tournament lobby
        if self.lobby != 0:
            if self.is_tournament:
                if self.logger is not None:
                    self.logger.info('Go to the tournament lobby: {}'.format(self.lobby))
                self.send_message('<CS lobby="{}" />'.format(self.lobby))
                time.sleep(SLEEP_BETWEEN_ACTIONS * 2)
                self.send_message('<DATE />')
            else:
                if self.logger is not None:
                    self.logger.info('Go to the lobby: {}'.format(self.lobby))
                self.send_message('<CHAT text="{}" />'.format(urllib.parse.quote('/lobby {}'.format(self.lobby))))
                time.sleep(SLEEP_BETWEEN_ACTIONS * 2)

        if self.reconnected_messages is not None:
            # we already in the game
            looking_for_game = False
            self.send_message('<GOK />')
            time.sleep(SLEEP_BETWEEN_ACTIONS)
        else:
            selected_game_type = mahjong_board.game_type
            lobby_and_game_type = '{},{}'.format(self.lobby, selected_game_type)

            if not self.is_tournament:
                self.send_message('<JOIN t="{}" />'.format(lobby_and_game_type))
                if self.logger is not None:
                    self.logger.info('Looking for the game...')

            start_time = datetime.datetime.now()

            while looking_for_game:
                time.sleep(SLEEP_BETWEEN_ACTIONS)

                messages = self.read_messages()
                for message in messages:
                    if message.tag.lower() == 'rejoin':
                        # game wasn't found, continue to wait
                        self.send_message('<JOIN t="{},r" />'.format(lobby_and_game_type))

                    if message.tag.lower() == 'go':
                        self.send_message('<GOK />')
                        self.send_message('<NEXTREADY />')

                        # we had to have it there
                        # because for tournaments we don't know
                        # what exactly game type was set
                        assert(mahjong_board.game_type == int(message.attrib['type']))

                    if message.tag.lower() == 'taikyoku':
                        looking_for_game = False
                        oya = int(message.attrib['oya'])
                        seat = (4 - oya) % 4
                        game_id = message.attrib['log']
                        log_link = 'http://tenhou.net/0/?log={}&tw={}'.format(game_id, seat)

                    if message.tag.lower() == 'un':
                        event_socket = event_to_process(EventDecoder.decode(message, mode='socket'))
                        mahjong_board.process_un(event_socket)

                    if message.tag.lower() == 'ln':
                        self.send_message(self._pxr_tag())

                current_time = datetime.datetime.now()
                time_difference = current_time - start_time

                if time_difference.seconds > 60 * WAITING_GAME_TIMEOUT_MINUTES:
                    break

        # we wasn't able to find the game in specified time range
        # sometimes it happens and we need to end process
        # and try again later
        if looking_for_game:
            if self.logger is not None:
                self.logger.error("Game is not started. Can't find the game")
            return False

        if self.logger is not None:
            self.logger.info('Game started')
            self.logger.info('Log: {}'.format(log_link))
            self.logger.info('Players: ' + ', '.join('{}'.format(name.encode('utf-8'))
                for name in mahjong_board.player_names))
        return True

    def end_game(self, success=True):
        self.game_is_continue = False
        if success:
            self.send_message('<BYE />')
        if self.logger is not None:
            if success:
                self.logger.info('End of the game')
            else:
                self.logger.error('Game was ended without success')

    def end_client(self):
        if self.keep_alive_thread:
            self.keep_alive_thread.join()
        try:
            self.socket.shutdown(socket.SHUT_RDWR)
            self.socket.close()
        except OSError:
            pass

        if self.logger is not None:
            self.logger.info('End client')

    def send_events(self, events, last_message=None):
        for ievent, event in enumerate(events):
            time.sleep(SLEEP_BETWEEN_ACTIONS)

            if event[0] == 'agari' and event[2][0] == 0: # tsumo
                if last_message is not None:
                    assert(last_message.tag[0].lower() == 't' and
                        't' in last_message.attrib and int(last_message.attrib['t']) in [16, 48])
                self.send_message('<N type="7" />')
            elif event[0] == 'agari' and event[2][0] != 0: # ron
                if last_message is not None:
                    assert(last_message.tag[0].lower() in ['e', 'f', 'g'] and
                        't' in last_message.attrib and int(last_message.attrib['t']) in [8, 9, 10, 11, 12, 13, 15])
                self.send_message('<N type="6" />')
            elif event[0] == 'ryuukyoku' and event[1] == 'yao9':
                if last_message is not None:
                    assert(last_message.tag[0].lower() == 't' and
                        't' in last_message.attrib and int(last_message.attrib['t']) == 64)
                self.send_message('<N type="9" />')
            elif event[0] == 'reach' and event[2] == 1:
                assert(ievent == 0 and len(events) == 2)
                thrown_tile = events[-1][2]
                self.send_message('<REACH hai="{}" />'.format(thrown_tile))
            elif event[0] == 'defg':
                thrown_tile = event[2]
                self.send_message('<D p="{}"/>'.format(thrown_tile))
            elif event[0] == 'n':
                if event[1] == 'chi':
                    meld_from_who, meld_called, meld_tiles = event[3]
                    if last_message is not None:
                        assert(last_message.tag[0].lower() == 'g' and
                            't' in last_message.attrib and int(last_message.attrib['t']) in [4, 5, 7])
                    tiles = meld_tiles[:]
                    del tiles[meld_called]
                    self.send_message('<N type="3" hai0="{}" hai1="{}" />'.format(tiles[0], tiles[1]))
                elif event[1] == 'pon':
                    meld_from_who, meld_called, meld_tiles = event[3]
                    if last_message is not None:
                        assert(last_message.tag[0].lower() in ['e', 'f', 'g'] and
                            't' in last_message.attrib and int(last_message.attrib['t']) in [1, 3, 5, 7])
                    tiles = meld_tiles[:]
                    del tiles[meld_called]
                    self.send_message('<N type="1" hai0="{}" hai1="{}" />'.format(tiles[0], tiles[1]))
                elif event[1] == 'daiminkan':
                    meld_from_who, meld_called, meld_tiles = event[3]
                    if last_message is not None:
                        assert(last_message.tag[0].lower() in ['e', 'f', 'g'] and
                            't' in last_message.attrib and int(last_message.attrib['t']) in [2, 3, 7])
                    self.send_message('<N type="2" />')
                elif event[1] == 'chakan':
                    meld_from_who, meld_called, meld_tiles = event[3]
                    if last_message is not None:
                        assert(last_message.tag[0].lower() == 't')
                    self.send_message('<N type="5" hai="{}" />'.format(meld_tiles[-1]))
                else:
                    assert(event[1] == 'ankan')
                    meld_from_who, meld_called, meld_tiles = event[3]
                    if last_message is not None:
                        assert(last_message.tag[0].lower() == 't')
                    self.send_message('<N type="4" hai="{}" />'.format(meld_tiles[0]))
            elif event[0] == 'no':
                self.send_message('<N />')
            else:
                raise ValueError('Invalid event: %s.'%(event[0]))


    def _pxr_tag(self):
        if self.is_tournament:
            return '<PXR V="-1" />'
        if self.id == 'NoName':
            return '<PXR V="1" />'
        else:
            return '<PXR V="9" />'

    def _send_keep_alive_ping(self):
        def send_request():
            while self.game_is_continue:
                self.send_message('<Z />')
                # we can't use sleep(15), because we want to be able
                # end thread in the middle of running
                seconds_to_sleep = 15
                for x in range(0, seconds_to_sleep * 2):
                    if self.game_is_continue:
                        time.sleep(0.5)
        self.keep_alive_thread = Thread(target=send_request)
        self.keep_alive_thread.start()

    def _generate_auth_token(self, auth_string):
        translation_table = [63006, 9570, 49216, 45888, 9822, 23121, 59830, 51114, 54831, 4189, 580, 5203, 42174, 59972,
            55457, 59009, 59347, 64456, 8673, 52710, 49975, 2006, 62677, 3463, 17754, 5357]
        parts = auth_string.split('-')
        if len(parts) != 2:
            return False
        first_part = parts[0]
        second_part = parts[1]
        if len(first_part) != 8 or len(second_part) != 8:
            return False
        table_index = int('2' + first_part[2:8]) % (12 - int(first_part[7:8])) * 2
        a = translation_table[table_index] ^ int(second_part[0:4], 16)
        b = translation_table[table_index + 1] ^ int(second_part[4:8], 16)
        postfix = format(a, '2x') + format(b, '2x')
        result = first_part + '-' + postfix
        return result


