''' Function to build tenhou log data to tfrecord format.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import numpy as np

import time

import tensorflow as tf

import copy

import xml.etree.ElementTree as etree

from .event_decoder import EventDecoder
from .mahjong_log_pb2 import MahjongLog, MahjongEvent
from .mahjong_log_io import MahjongLogIO

def decode_log_text(log_text):
    mahjong_log = MahjongLog()
    events = etree.fromstring(log_text)
    go_flag = False
    for event in events:
        if event.tag == 'go':
            assert(not go_flag)
            mahjong_log.game_type = int(event.attrib['type'])
            go_flag = True
        elif event.tag == 'taikyoku':
            assert(int(event.attrib['oya']) == 0)
        elif event.tag == 'shuffle':
            pass
        else:
            event_tf = mahjong_log.events.add()
            EventDecoder.decode(event, event_tf, mode='log', my_player=0)
    return mahjong_log

def build_logs(link_path, log_path, build_path):
    start_time = time.time()
    io_log = MahjongLogIO(build_path)
    f = open(link_path, 'r')
    link_lines = f.readlines()
    f.close()
    game_times = [int(link_line.split('"')[1].split('=')[1][:10]) for link_line in link_lines]
    f = open(log_path, 'r')
    for i, line_log in enumerate(f):
        game_time = game_times[i]
        log_text = line_log.strip()
        mahjong_log = decode_log_text(log_text)
        mahjong_log.game_time = game_time
        io_log.write(mahjong_log)
        if (i+1) % 1000 == 0:
            end_time = time.time()
            elapsed_time = end_time - start_time
            print("%d'th iteration. Time elapsed : %d seconds."%(i+1, elapsed_time))
    f.close()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Total %d data. Time elapsed : %d seconds."%(i+1, elapsed_time))

if __name__ == "__main__":
    if len(sys.argv) == 1:
        link_path = '../data/log_links.txt'
        log_path = '../data/logs.txt'
        build_path = '../data/logs.tfrecord'
    else:
        link_path = sys.argv[1]
        log_path = sys.argv[2]
        build_path = sys.argv[3]
    print('Build logs.')
    build_logs(link_path, log_path, build_path)

