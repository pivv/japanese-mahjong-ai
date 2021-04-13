''' Function to process tenhou socket data to tfrecord format.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import numpy as np

import time

import tensorflow as tf

import xml.etree.ElementTree as etree

from ..event.event_decoder import EventDecoder
from ..event.mahjong_log_pb2 import MahjongEvent

def decode_one_log_event(event, my_player):
    assert(event.tag != 'go')
    if event.tag == 'taikyoku':
        assert(int(event.attrib['oya']) == 0)
        return None
    elif event.tag == 'shuffle':
        return None
    else:
        event_tf = EventDecoder.decode(event, mode='socket', my_player=my_player)
    return event_tf

