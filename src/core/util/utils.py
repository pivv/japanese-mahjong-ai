''' Some useful utils
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time

import numpy as np
from copy import copy, deepcopy

def lengths_to_pos(lengths, len_max, pad=None):
    assert(len(lengths.shape) == 1)
    if pad is None:
        pad = len_max
    ranges = np.arange(len_max, dtype='uint8')
    ranges_expand = np.repeat(ranges.reshape(1, -1), len(lengths), axis=0)
    lengths_expand = np.repeat(lengths.reshape(-1, 1), len_max, axis=1).astype('uint8')
    lengths_mask = (ranges_expand < lengths_expand)
    ranges_expand[np.logical_not(lengths_mask)] = pad
    return ranges_expand

def lengths_to_mask(lengths_all, len_max): # lengths_all: bs x lq
    assert(len(lengths_all.shape) == 2)
    ranges = np.arange(len_max, dtype='uint8')
    ranges_expand = np.repeat(ranges.reshape(1, -1),
        lengths_all.shape[0]*lengths_all.shape[1], axis=0).reshape(lengths_all.shape+(len_max,))
    lengths_all_expand = np.repeat(lengths_all.reshape(lengths_all.shape+(1,)), len_max, axis=2).astype('uint8')
    lengths_all_mask = (ranges_expand < lengths_all_expand)
    return np.logical_not(lengths_all_mask).astype('uint8')
