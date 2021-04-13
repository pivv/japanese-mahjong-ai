from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time

import numpy as np
from copy import copy, deepcopy

from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from .mahjong_constant import *

def acquire_images_from_tile_types(tile_types, is_aka_infos=None):
    if is_aka_infos is None:
        is_aka_infos = [False] * len(tile_types)
    images = []
    for tile_type, is_aka in zip(tile_types, is_aka_infos):
        images.append(np.copy(MAHJONG_TILE_IMAGES[tile_type][int(is_aka)]))
    return images

'''
def acquire_images_from_tiles(tiles):
    akadoras = AKADORAS[:]
    tile_types = [tile//4 for tile in tiles]
    is_aka_infos = [tile in akadoras for tile in tiles]
    return acquire_images_from_tile_types(tile_types, is_aka_infos)
'''

def rotate_image(image, rotate_type=1): # 0 ~ 3
    assert(rotate_type in [0, 1, 2, 3])
    if rotate_type == 0:
        return image
    elif rotate_type == 1:
        return np.transpose(image, (1,0,2))[::-1]
    elif rotate_type == 2:
        return image[::-1, ::-1]
    else:
        return np.transpose(image, (1,0,2))[:, ::-1]

def pad_image(image, pad_shape, row_mode=0, col_mode=0): # 0 means top / left
    row, col, _ = image.shape
    row_pad, col_pad = pad_shape
    assert(row <= row_pad and col <= col_pad)
    image_pad = np.zeros((row_pad, col_pad, image.shape[2]), dtype='float32')
    assert(row_mode == 0 or row_mode == 1)
    assert(col_mode == 0 or col_mode == 1)
    if row_mode == 0 and col_mode == 0:
        image_pad[:row, :col, :] = image
    elif row_mode == 0 and col_mode == 1:
        image_pad[:row, -col:, :] = image
    elif row_mode == 1 and col_mode == 0:
        image_pad[-row:, :col, :] = image
    elif row_mode == 1 and col_mode == 1:
        image_pad[-row:, -col:, :] = image
    return image_pad

def mask_image(image, color='r', alpha=0.15):
    mask = np.zeros_like(image)
    mask[:,:,3] = image[:,:,3]
    if color == 'r':
        mask[:,:,0] = 1.
    elif color == 'g':
        mask[:,:,1] = 1.
    elif color == 'b':
        mask[:,:,2] = 1.
    elif color == 'k':
        pass
    elif color == 'back':
        mask[:,:,1] = 0.3373
        mask[:,:,2] = 0.4
    else:
        raise ValueError('Not implemented color.')
    return (1.-alpha)*image + alpha*mask

def combine_images(images, axis=1, border=5): # axis: combine axis (0 or 1)
    assert(axis == 0 or axis == 1)
    image_combined = np.copy(images[0])
    for image in images[1:]:
        if image_combined.shape[1-axis] > image.shape[1-axis]:
            image_old = image
            if axis == 0:
                image = np.zeros((image.shape[0], image_combined.shape[1], image.shape[2]), dtype='float32')
                image[:, :image_old.shape[1], :] = image_old
            else:
                image = np.zeros((image_combined.shape[0], image.shape[1], image.shape[2]), dtype='float32')
                image[-image_old.shape[0]:, :, :] = image_old
        elif image_combined.shape[1-axis] < image.shape[1-axis]:
            image_combined_old = image_combined
            if axis == 0:
                image_combined = np.zeros((image_combined.shape[0], image.shape[1], image_combined.shape[2]), dtype='float32')
                image_combined[:, :image_combined_old.shape[1], :] = image_combined_old
            else:
                image_combined = np.zeros((image.shape[0], image_combined.shape[1], image_combined.shape[2]), dtype='float32')
                image_combined[-image_combined_old.shape[0]:, :, :] = image_combined_old
        if axis == 0:
            image_border = np.zeros((border, image.shape[1], image.shape[2]), dtype='float32')
        else:
            image_border = np.zeros((image.shape[0], border, image.shape[2]), dtype='float32')
        image_combined = np.concatenate([image_combined, image_border, image], axis=axis)
    return image_combined

def mask_images(images, mask):
    assert(len(images) == len(mask))
    max_mask = [np.max(v) for v in mask]
    for index, v in enumerate(max_mask):
        if v == 0:
            images[index] = mask_image(images[index], color='r')
        elif v == 1:
            images[index] = mask_image(images[index], color='g')
        elif v == 2:
            images[index] = mask_image(images[index], color='b')
        elif v == 3:
            images[index] = mask_image(images[index], color='k')
    return images

def acquire_hand_image_from_tile_types(tile_types, is_aka_infos=None, mask=None, border=5):
    images = acquire_images_from_tile_types(tile_types, is_aka_infos)
    if mask is not None:
        images = mask_images(images, mask)
    return combine_images(images, 1, border)

def acquire_hand_image_from_tiles(tiles, mask=None, border=5):
    akadoras = AKADORAS[:]
    tile_types = [tile//4 for tile in tiles]
    is_aka_infos = [tile in akadoras for tile in tiles]
    return acquire_hand_image_from_tile_types(tile_types, is_aka_infos, mask, border)

def acquire_thrown_image_from_tile_types(tile_types, is_aka_infos=None,
    is_called_infos=None, is_reached_infos=None, border=5):
    images = acquire_images_from_tile_types(tile_types, is_aka_infos)
    if is_reached_infos is not None:
        assert(len(images) == len(is_reached_infos))
        if np.sum(is_reached_infos) > 0:
            index_to_rotate = np.where(is_reached_infos)[0][0]
            if is_called_infos is not None:
                while index_to_rotate < len(images) and is_called_infos[index_to_rotate]:
                    index_to_rotate += 1
            if index_to_rotate < len(images):
                images[index_to_rotate] = rotate_image(images[index_to_rotate], 1)
    if is_called_infos is not None:
        assert(len(images) == len(is_called_infos))
        images = [images[index] for index in range(len(images)) if not is_called_infos[index]]
    if len(images) == 0: # no thrown
        return np.zeros((0, 0, 4), dtype='float32')

    line_images = []
    line_num = 6
    for iline in range(3):
        start_index = iline * line_num
        end_index = min((iline+1) * line_num, len(images)) if iline < 2 else len(images)
        line_images.append(combine_images(images[start_index:end_index], 1, border))
        if end_index == len(images):
            break
    return combine_images(line_images, 0, border)

def acquire_thrown_image_from_tiles(tiles, is_called_infos=None, is_reached_infos=None, border=5):
    akadoras = AKADORAS[:]
    tile_types = [tile//4 for tile in tiles]
    is_aka_infos = [tile in akadoras for tile in tiles]
    return acquire_thrown_image_from_tile_types(tile_types, is_aka_infos, is_called_infos, is_reached_infos, border)

def acquire_meld_image_from_tile_types(meld_type, meld_called, tile_types, is_aka_infos=None, border=5):
    if meld_type in ['chi', 'pon', 'daiminkan']:
        images = acquire_images_from_tile_types(tile_types, is_aka_infos)
        images[meld_called] = rotate_image(images[meld_called], 1)
    elif meld_type == 'chakan':
        images = acquire_images_from_tile_types(tile_types, is_aka_infos)
        center_image = rotate_image(combine_images([images[meld_called], images[-1]], 1, border), 1)
        images[meld_called] = center_image
        images = images[:-1]
    else:
        assert(meld_type == 'ankan')
        assert(not is_aka_infos[1] and not is_aka_infos[2])
        images = acquire_images_from_tile_types(
            [tile_types[0], NUM_TILE_TYPE, NUM_TILE_TYPE, tile_types[-1]], is_aka_infos)
    return combine_images(images, 1, border)

def acquire_meld_image_from_tiles(meld_type, meld_called, meld_tiles, border=5):
    akadoras = AKADORAS[:]
    tile_types = [tile//4 for tile in meld_tiles]
    is_aka_infos = [tile in akadoras for tile in meld_tiles]
    return acquire_meld_image_from_tile_types(meld_type, meld_called, tile_types, is_aka_infos, border)

def acquire_meld_image_from_index(meld_index, num_aka=0, border=5):
    if meld_index < 21: # schunz
        meld_type = 'chi'
        base = (meld_index//7) * 9 + (meld_index%7)
        tile_types = [base, base+1, base+2]
        meld_called = int(np.random.random_sample() * 3)
    elif meld_index < 55: # minkou
        meld_type = 'pon'
        base = meld_index - 21
        tile_types = [base, base, base]
        meld_called = int(np.random.random_sample() * 3)
    elif meld_index < 89: # minkan
        meld_type = 'daiminkan'
        base = meld_index - 55
        tile_types = [base, base, base, base]
        meld_called = int(np.random.random_sample() * 4)
    elif meld_index < 123: # ankan
        meld_type = 'ankan'
        base = meld_index - 89
        tile_types = [base, 34, 34, base]
        meld_called = -1
    else: # chakan
        meld_type = 'chakan'
        base = meld_index - 123
        tile_types = [base, base, base, base]
        meld_called = int(np.random.random_sample() * 3)
    is_aka_infos = [False] * len(tile_types)
    if num_aka > 0:
        akadoras = AKADORAS[:]
        assert(num_aka == 1)
        assert(np.sum([(tile_type*4) in akadoras for tile_type in tile_types]) >= 1)
        aka_index = np.where([(tile_type*4) in akadoras for tile_type in tile_types])[0][0]
        is_aka_infos[aka_index] = True
    return acquire_meld_image_from_tile_types(meld_type, meld_called, tile_types, is_aka_infos, border)

def acquire_player_image(player, mask=None, blind=False, border=5, large_border=100):
    if blind:
        if player.hand is None:
            hand = [NUM_TILE_TYPE*4] * (13 - 3 * (
                len(player.chis)+len(player.pons)+len(player.daiminkans)+len(player.chakans)+len(player.ankans)))
        else:
            hand = [NUM_TILE_TYPE*4] * len(player.hand)
    else:
        hand = player.hand
    hand_image = acquire_hand_image_from_tiles(hand, mask, border)
    hand_meld_images = [hand_image]
    for meld_called, meld_tiles in zip(player.chis_called, player.chis):
        meld_image = acquire_meld_image_from_tiles('chi', meld_called, meld_tiles, border)
        hand_meld_images.append(meld_image)
    for meld_called, meld_tiles in zip(player.pons_called, player.pons):
        meld_image = acquire_meld_image_from_tiles('pon', meld_called, meld_tiles, border)
        hand_meld_images.append(meld_image)
    for meld_called, meld_tiles in zip(player.daiminkans_called, player.daiminkans):
        meld_image = acquire_meld_image_from_tiles('daiminkan', meld_called, meld_tiles, border)
        hand_meld_images.append(meld_image)
    for meld_called, meld_tiles in zip(player.chakans_called, player.chakans):
        meld_image = acquire_meld_image_from_tiles('chakan', meld_called, meld_tiles, border)
        hand_meld_images.append(meld_image)
    for meld_tiles in player.ankans:
        meld_image = acquire_meld_image_from_tiles('ankan', -1, meld_tiles, border)
        hand_meld_images.append(meld_image)
    hand_meld_image = combine_images(hand_meld_images, 1, large_border)
    thrown_image = acquire_thrown_image_from_tiles(player.thrown,
        player.thrown_is_called, player.thrown_is_reached, border)
    return hand_meld_image, thrown_image

def acquire_board_image(iplayer, mahjong_board, mask_players=None, blind_players=None):
    border = 5
    large_border = 100

    board_size = 3600
    board_image_shape = (board_size, board_size)
    board_image = np.zeros(board_image_shape + (4,), dtype='float32')

    dora_indicator_image_shape = (MAHJONG_TILE_SHAPE[0], MAHJONG_TILE_SHAPE[1]*5 + border*4)
    uradora_indicator_image_shape = (MAHJONG_TILE_SHAPE[0], MAHJONG_TILE_SHAPE[1]*5 + border*4)
    hand_meld_image_shape = (MAHJONG_TILE_SHAPE[1]*2 + border, 3365)
    thrown_image_shape = (MAHJONG_TILE_SHAPE[0]*3 + border*2, 2000)

    dora_indicator_image = acquire_hand_image_from_tiles(mahjong_board.dora_indicators, None, border)
    for ishape in range(2):
        assert(dora_indicator_image.shape[ishape] <= dora_indicator_image_shape[ishape])
    if len(mahjong_board.uradora_indicators) > 0:
        uradora_indicator_image = acquire_hand_image_from_tiles(mahjong_board.uradora_indicators, None, border)
        for ishape in range(2):
            assert(uradora_indicator_image.shape[ishape] <= uradora_indicator_image_shape[ishape])
    else:
        uradora_indicator_image = np.zeros((0, 0, 4), dtype='float32')

    player_images = []
    for j in range(4):
        if iplayer == -1:
            jplayer = j
            blind = False
        else:
            jplayer = (iplayer+j)%NUM_PLAYER
            blind = (iplayer != jplayer)
        mask = None
        if mask_players is not None:
            mask = mask_players[jplayer]
        if blind_players is not None:
            blind = blind and blind_players[jplayer]
        hand_meld_image, thrown_image = acquire_player_image(
            mahjong_board.players[jplayer], mask, blind, border, large_border)
        for ishape in range(2):
            assert(hand_meld_image.shape[ishape] <= hand_meld_image_shape[ishape])
            assert(thrown_image.shape[ishape] <= thrown_image_shape[ishape])
        player_images.append((hand_meld_image, thrown_image))

    # Now ready to draw
    for j in range(3, -1, -1):
        hand_meld_image, thrown_image = player_images[j]
        hand_meld_pos_0 = board_size//2+1300
        hand_meld_pos_1 = 200 if hand_meld_image.shape[1] < 3000 else 0
        thrown_pos_0 = board_size//2+500
        thrown_pos_1 = board_size//2-500
        if j == 0:
            hand_meld_position = (hand_meld_pos_0, hand_meld_pos_1)
            thrown_position = (thrown_pos_0, thrown_pos_1)
        elif j == 1:
            hand_meld_position = (-hand_meld_pos_1-hand_meld_image.shape[1], hand_meld_pos_0)
            thrown_position = (-thrown_pos_1-thrown_image.shape[1], thrown_pos_0)
        elif j == 2:
            hand_meld_position = (-hand_meld_pos_0-hand_meld_image.shape[0], -hand_meld_pos_1-hand_meld_image.shape[1])
            thrown_position = (-thrown_pos_0-thrown_image.shape[0], -thrown_pos_1-thrown_image.shape[1])
        else:
            hand_meld_position = (hand_meld_pos_1, -hand_meld_pos_0-hand_meld_image.shape[0])
            thrown_position = (thrown_pos_1, -thrown_pos_0-thrown_image.shape[0])
        rotated_hand_meld_image = rotate_image(hand_meld_image, j)
        rotated_thrown_image = rotate_image(thrown_image, j)
        board_image[hand_meld_position[0]:hand_meld_position[0]+rotated_hand_meld_image.shape[0],
            hand_meld_position[1]:hand_meld_position[1]+rotated_hand_meld_image.shape[1]] = rotated_hand_meld_image
        board_image[thrown_position[0]:thrown_position[0]+rotated_thrown_image.shape[0],
            thrown_position[1]:thrown_position[1]+rotated_thrown_image.shape[1]] = rotated_thrown_image

    dora_indicator_position = (board_size//2-MAHJONG_TILE_SHAPE[0], board_size//2-500+MAHJONG_TILE_SHAPE[1]+border)
    uradora_indicator_position = (board_size//2+50, board_size//2-500+MAHJONG_TILE_SHAPE[1]+border)
    board_image[dora_indicator_position[0]:dora_indicator_position[0]+dora_indicator_image.shape[0],
        dora_indicator_position[1]:dora_indicator_position[1]+dora_indicator_image.shape[1]] = dora_indicator_image
    board_image[uradora_indicator_position[0]:uradora_indicator_position[0]+uradora_indicator_image.shape[0],
        uradora_indicator_position[1]:uradora_indicator_position[1]+uradora_indicator_image.shape[1]] = uradora_indicator_image

    return board_image

def acquire_hand_meld_dora_image(tile_types, melds=None, doras=None, tile_aka_infos=None, meld_aka_infos=None, mask=None):
    hand_image = acquire_hand_image_from_tile_types(tile_types, tile_aka_infos, mask)
    hand_meld_dora_image = hand_image
    if melds:
        for meld_index in melds:
            meld_image = acquire_meld_image_from_index(meld_index, meld_aka_infos)
            hand_meld_dora_image = combine_images([hand_meld_dora_image, meld_image], border=100)
    if doras:
        dora_image = acquire_hand_image_from_tile_types(doras)
        hand_meld_dora_image = combine_images([hand_meld_dora_image, dora_image], border=200)
    return hand_meld_dora_image

def sort_hand_images(hand_images, tile_types):
    tile_images = hand_images[1]
    tile_images = [v[1] for v in sorted(enumerate(tile_images), key=lambda v:tile_types[v[0]])]
    hand_images[1] = tile_images
    return hand_images

def combine_hand_images(hand_images):
    dora_images = hand_images[0]
    tile_images = hand_images[1]
    image_combined = combine_images(tile_images)
    for meld_images in hand_images[2:]:
        meld_image = combine_images(meld_images)
        image_combined = combine_images([image_combined, meld_image], border=100)
    if len(dora_images) > 0:
        dora_image = combine_images(dora_images)
        image_combined = combine_images([image_combined, dora_image], border=200)
    return image_combined

def plot_image(image, figsize=(15,15), scale=None):
    row, col, _ = image.shape
    if scale is None:
        scale = max(col / float(figsize[0]), row / float(figsize[1]))
    plt.figure()
    plt.gca().clear()
    plt.imshow(image)
    plt.axis('off')
    plt.gcf().set_size_inches(float(col) / scale, float(row) / scale)
    plt.show()


MAHJONG_TILE_DIR = '../image/tiles/'
MAHJONG_TILE_SHAPE = (200, 150)
MAHJONG_TILE_NAMES = ([['Man_%d'%(i+1)] for i in range(9)] +
    [['Pin_%d'%(i+1)] for i in range(9)] +
    [['Sou_%d'%(i+1)] for i in range(9)] +
    [['Jihai_' + s] for s in ['ton', 'nan', 'shaa', 'pei', 'haku', 'hatsu', 'chun']])
MAHJONG_TILE_NAMES[4].append('Man_5_aka')
MAHJONG_TILE_NAMES[9+4].append('Pin_5_aka')
MAHJONG_TILE_NAMES[9+9+4].append('Sou_5_aka')
MAHJONG_TILE_IMAGES = [[mpimg.imread(os.path.join(MAHJONG_TILE_DIR, name+'.png')).astype('float32')
    for name in names] for names in MAHJONG_TILE_NAMES]
MAHJONG_TILE_IMAGES += [[mask_image(MAHJONG_TILE_IMAGES[-3][0], color='back', alpha=0.5)]]
assert(len(MAHJONG_TILE_IMAGES) == 35)