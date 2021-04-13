cdef _is_mentsu(int m):
    cdef int a = m & 7
    cdef int b = 0
    cdef int c = 0
    cdef bint is_not_mentsu = False
    cdef int i
    if a == 1 or a == 4:
        b = c = 1
    elif a == 2:
        b = c = 2
    m >>= 3
    a = (m & 7) - b
    if a < 0:
        return False
    for i in range(6):
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

cdef _is_atama_mentsu(int nn, int m):
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

cdef _to_meld(int[:] tiles, int d):
    cdef int result = 0
    cdef int i
    for i in range(9):
        result |= (tiles[d + i] << i * 3)
    return result

cdef check_agari_from_keep_nums_cython_in_c(int[:] tiles):
    cdef int j = (1 << tiles[27]) | (1 << tiles[28]) | (1 << tiles[29]) | (1 << tiles[30]) | \
        (1 << tiles[31]) | (1 << tiles[32]) | (1 << tiles[33])
    if j >= 0x10:
        return False
    # 13 orphans
    if ((j & 3) == 2) and (tiles[0] * tiles[8] * tiles[9] * tiles[17] * tiles[18] *
        tiles[26] * tiles[27] * tiles[28] * tiles[29] * tiles[30] *
        tiles[31] * tiles[32] * tiles[33] == 2):
        return True
    # seven pairs
    cdef num_pair = 0
    cdef int i
    for i in range(34):
        if tiles[i] == 2:
            num_pair += 1
    if not (j & 10) and num_pair == 7:
        return True
    if j & 2:
        return False
    cdef int n00 = tiles[0] + tiles[3] + tiles[6]
    cdef int n01 = tiles[1] + tiles[4] + tiles[7]
    cdef int n02 = tiles[2] + tiles[5] + tiles[8]
    cdef int n10 = tiles[9] + tiles[12] + tiles[15]
    cdef int n11 = tiles[10] + tiles[13] + tiles[16]
    cdef int n12 = tiles[11] + tiles[14] + tiles[17]
    cdef int n20 = tiles[18] + tiles[21] + tiles[24]
    cdef int n21 = tiles[19] + tiles[22] + tiles[25]
    cdef int n22 = tiles[20] + tiles[23] + tiles[26]
    cdef int n0 = (n00 + n01 + n02) % 3
    if n0 == 1:
        return False
    cdef int n1 = (n10 + n11 + n12) % 3
    if n1 == 1:
        return False
    cdef int n2 = (n20 + n21 + n22) % 3
    if n2 == 1:
        return False
    if (int(n0 == 2) + int(n1 == 2) + int(n2 == 2) + int(tiles[27] == 2) + int(tiles[28] == 2) +
            int(tiles[29] == 2) + int(tiles[30] == 2) + int(tiles[31] == 2) + int(tiles[32] == 2) +
            int(tiles[33] == 2) != 1):
        return False
    cdef int nn0 = (n00 * 1 + n01 * 2) % 3
    cdef int m0 = _to_meld(tiles, 0)
    cdef int nn1 = (n10 * 1 + n11 * 2) % 3
    cdef int m1 = _to_meld(tiles, 9)
    cdef int nn2 = (n20 * 1 + n21 * 2) % 3
    cdef int m2 = _to_meld(tiles, 18)
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

def check_agari_from_keep_nums_cython(int[:] tile_type_keep_nums):
    return check_agari_from_keep_nums_cython_in_c(tile_type_keep_nums)

cdef acquire_waiting_tile_types_from_keep_nums_cython_in_c(int[:] tiles, bint multiple_return):
    cdef int *kokushi_list = [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33]
    cdef int kokushi_num = (tiles[0] + tiles[8] + tiles[9] + tiles[17] + tiles[18] + tiles[26] +
        tiles[27] + tiles[28] + tiles[29] + tiles[30] + tiles[31] + tiles[32] + tiles[33])
    cdef int it
    cdef int kokushi_kind = 0
    cdef int kokushi_tile_type = -1
    if kokushi_num == 13:
        for it in range(13):
            if tiles[kokushi_list[it]] > 0:
                kokushi_kind += 1
        if kokushi_kind == 13:
            return [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33]
        elif kokushi_kind == 12:
            for it in range(13):
                if tiles[kokushi_list[it]] == 0:
                    kokushi_tile_type = kokushi_list[it]
                    break
            return [kokushi_tile_type]
    # Now no kokushi
    cdef int num_candidates = 0
    cdef int waiting_type_candidates[13]
    cdef bint tile_walls[13]
    tile_walls[0] = False
    tile_walls[1] = False
    tile_walls[11] = False
    tile_walls[12] = False
    cdef int tile_type
    cdef int n
    cdef int i
    cdef bint flag = False
    for it in range(3):
        flag = False
        for n in range(9):
            if tiles[9*it+n] > 0:
                tile_walls[n+2] = True
                flag = True
            else:
                tile_walls[n+2] = False
        if not flag:
            continue
        for tile_type in range(9*it, 9*(it+1)):
            if tiles[tile_type] == 4:
                continue
            elif tiles[tile_type] > 0:
                waiting_type_candidates[num_candidates] = tile_type
                num_candidates += 1
            else: # tiles[tile_type] == 0
                i = tile_type - 9 * it + 2
                if ((tile_walls[i-2] and tile_walls[i-1]) or (tile_walls[i-1] and tile_walls[i+1]) or
                    (tile_walls[i+1] and tile_walls[i+2])):
                    waiting_type_candidates[num_candidates] = tile_type
                    num_candidates += 1
    for tile_type in range(27, 34):
        if 0 < tiles[tile_type] < 4:
            waiting_type_candidates[num_candidates] = tile_type
            num_candidates += 1
    cdef list waiting_type = []
    for it in range(num_candidates):
        tile_type = waiting_type_candidates[it]
        tiles[tile_type] += 1
        if check_agari_from_keep_nums_cython_in_c(tiles):
            waiting_type.append(tile_type)
            if not multiple_return:
                return waiting_type
        tiles[tile_type] -= 1
    return waiting_type

def acquire_waiting_tile_types_from_keep_nums_cython(int[:] tile_type_keep_nums, bint multiple_return):
    return acquire_waiting_tile_types_from_keep_nums_cython_in_c(tile_type_keep_nums, multiple_return)
