#-*-coding:utf-8-*-

''' Constants
'''

NUM_PLAYER = 4

NUM_TILE = 136
NUM_TILE_TYPE = NUM_TILE // 4
NUM_WANG_TILE = 14
NUM_USABLE_TILE = NUM_TILE - NUM_WANG_TILE

NUM_STARTING_HAND_TILE = 13

AKADORAS = [16, 52, 88]

## AI-related
NUM_CALL_TYPE = 55
NUM_MELD_TYPE = 123

PAD_HAND = 14
PAD_MELD = 4
HAND_TILE_INFO_DIMENSION = 22
HAND_MELD_INFO_DIMENSION = 9

THROWN_MAX_LENGTH = 24
PAD_THROWN = THROWN_MAX_LENGTH + 10
PAD_CUM_MELD = 8
THROWN_TILE_INFO_DIMENSION = 15
CUM_MELD_INFO_DIMENSION = 9

ROUND_INFO_DIMENSION = 55
AGARI_INFO_DIMENSION = 6
#############

ROUND_NAMES = "東1,東2,東3,東4,南1,南2,南3,南4,西1,西2,西3,西4,北1,北2,北3,北4".split(",")
YAKUS = (
    # 一飜
    'mentsumo',        # 門前清自摸和
    'riichi',          # 立直
    'ippatsu',         # 一発
    'chakan',         # 槍槓
    'rinshan kaihou',  # 嶺上開花
    'haitei raoyue',   # 海底摸月
    'houtei raoyui',   # 河底撈魚
    'pinfu',           # 平和
    'tanyao',          # 断幺九
    'iipeiko',         # 一盃口
    # seat winds
    'ton',             # 自風 東
    'nan',             # 自風 南
    'xia',             # 自風 西
    'pei',             # 自風 北
    # round winds
    'ton',             # 場風 東
    'nan',             # 場風 南
    'xia',             # 場風 西
    'pei',             # 場風 北
    'haku',            # 役牌 白
    'hatsu',           # 役牌 發
    'chun',            # 役牌 中
    # 二飜
    'daburu riichi',   # 両立直
    'chiitoitsu',      # 七対子
    'chanta',          # 混全帯幺九
    'ittsu',           # 一気通貫
    'sanshoku doujun', # 三色同順
    'sanshoku doukou', # 三色同刻
    'sankantsu',       # 三槓子
    'toitoi',          # 対々和
    'sanankou',        # 三暗刻
    'shousangen',      # 小三元
    'honroutou',       # 混老頭
    # 三飜
    'ryanpeikou',      # 二盃口
    'junchan',         # 純全帯幺九
    'honitsu',         # 混一色
    # 六飜
    'chinitsu',        # 清一色
    # 満貫
    'renhou',          # 人和
    # 役満
    'tenhou',                # 天和
    'chihou',                # 地和
    'daisangen',             # 大三元
    'suuankou',              # 四暗刻
    'suuankou tanki',        # 四暗刻単騎
    'tsuuiisou',             # 字一色
    'ryuuiisou',             # 緑一色
    'chinroutou',            # 清老頭
    'chuuren pouto',         # 九蓮宝燈
    'chuuren pouto 9-wait',  # 純正九蓮宝燈
    'kokushi musou',         # 国士無双
    'kokushi musou 13-wait', # 国士無双１３面
    'daisuushi',             # 大四喜
    'shousuushi',            # 小四喜
    'suukantsu',             # 四槓子
    # 懸賞役
    'dora',    # ドラ
    'uradora', # 裏ドラ
    'akadora', # 赤ドラ
)
LIMITS=",mangan,haneman,baiman,sanbaiman,yakuman".split(",")
TILES_UNICODE = """
    🀇 🀈 🀉 🀊 🀋 🀌 🀍 🀎 🀏 
    🀙 🀚 🀛 🀜 🀝 🀞 🀟 🀠 🀡
    🀐 🀑 🀒 🀓 🀔 🀕 🀖 🀗 🀘
    🀀 🀁 🀂 🀃
    🀆 🀅 🀄
    🀫
    """.split()
TILES_STRING = """
    1m 2m 3m 4m 5m 6m 7m 8m 9m
    1p 2p 3p 4p 5p 6p 7p 8p 9p
    1s 2s 3s 4s 5s 6s 7s 8s 9s
    ew sw ww nw
    wd gd rd
    xx
    """.split()
