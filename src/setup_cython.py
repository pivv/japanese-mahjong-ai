from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
ext_modules = [
    Extension('core.game.mahjong_logic_cython', ['./core/game/mahjong_logic_cython.pyx'],)
    ]
setup(name = 'mahjong',
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules)
