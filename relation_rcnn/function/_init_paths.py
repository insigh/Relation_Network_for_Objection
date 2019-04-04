# --------------------------------------------------------
# Relation Networks for Object Detection
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiayuan Gu, Dazhi Cheng
# --------------------------------------------------------

import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

lib_path = osp.join(this_dir, '..', 'lib')
lib_path = osp.join(this_dir, '..')
add_path(lib_path)