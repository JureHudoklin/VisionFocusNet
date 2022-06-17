#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

# from .layers import TDID_Embedding, ROI_Align
# from .utils import compute_iou, convert_to_xywh, roi_align, decode_bb

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, 'configs'))
from tdid_basic_config import Config

DATA_PATH = "/datasets/AVD/data_loader"
sys.path.append(DATA_PATH)
from active_vision_dataset import AVD_Loader, plot_BB

DATA_PATH_2 = "/datasets/GMU_Kitchens"
sys.path.append(DATA_PATH_2)
from gmu_kitchens import GMU_Loader


def build_