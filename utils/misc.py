
"""
Misc functions, including distributed helpers.
Mostly copy-paste from torchvision references.
"""

import os
import subprocess
import time
from collections import defaultdict, deque
import datetime
import pickle
from typing import Optional, List

import torch
from torch import Tensor


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        """
        Packs in tensor and mask into a single tensor object.
        """
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)