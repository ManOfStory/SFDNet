from .coco import CocoDataset

import itertools
import logging
from collections import OrderedDict

import numpy as np
from aitodpycocotools.cocoeval import COCOeval
from terminaltables import AsciiTable

from mmdet.registry import DATASETS


@DATASETS.register_module()
class AITODDataset(CocoDataset):
    METAINFO = {
        'classes':
        ('airplane', 'bridge', 'storage-tank', 'ship', 'swimming-pool', 'vehicle', 'person', 'wind-mill'),
        # palette is a list of color tuples, which is used for visualization.
        'palette':
        [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
         (0, 60, 100), (0, 80, 100), (0, 0, 70)]
    }