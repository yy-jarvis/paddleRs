import os
import glob
from pathlib import Path

import cv2
import numpy as np


def calc_dice_by_iou(iou):
    """
    dice = (2 * iou) / (1 + iou)
    Args:
        iou:

    Returns:

    """

    return 2 * iou / (1 + iou)


print(calc_dice_by_iou(0.537616))