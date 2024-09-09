import numpy as np
import cv2

import glob
from pathlib import Path


def read_image(image_path, is_gray=False):
    """
    读取图像。
    """
    if is_gray:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(image_path)
    return img


def overlay_masks(original_img, pred_mask, label_mask):
    """
    将预测和标注的二值化图像绘制到原图上。
    """
    # 确保原图是彩色图像
    if len(original_img.shape) == 2:
        original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)

    # 将预测和标注的二值化图像转换为彩色
    pred_color = np.zeros_like(original_img)
    label_color = np.zeros_like(original_img)

    pred_color[pred_mask] = [0, 0, 255]  # 红色
    label_color[label_mask] = [0, 255, 0]  # 绿色

    # 将预测和标注图像叠加到原图上
    overlay_img = cv2.addWeighted(original_img, 1.0, pred_color, 0.5, 0)
    overlay_img = cv2.addWeighted(overlay_img, 1.0, label_color, 0.5, 0)

    return overlay_img


# 读取原图和二值化掩码
path_root = '/home/pkc/AJ/2024/datasets/2408/wound/240822/paddlers'
original_image_path = '{}/B'.format(path_root)
pred_image_path = '{}/P'.format(path_root)
label_image_path = '{}/label'.format(path_root)
visual_path = '{}/V'.format(path_root)

files = glob.glob('{}/*.png'.format(pred_image_path))
for file in files:
    file_name = Path(file).name
    original_img = read_image('{}/{}'.format(original_image_path, file_name), is_gray=True)

    pred_mask = read_image(file, is_gray=True) > 127  # 转为二值化掩码
    # label_mask = read_image('{}/{}'.format(label_image_path, file_name), is_gray=True) > 127  # 转为二值化掩码
    label_mask = np.zeros_like(original_img)

    # 确保掩码和原图的大小一致
    assert original_img.shape[:2] == pred_mask.shape
    assert original_img.shape[:2] == label_mask.shape

    # 绘制掩码到原图上
    overlay_img = overlay_masks(original_img, pred_mask, label_mask)

    changed_img = read_image('{}/{}'.format(original_image_path, file_name), is_gray=False)
    concate_img = np.hstack((changed_img, overlay_img))

    # 保存结果
    cv2.imwrite('{}/{}'.format(visual_path, file_name), concate_img)
