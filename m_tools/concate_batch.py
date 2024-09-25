# 拼接A\B\P\V的图像：2×2

import numpy as np
import cv2

import glob
from pathlib import Path

from tqdm import tqdm

# 读取原图和二值化掩码
path_root = '/home/pkc/AJ/2024/datasets/2409/wound/20240920wound/cdData'
A_image_path = '{}/A'.format(path_root)
B_image_path = '{}/B'.format(path_root)
P_image_path = '{}/P'.format(path_root)
V_image_path = '{}/V'.format(path_root)
C_path = '{}/C'.format(path_root)

files = glob.glob(P_image_path + '/*.png')
for file in tqdm(files):
    file_name = Path(file).name

    file_A = '{}/{}'.format(A_image_path, file_name)
    file_B = '{}/{}'.format(B_image_path, file_name)
    file_V = '{}/{}'.format(V_image_path, file_name)
    file_C = '{}/{}'.format(C_path, file_name)

    # 加载图像，这里假设所有图像的尺寸都相同
    # 替换为你的图像路径
    image1 = cv2.imread(file_A)
    image2 = cv2.imread(file_B)
    image3 = cv2.imread(file)
    image4 = cv2.imread(file_V)

    # 确保所有图像都已正确加载
    if image1 is None or image2 is None or image3 is None or image4 is None:
        print("Error: One of the images did not load.")
        exit()

        # 水平拼接图像1和图像2，图像3和图像4
    h_combined1 = np.concatenate((image1, image2), axis=1)
    h_combined2 = np.concatenate((image3, image4), axis=1)

    # 垂直拼接上面得到的两个水平拼接图像
    v_combined = np.concatenate((h_combined1, h_combined2), axis=0)

    cv2.imwrite(file_C, v_combined)