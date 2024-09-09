import glob
from pathlib import Path

import paddlers as pdrs
from paddlers import transforms as T

import numpy as np
import cv2
from PIL import Image


import paddle

from paddlers.rs_models.cd import CDNet

transforms = T.Compose([
    T.Resize(target_size=256),
    T.Normalize(
        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


state_dict_path = '/home/pkc/AJ/2024/datasets/2408/wound/crop_train/output/20240820-3-b/best_model/model.pdparams'
path_test = '/home/pkc/AJ/2024/datasets/2408/wound/240822/paddlers'
path_img = '20240729_extract+000008_crop.png'
A = '{}/A'.format(path_test)
B = '{}/B'.format(path_test)
P = '{}/P'.format(path_test)

eval_transforms = T.Compose([
    T.DecodeImg(),
    T.Resize(target_size=256),
    T.Normalize(
        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    T.ArrangeChangeDetector('test')
])

def predict():
    model = CDNet(6, 2)

    state_dict = paddle.load(state_dict_path)
    model.set_state_dict(state_dict)
    model.eval()

    files = glob.glob('{}/*.png'.format(A))
    for file in files:
        file_name = Path(file).name
        file_B = '{}/{}'.format(B, file_name)
        fileC = '{}/{}'.format(P, file_name)

        image = {'image_t1': file, 'image_t2': file_B}
        image = eval_transforms(image)
        t1 = paddle.to_tensor(image[0]['image']).unsqueeze(0)
        t2 = paddle.to_tensor(image[0]['image2']).unsqueeze(0)

        with paddle.no_grad():
            pred = paddle.argmax(model(t1, t2)[-1], 1)[0].numpy()
        vis = pred.astype("uint8") * 255
        cv2.imwrite(fileC, vis)

predict()