# 变化检测数据集预处理

import os.path as osp

from common import (get_default_parser, add_crop_options, crop_patches,
                    get_path_tuples, create_file_list, link_dataset)

SUBSETS = ('train', 'val', 'test')
SUBDIRS = ('A', 'B', 'label')
FILE_LIST_PATTERN = "{subset}.txt"

crop_size = None
crop_stride = None

in_data_dir = '/home/pkc/AJ/2024/datasets/2409/wound/train/w0822'
out_data_dir = '/home/pkc/AJ/2024/datasets/2409/wound/train/train0822'

if __name__ == '__main__':

    out_dir = osp.join(out_data_dir,
                       osp.basename(osp.normpath(in_data_dir)))

    if crop_size is not None:
        crop_patches(
            crop_size,
            crop_stride,
            data_dir= in_data_dir,
            out_dir=out_dir,
            subsets=SUBSETS,
            subdirs=SUBDIRS,
            glob_pattern='*.png',
            max_workers=0)
    else:
        link_dataset(in_data_dir, out_data_dir)

    for subset in SUBSETS:
        path_tuples = get_path_tuples(
            *(osp.join(out_dir, subset, subdir) for subdir in SUBDIRS),
            glob_pattern='**/*.png',
            data_dir=out_data_dir)
        file_list = osp.join(
            out_data_dir, FILE_LIST_PATTERN.format(subset=subset))
        create_file_list(file_list, path_tuples)
        print(f"Write file list to {file_list}.")