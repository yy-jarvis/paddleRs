import os.path as osp

from common import (get_default_parser, add_crop_options, crop_patches,
                    get_path_tuples, create_file_list, link_dataset)

SUBSETS = ('train', 'val', 'test')
SUBDIRS = ('A', 'B', 'label')
FILE_LIST_PATTERN = "{subset}.txt"
crop_size = None
crop_stride = None

in_dataset_dir = '/home/pkc/AJ/2024/datasets/2408/wound/crop'
out_dataset_dir = '/home/pkc/AJ/2024/datasets/2408/wound/crop_train'

if __name__ == '__main__':


    out_dir = osp.join(out_dataset_dir,
                       osp.basename(osp.normpath(in_dataset_dir)))

    if crop_size is not None:
        crop_patches(
            crop_size,
            crop_stride,
            data_dir=in_dataset_dir,
            out_dir=out_dir,
            subsets=SUBSETS,
            subdirs=SUBDIRS,
            glob_pattern='*.png',
            max_workers=0)
    else:
        link_dataset(in_dataset_dir, out_dataset_dir)

    for subset in SUBSETS:
        path_tuples = get_path_tuples(
            *(osp.join(out_dir, subset, subdir) for subdir in SUBDIRS),
            glob_pattern='**/*.png',
            data_dir=out_dataset_dir)
        file_list = osp.join(
            out_dataset_dir, FILE_LIST_PATTERN.format(subset=subset))
        create_file_list(file_list, path_tuples)
        print(f"Write file list to {file_list}.")