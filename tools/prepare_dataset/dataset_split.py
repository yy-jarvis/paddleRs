import os
import shutil
import random


def create_dir_structure(base_path):
    for sub_dir in ["train", "val", "test"]:
        for folder in ["A", "B", "label"]:
            os.makedirs(os.path.join(base_path, sub_dir, folder), exist_ok=True)


def split_dataset(base_path, train_ratio=0.85, val_ratio=0.1):
    A_path = os.path.join(base_path, "A")
    B_path = os.path.join(base_path, "B")
    label_path = os.path.join(base_path, "label")

    filenames = os.listdir(A_path)

    # Ensure filenames are the same in A, B, and label
    assert set(filenames) == set(os.listdir(B_path)) == set(os.listdir(label_path))

    random.shuffle(filenames)

    total_count = len(filenames)
    train_split = int(total_count * train_ratio)
    val_split = int(total_count * (train_ratio + val_ratio))

    train_files = filenames[:train_split]
    val_files = filenames[train_split:val_split]
    test_files = filenames[val_split:]

    return train_files, val_files, test_files


def copy_files(file_list, source_dir, dest_dir):
    for filename in file_list:
        shutil.copy(os.path.join(source_dir, filename), os.path.join(dest_dir, filename))


def main(base_path):
    create_dir_structure(base_path)

    train_files, val_files, test_files = split_dataset(base_path)

    for filename in train_files:
        copy_files([filename], os.path.join(base_path, "A"), os.path.join(base_path, "train", "A"))
        copy_files([filename], os.path.join(base_path, "B"), os.path.join(base_path, "train", "B"))
        copy_files([filename], os.path.join(base_path, "label"), os.path.join(base_path, "train", "label"))

    for filename in val_files:
        copy_files([filename], os.path.join(base_path, "A"), os.path.join(base_path, "val", "A"))
        copy_files([filename], os.path.join(base_path, "B"), os.path.join(base_path, "val", "B"))
        copy_files([filename], os.path.join(base_path, "label"), os.path.join(base_path, "val", "label"))

    for filename in test_files:
        copy_files([filename], os.path.join(base_path, "A"), os.path.join(base_path, "test", "A"))
        copy_files([filename], os.path.join(base_path, "B"), os.path.join(base_path, "test", "B"))
        copy_files([filename], os.path.join(base_path, "label"), os.path.join(base_path, "test", "label"))


if __name__ == "__main__":
    base_path = "/home/pkc/AJ/2024/datasets/2408/wound/crop"
    main(base_path)

