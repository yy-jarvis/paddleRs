#!/usr/bin/env python

# 变化检测模型CDNet训练示例脚本
# 执行此脚本前，请确认已正确安装PaddleRS库

import paddlers as pdrs
from paddlers import transforms as T

import paddle

# 数据集存放目录
DATA_DIR = '/home/pkc/AJ/2024/datasets/2409/wound/train/train0822'
# 训练集`file_list`文件路径
TRAIN_FILE_LIST_PATH = '{}/train.txt'.format(DATA_DIR)
# 验证集`file_list`文件路径
EVAL_FILE_LIST_PATH = '{}/val.txt'.format(DATA_DIR)
# 实验目录，保存输出的模型权重和结果
EXP_DIR = '{}/output/cdnet/'.format(DATA_DIR)



# 定义训练和验证时使用的数据变换（数据增强、预处理等）
# 使用Compose组合多种变换方式。Compose中包含的变换将按顺序串行执行
# API说明：https://github.com/PaddlePaddle/PaddleRS/blob/develop/docs/apis/data.md
train_transforms = [
    # 随机裁剪
    T.RandomCrop(
        # 裁剪区域将被缩放到256x256
        crop_size=256,
        # 裁剪区域的横纵比在0.5-2之间变动
        aspect_ratio=[0.5, 2.0],
        # 裁剪区域相对原始影像长宽比例在一定范围内变动，最小不低于原始长宽的1/5
        scaling=[0.2, 1.0]),
    # 随机交换
    T.RandomSwap(),
    # 以50%的概率实施随机水平翻转
    T.RandomHorizontalFlip(prob=0.5),
    # 将数据归一化到[-1,1]
    T.Normalize(
        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
]

eval_transforms = [
    # 验证阶段与训练阶段的数据归一化方式必须相同
    T.Normalize(
        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    T.ReloadMask()
]

# 分别构建训练和验证所用的数据集
train_dataset = pdrs.datasets.CDDataset(
    data_dir=DATA_DIR,
    file_list=TRAIN_FILE_LIST_PATH,
    label_list=None,
    transforms=train_transforms,
    num_workers=0,
    shuffle=True,
    with_seg_labels=False,
    binarize_labels=True)

eval_dataset = pdrs.datasets.CDDataset(
    data_dir=DATA_DIR,
    file_list=EVAL_FILE_LIST_PATH,
    label_list=None,
    transforms=eval_transforms,
    num_workers=0,
    shuffle=False,
    with_seg_labels=False,
    binarize_labels=True)

# 使用默认参数构建CDNet模型
# 目前已支持的模型请参考：https://github.com/PaddlePaddle/PaddleRS/blob/develop/docs/intro/model_zoo.md
# 模型输入参数请参考：https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/tasks/change_detector.py

use_mixed_loss = [('CrossEntropyLoss', 0.8), ('DiceLoss', 0.2)]
model = pdrs.tasks.cd.CDNet(
    use_mixed_loss=use_mixed_loss,

)


pre_path = '/home/pkc/AJ/2024/datasets/2409/wound/train/train0822/output/cdnet-0.874/best_model'


# 制定定步长学习率衰减策略
lr_scheduler = paddle.optimizer.lr.StepDecay(
    0.01,
    step_size=100,
    # 学习率衰减系数，这里指定每次减半
    gamma=0.01
)

# 构造AdamW优化器
optimizer = paddle.optimizer.Adamax(
    learning_rate=0.01,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-08,
    parameters=model.net.parameters(),
    # weight_decay=0.01,
    # apply_decay_param_fun=None,
    # grad_clip=None,
    # lazy_mode=False,
    # name=None
)




# 执行模型训练
model.train(
    num_epochs=200,
    train_dataset=train_dataset,
    train_batch_size=16,
    eval_dataset=eval_dataset,
    optimizer=optimizer,
    # learning_rate=0.9,   # 0.1
    # lr_decay_power=0.09,
    save_interval_epochs=10,
    pretrain_weights='IMAGENET',
    # 每多少次迭代记录一次日志
    log_interval_steps=100,
    save_dir=EXP_DIR,
    # 是否使用early stopping策略，当精度不再改善时提前终止训练
    early_stop=False,
    # 是否启用VisualDL日志功能
    use_vdl=True,
    # 指定从某个检查点继续训练
    resume_checkpoint=None)

model.evaluate(eval_dataset=eval_dataset, batch_size=4)
