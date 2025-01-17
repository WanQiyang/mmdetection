_base_ = [
    "../../configs/_base_/default_runtime.py",
]

# model settings
model = dict(
    type='CornerNet',
    backbone=dict(
        type='HourglassNet',
        downsample_times=5,
        num_stacks=2,
        stage_channels=[256, 256, 384, 384, 384, 512],
        stage_blocks=[2, 2, 2, 2, 2, 4],
        norm_cfg=dict(type='BN', requires_grad=True)),
    neck=None,
    bbox_head=dict(
        type='CornerHead',
        num_classes=12,
        in_channels=256,
        num_feat_levels=2,
        corner_emb_channels=1,
        loss_heatmap=dict(
            type='GaussianFocalLoss', alpha=2.0, gamma=4.0, loss_weight=1),
        loss_embedding=dict(
            type='AssociativeEmbeddingLoss',
            pull_weight=0.10,
            push_weight=0.10),
        loss_offset=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1)),
    # training and testing settings
    train_cfg=None,
    test_cfg=dict(
        corner_topk=100,
        local_maximum_kernel=3,
        distance_threshold=0.5,
        score_thr=0.05,
        max_per_img=100,
        nms=dict(type='soft_nms', iou_threshold=0.5, method='gaussian')))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)

train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(
        type='RandomCenterCropPad',
        crop_size=(511, 511),
        ratios=(0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3),
        test_mode=False,
        test_pad_mode=None,
        **img_norm_cfg),
    dict(type='Resize', img_scale=(511, 511), keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(
        type='MultiScaleFlipAug',
        scale_factor=1.0,
        flip=True,
        transforms=[
            dict(type='Resize'),
            dict(
                type='RandomCenterCropPad',
                crop_size=None,
                ratios=None,
                border=None,
                test_mode=True,
                test_pad_mode=['logical_or', 127],
                **img_norm_cfg),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(
                type='Collect',
                keys=['img'],
                meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape',
                           'scale_factor', 'flip', 'img_norm_cfg', 'border')),
        ])
]

dataset_type = "CustomDataset"
data_root = "/data/CUB/"

classes = (
    "back",
    "beak",
    "belly",
    "breast",
    "crown",
    "forehead",
    "eye",
    "leg",
    "wing",
    "nape",
    "tail",
    "throat",
)

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=3,
    train=dict(
        type=dataset_type,
        ann_file="bbox_annotation_train_12.pkl",
        pipeline=train_pipeline,
        classes=classes,
        data_root=data_root,
        img_prefix="images/",
        filter_empty_gt=False,
    ),
    val=dict(
        type=dataset_type,
        ann_file="bbox_annotation_test_12.pkl",
        pipeline=test_pipeline,
        classes=classes,
        data_root=data_root,
        img_prefix="images/",
        filter_empty_gt=False,
    ),
    test=dict(
        type=dataset_type,
        ann_file="bbox_annotation_test_12.pkl",
        pipeline=test_pipeline,
        classes=classes,
        data_root=data_root,
        img_prefix="images/",
        filter_empty_gt=False,
    ),
)

# evaluation = dict(interval=1, metric="mAP")

# optimizer
optimizer = dict(type='Adam', lr=0.0005)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[180])
runner = dict(type='EpochBasedRunner', max_epochs=210)

load_from = "workspace/checkpoints/cornernet_hourglass104_mstest_10x5_210e_coco_20200824_185720-5fefbf1c.pth"

