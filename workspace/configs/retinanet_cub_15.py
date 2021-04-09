_base_ = [
    "../../configs/_base_/models/retinanet_r50_fpn.py",
    "../../configs/_base_/schedules/schedule_1x.py",
    "../../configs/_base_/default_runtime.py",
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
    "left eye",
    "left leg",
    "left wing",
    "nape",
    "right eye",
    "right leg",
    "right wing",
    "tail",
    "throat",
)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="Resize", img_scale=(1333, 800), keep_ratio=True),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file="bbox_annotation_train.pkl",
        pipeline=train_pipeline,
        classes=classes,
        data_root=data_root,
        img_prefix="images/",
        filter_empty_gt=False,
    ),
    val=dict(
        type=dataset_type,
        ann_file="bbox_annotation_test.pkl",
        pipeline=test_pipeline,
        classes=classes,
        data_root=data_root,
        img_prefix="images/",
        filter_empty_gt=False,
    ),
    test=dict(
        type=dataset_type,
        ann_file="bbox_annotation_test.pkl",
        pipeline=test_pipeline,
        classes=classes,
        data_root=data_root,
        img_prefix="images/",
        filter_empty_gt=False,
    ),
)

evaluation = dict(interval=1, metric="mAP")

model = dict(pretrained=None, bbox_head=dict(num_classes=15))

load_from = "workspace/checkpoints/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth"

