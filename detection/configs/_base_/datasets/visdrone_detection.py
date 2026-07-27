dataset_type = 'VisDroneDataset'
data_root = '/mnt/ssd1/guoyang/Dataset/RawDataset/Visdrone2019/'
file_client_args = None
backend_args = None
classes = ('pedestrian', 'people', 'bicycle', 'car', 'van',
             'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor')

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)


albu_train_transforms = [
    dict(
        type='ShiftScaleRotate',
        shift_limit=0.0625,
        scale_limit=0.1,
        rotate_limit=45,
        interpolation=1,
        p=0.5),
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=[0.1, 0.3],
        contrast_limit=[0.1, 0.3],
        p=0.2),
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='RGBShift',
                r_shift_limit=10,
                g_shift_limit=10,
                b_shift_limit=10,
                p=1.0),
            dict(
                type='HueSaturationValue',
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1.0)
        ],
        p=0.1),
    dict(type='JpegCompression', quality_lower=85, quality_upper=95, p=0.2),
    dict(type='ChannelShuffle', p=0.1),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=3, p=1.0),
            dict(type='MedianBlur', blur_limit=3, p=1.0)
        ],
        p=0.1),
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1536, 1536), keep_ratio=True),
    dict(type='Pad', size=(1536,1536), pad_val=dict(img=(114,114,114))),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            # 'gt_masks': 'masks',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=True),
    dict(type='PackDetInputs'),
]

test_pipeline = [ 
    dict(type='LoadImageFromFile'), 
    dict(type='MultiScaleFlipAug', 
         img_scale=(1536, 1536), 
         flip=True, 
         transforms=[ 
            dict(type='Resize', keep_ratio=True), 
            dict(type='RandomFlip'), 
            dict(type='Normalize', **img_norm_cfg), 
            dict(type='Pad', size=(1536,1536), pad_val=dict(img=(114,114,114))), 
            dict(type='PackDetInputs')
        ]
    ) 
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1536, 1536), keep_ratio=True),
    dict(type='Pad', size=(1536,1536), pad_val=dict(img=(114,114,114))),
    dict(type='PackDetInputs')
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'VisDrone2019-DET-train/train.json',
        img_prefix=data_root + 'VisDrone2019-DET-train/images/',
        pipeline=train_pipeline,
        ),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'VisDrone2019-DET-val/val.json',
        img_prefix=data_root + 'VisDrone2019-DET-val/images/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'VisDrone2019-DET-val/val.json',
        img_prefix=data_root + 'VisDrone2019-DET-val/images/',
        pipeline=test_pipeline))

train_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='VisDrone2019-DET-train/train.json',
        data_prefix=dict(img='VisDrone2019-DET-train/images/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='VisDrone2019-DET-val/val.json',
        data_prefix=dict(img='VisDrone2019-DET-val/images/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'VisDrone2019-DET-val/val.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args)
test_evaluator = val_evaluator