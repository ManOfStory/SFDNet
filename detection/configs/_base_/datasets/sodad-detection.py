dataset_type = 'SODADDataset'
data_root = "/home/yangzh/Datasets/SODA-D/"
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

file_client_args = None
backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args = backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1200, 1200), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args = backend_args),
    dict(type = 'Resize', scale = (1200, 1200), keep_ratio = True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type = 'PackDetInputs', meta_keys = ('img_id','img_path','ori_shape','img_shape','scale_factor'))
]
train_dataloader = dict(
    batch_size = 2,
    num_workers = 4,
    persistent_workers = True,
    sampler = dict(type='DefaultSampler', shuffle = True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset = dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='divData/Annotations/train.json',
        ori_ann_file = data_root + '/rawData/Annotations/train.json',
        data_prefix=dict(img = 'divData/Images/train'),
        pipeline = train_pipeline,
        backend_args = backend_args
    )
)
val_dataloader = dict(
    batch_size = 1,
    num_workers = 2,
    persistent_workers = True,
    drop_last = False,
    sampler = dict(type='DefaultSampler', shuffle = False),
    dataset = dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='divData/Annotations/test.json',
        ori_ann_file=data_root + '/rawData/Annotations/test_wo_ignore.json',
        data_prefix=dict(img = 'divData/Images/test/'),
        test_mode = True,
        pipeline = test_pipeline,
        backend_args = backend_args
    )
)
test_dataloader = val_dataloader


val_evaluator = dict(
    type='SODADMetric',
    ann_file=data_root + 'divData/Annotations/test.json',
    ori_ann_file=data_root + 'rawData/Annotations/test_wo_ignore.json',
    metric=['bbox'],
    format_only=False,
    backend_args=backend_args)
test_evaluator = val_evaluator