dataset_type = 'AITODDataset'
data_root = "/home/yangzh/Datasets/AI-TOD/"
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

file_client_args = None
backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args = backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(800, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args = backend_args),
    dict(type = 'Resize', scale = (800, 800), keep_ratio = True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type = 'PackDetInputs', meta_keys = ('img_id','img_path','ori_shape','img_shape','scale_factor'))
]
train_dataloader = dict(
    batch_size = 4,
    num_workers = 8,
    persistent_workers = True,
    sampler = dict(type='DefaultSampler', shuffle = True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset = dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations-v1/aitod_trainval_v1.json',
        data_prefix=dict(img = 'trainval/'),
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
        ann_file='annotations-v1/aitod_test_v1.json',
        data_prefix=dict(img = 'test/'),
        test_mode = True,
        pipeline = test_pipeline,
        backend_args = backend_args
    )
)
test_dataloader = val_dataloader


val_evaluator = dict(
    type='AITODMetric',
    ann_file=data_root + 'annotations-v1/aitod_test_v1.json',
    metric=['bbox'],
    format_only=False,
    backend_args=backend_args)
test_evaluator = val_evaluator