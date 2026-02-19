_base_ = [
    '../_base_/models/segformer_mit-b0.py',
    '../_base_/datasets/cityscapes_344x688.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_50e.py',
    '../_base_/default_experiment.py',
]

crop_size = (344, 688)
data_preprocessor = dict(size=crop_size)

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth'

model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(init_cfg=dict(type='Pretrained', checkpoint=checkpoint)),
)

optimizer = dict(betas=(0.9, 0.999), lr=6e-05, type='AdamW', weight_decay=0.01)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=10.0),
            norm=dict(decay_mult=0.0),
            pos_block=dict(decay_mult=0.0)
        )
    )
)

param_scheduler = [
    dict(type='PolyLR', eta_min=1e-07, power=1.0),
]

train_dataloader = dict(batch_size=1, num_workers=4)
val_dataloader = dict(batch_size=1, num_workers=4)
query_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = query_dataloader
