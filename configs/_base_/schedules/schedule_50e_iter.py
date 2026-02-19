num_iters = 160000

# optimizer
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

# learning policy
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-06, begin=0, end=1500, by_epoch=False),
    dict(type='PolyLR', eta_min=0.0, power=1.0, begin=1500, end=num_iters, by_epoch=False),
]

# training schedule
train_cfg = dict(type='IterBasedTrainLoop', max_iters=num_iters, val_interval=1000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, save_last=False, save_best='mIoU', max_keep_ckpts=1,
                    save_optimizer=False, save_param_scheduler=False, interval=1000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))
