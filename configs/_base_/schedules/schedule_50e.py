num_epochs = 50

# optimizer
optimizer = dict(type='Adam', lr=0.0005, betas=[0.9, 0.999], weight_decay=0.0002, eps=1e-07)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)

# learning policy
param_scheduler = [
    dict(type='PolyLR', eta_min=1e-05, power=0.9, begin=0, end=num_epochs, by_epoch=True)
]

# training schedule for 50 epochs
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=num_epochs, val_begin=num_epochs - 10, val_interval=2)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=True),
    checkpoint=dict(type='CheckpointHook', by_epoch=True, save_last=False, save_best='mIoU', max_keep_ckpts=1,
                    save_optimizer=False, save_param_scheduler=False, save_begin=num_epochs - 10, interval=2),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))
