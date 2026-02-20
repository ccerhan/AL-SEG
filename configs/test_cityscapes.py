_base_ = './_template_/deeplabv3_r18_cityscapes-344x688.py'

default_hooks = dict(
    logger=dict(interval=1),
    checkpoint=dict(save_begin=1, interval=1),
    # visualization=dict(type='SegVisualizationHook', draw=True, interval=100)
)

query_dataloader = dict(batch_size=1)
# val_dataloader = dict(batch_size=1)

train_cfg = dict(max_epochs=1, val_begin=1, val_interval=1)

query_cfg = dict(type='Random')

experiment_cfg = dict(init_samples=50, num_query=50, num_cycles=3)
