_base_ = './segformer_mit-b0_ade20k-512x512.py'

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b1_20220624-02e5a6a1.pth'

model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        embed_dims=64),
    decode_head=dict(in_channels=[64, 128, 320, 512]))
