_base_ = './deeplabv3_r50_bdd100k-360x640.py'

model = dict(
    pretrained='torchvision://resnet18',
    backbone=dict(type='ResNet', depth=18),
    decode_head=dict(in_channels=512, channels=128),
    auxiliary_head=dict(in_channels=256, channels=64)
)
