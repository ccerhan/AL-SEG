_base_ = [
    '../_base_/models/deeplabv3_r50.py',
    '../_base_/datasets/voc2012_480x480.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_50e.py',
    '../_base_/default_experiment.py',
]

crop_size = (480, 480)

data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=21),
    auxiliary_head=dict(num_classes=21),
)
