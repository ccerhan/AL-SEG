_base_ = [
    '../_base_/models/deeplabv3_r50.py',
    '../_base_/datasets/cityscapes_512x1024.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_50e.py',
    '../_base_/default_experiment.py',
]

crop_size = (512, 1024)

data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=data_preprocessor)
