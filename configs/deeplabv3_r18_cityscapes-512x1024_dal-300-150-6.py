_base_ = './_template_/deeplabv3_r18_cityscapes-512x1024.py'

experiment_cfg = dict(init_samples=300, num_query=150, num_cycles=6)
