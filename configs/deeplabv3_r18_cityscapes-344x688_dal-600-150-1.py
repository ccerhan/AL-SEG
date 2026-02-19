_base_ = './_template_/deeplabv3_r18_cityscapes-344x688.py'

experiment_cfg = dict(init_samples=600, num_query=150, num_cycles=1)
