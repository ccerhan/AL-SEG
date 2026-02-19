_base_ = './_template_/segformer_mit-b1_cityscapes-512x1024.py'

experiment_cfg = dict(init_samples=300, num_query=150, num_cycles=6)
