_base_ = './_template_/segformer_mit-b1_voc2012-480x480.py'

experiment_cfg = dict(init_samples=150, num_query=75, num_cycles=6)
