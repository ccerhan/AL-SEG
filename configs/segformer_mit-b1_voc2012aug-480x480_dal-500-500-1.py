_base_ = './_template_/segformer_mit-b1_voc2012aug-480x480.py'

experiment_cfg = dict(init_samples=500, num_query=500, num_cycles=1)
