_base_ = './_template_/deeplabv3_r18_bdd100k-360x640.py'

experiment_cfg = dict(init_samples=600, num_query=300, num_cycles=6)
