_base_ = './_template_/segformer_mit-b1_bdd100k-360x640.py'

experiment_cfg = dict(init_samples=700, num_query=350, num_cycles=6)
