# AL-SEG: Active Learning for Semantic Segmentation #

**AL-SEG** is an open-source active learning tool for semantic segmentation based on PyTorch. It is built on top of [mmsegmentation](https://github.com/open-mmlab/mmsegmentation), which provides extensive support for various datasets, pre-trained backbones, and state-of-the-art segmentation methods right out of the box.

This repository contains the **official implementation** of the mathod:
**Conformal Risk Controlled Active Learning (CRC-AL)**

```
@Article{ai6100270,
AUTHOR = {Erhan, Can and Ure, Nazim Kemal},
TITLE = {Reducing Annotation Effort in Semantic Segmentation Through Conformal Risk Controlled Active Learning},
JOURNAL = {AI},
VOLUME = {6},
YEAR = {2025},
NUMBER = {10},
ARTICLE-NUMBER = {270},
URL = {https://www.mdpi.com/2673-2688/6/10/270},
ISSN = {2673-2688},
ABSTRACT = {Modern semantic segmentation models require extensive pixel-level annotations, creating a significant barrier to practical deployment as labeling a single image can take hours of human effort. Active learning offers a promising way to reduce annotation costs through intelligent sample selection. However, existing methods rely on poorly calibrated confidence estimates, making uncertainty quantification unreliable. We introduce Conformal Risk Controlled Active Learning (CRC-AL), a novel framework that provides statistical guarantees on uncertainty quantification for semantic segmentation, in contrast to heuristic approaches. CRC-AL calibrates class-specific thresholds via conformal risk control, transforming softmax outputs into multi-class prediction sets with formal guarantees. From these sets, our approach derives complementary uncertainty representations: risk maps highlighting uncertain regions and class co-occurrence embeddings capturing semantic confusions. A physics-inspired selection algorithm leverages these representations with a barycenter-based distance metric that balances uncertainty and diversity. Experiments on Cityscapes and PascalVOC2012 show CRC-AL consistently outperforms baseline methods, achieving 95% of fully supervised performance with only 30% of labeled data, making semantic segmentation more practical under limited annotation budgets.},
DOI = {10.3390/ai6100270}
}
```

In addition to our proposed CRC-AL approach, this project also includes Python implementations of several **active learning algorithms** adapted for semantic segmentation outputs:

| Methods                   | References                                                                                                                                          |
|---------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|
| Random Sampling           | -                                                                                                                                                   |
| Entropy Sampling          | D. D. Lewis, J. Catlett, Heterogeneous uncertainty sampling for supervised learning, in: Machine Learning Proceedings, 1994.                        |
| Margin Sampling           | D. Roth, K. Small, Margin-based active learning for structured output spaces, in: Machine Learning: ECML, 2006.                                     |
| Least Confidence          | A. J. Joshi, F. Porikli, N. Papanikolopoulos, Multi-class active learning for image classification, in: IEEE CVPR, 2009.                            |
| Stochastic Batch Sampling | M. Gaillochet, C. Desrosiers, H. Lombaert, Active learning for medical image segmentation with stochastic batches, Medical Image Analysis, 2023.    |
| Core-Set Selection        | O. Sener, S. Savarese, Active learning for convolutional neural networks: A core-set approach, in: ICLR, 2018.                                      |
| Contextual Diversity      | S. Agarwal, H. Arora, S. Anand, C. Arora, Contextual diversity for active learning, in: Computer Vision – ECCV, 2020.                               |
| BADGE                     | J. T. Ash, C. Zhang, A. Krishnamurthy, J. Langford, A. Agarwal, Deep batch active learning by diverse, uncertain gradient lower bounds, CoRR, 2019. |

## Setup Environment ##

### Setup with CUDA (Nvidia) ###

This section applies if you have an Nvidia GPU and want to leverage CUDA for hardware acceleration.

```shell
conda env create -f install/env_cu116.yaml
conda activate AL-SEG
pip install mmengine==0.8.5
pip install mmcv==2.0.0rc4 -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.13/index.html
pip install mmsegmentation==1.1.2
```

### Setup with MPS (Apple Silicon) ###

This section is relevant for users on Apple Silicon (e.g., M1 or M2 Macs) who want to utilize Metal Performance Shaders (MPS) acceleration.

```shell
conda env create -f install/env_mps.yaml
conda activate AL-SEG
pip install mmengine==0.8.5
pip install mmcv==2.0.0rc4 -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.13/index.html
pip install mmsegmentation==1.1.2
```

### Verify Device ###

To confirm that your CUDA or MPS setup is functioning correctly and to run basic performance tests, execute:

```shell
python scripts/test_device.py
```

## Download Datasets ##

### Cityscapes ###

Cityscapes requires a registered account:
https://www.cityscapes-dataset.com/downloads

```shell
cd data/cityscapes
sh download.sh <USERNAME> <PASSWORD>
```

Prepare labels using the mmsegmentation converter:

```shell
# from data/cityscapes
python cityscapes.py .
```

### PascalVOC2012 ###

Download VOC 2012 and the Berkeley augmentation set:

```shell
cd data/VOCdevkit/VOC2012
sh download.sh
```

Build segmentation augmentation splits:

```shell
# from data/VOCdevkit/VOC2012
DEVKIT_PATH=../../VOCdevkit
AUG_PATH=benchmark_RELEASE
python voc_aug.py $DEVKIT_PATH $AUG_PATH
python voc_merge.py $DEVKIT_PATH
```

## Training with Entire Dataset ##

For full-dataset training, run `tools/train.py` without `--split`.
The script will use the default training annotations defined in the config.

If you are on Apple Silicon, enable MPS fallback first:

```shell
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

### Example: Cityscapes ###

```shell
conda activate AL-SEG
CONFIG=configs/_template_/deeplabv3_r18_cityscapes-344x688.py
python tools/train.py $CONFIG --seed 42
```

### Example: Pascal VOC 2012 ###

```shell
conda activate AL-SEG
CONFIG=configs/_template_/segformer_mit-b1_voc2012-480x480.py
python tools/train.py $CONFIG --seed 42
```

### Optional Arguments ###

```shell
python tools/train.py $CONFIG \
  --seed 42 \
  --experiment-name FullTrain \
  --work-dir ./logs \
  --options train_cfg.max_epochs=100
```

- `--work-dir`: base directory for logs/checkpoints
- `--experiment-name`: subdirectory name for this run
- `--options`: override config values from CLI
- `--use-single-thread`: set dataloader workers to 0 (useful for debugging or constrained systems)

By default, outputs are saved under:
`logs/<config_name>/<experiment_name>/`
If `--seed` is set, an extra `seed_<seed>/` subdirectory is added.

## Active Learning Experiments ##

Use `tools/experiment.py` to run the full active-learning loop
(query -> train -> repeat).

Use a DAL config (`configs/*_dal-*.py`) because it already defines
`experiment_cfg.init_samples`, `experiment_cfg.num_query`, and `experiment_cfg.num_cycles`.

If you are on Apple Silicon, enable MPS fallback first:

```shell
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

### 1) Run a Random baseline ###

```shell
conda activate AL-SEG
CONFIG=configs/deeplabv3_r18_cityscapes-344x688_dal-300-150-6.py
python tools/experiment.py $CONFIG --seed 42 --experiment-name Random
```

### 2) Run another strategy ###

```shell
conda activate AL-SEG
CONFIG=configs/deeplabv3_r18_cityscapes-344x688_dal-300-150-6.py
python tools/experiment.py $CONFIG \
  --seed 42 \
  --experiment-name Entropy \
  --options query_cfg.type=Entropy
```

### 3) Run CRC-AL strategy (ConformalRisk) ###

```shell
conda activate AL-SEG
CONFIG=configs/deeplabv3_r18_cityscapes-344x688_dal-300-150-6.py
python tools/experiment.py $CONFIG \
  --seed 42 \
  --experiment-name CRC-AL \
  --options query_cfg.type=ConformalRisk query_cfg.alpha=0.05 query_cfg.tau=0.5
```

For non-random strategies, if a matching Random baseline exists at
`logs/<config_name>/Random/seed_<seed>/`, the script automatically reuses
the initial `query_0` and `train_0` state for fair comparison.

### Optional Arguments ###

```shell
python tools/experiment.py $CONFIG \
  --seed 42 \
  --experiment-name Margin \
  --work-dir ./logs \
  --options query_cfg.type=Margin experiment_cfg.num_cycles=3 \
  --init-query-dir <PATH_TO_QUERY_0_DIR> \
  --init-train-dir <PATH_TO_TRAIN_0_DIR>
```

- `--work-dir`: base directory for logs/checkpoints
- `--experiment-name`: strategy/run name (used in output directory)
- `--options`: override config values from CLI
- `--init-query-dir` and `--init-train-dir`: manually warm-start from an existing cycle-0 state
- `--use-single-thread`: set dataloader workers to 0 (useful for debugging or constrained systems)

Results are written under:
`logs/<config_name>/<experiment_name>/<timestamp>/`
If `--seed` is set, results go to:
`logs/<config_name>/<experiment_name>/seed_<seed>/<timestamp>/`
Each cycle creates `query_<k>/` and `train_<k>/` subdirectories.

## Note

No coding agent has been used in this codebase.
