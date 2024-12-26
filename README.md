# AL-SEG: Active Learning for Semantic Segmentation #

**AL-SEG** is an open-source active learning tool for semantic segmentation based on PyTorch. It is built on top of [mmsegmentation](https://github.com/open-mmlab/mmsegmentation), which provides extensive support for various datasets, pre-trained backbones, and state-of-the-art segmentation methods right out of the box.

This repository contains the **official implementation** of the paper:
**Conformal Risk Controlled Active Learning (CRC-AL)**, which has been recently submitted. The source code of this project will be publicly available once the paper is accepted.

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
