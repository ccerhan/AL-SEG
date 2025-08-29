## Pascal VOC 2012 Dataset

http://host.robots.ox.ac.uk/pascal/VOC/voc2012

### Download

```shell
sh download.sh
```

### Augmentation

https://github.com/open-mmlab/mmsegmentation/blob/main/tools/dataset_converters/voc_aug.py

```shell
DEVKIT_PATH=../../VOCdevkit
AUG_PATH=benchmark_RELEASE
python voc_aug.py $DEVKIT_PATH $AUG_PATH
python voc_merge.py $DEVKIT_PATH
```
