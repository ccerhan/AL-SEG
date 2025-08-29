## Cityscapes Dataset

https://www.cityscapes-dataset.com/dataset-overview

### Download

```shell
sh download.sh <USERNAME> <PASSWORD>
```

### Prepare

https://github.com/open-mmlab/mmsegmentation/blob/main/tools/dataset_converters/cityscapes.py

```shell
DATASET_PATH=../cityscapes
python cityscapes.py $DATASET_PATH
```
