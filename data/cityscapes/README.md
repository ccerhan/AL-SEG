## Cityscapes Dataset

https://www.cityscapes-dataset.com/dataset-overview

### Download

```shell
sh download.sh <USERNAME> <PASSWORD>
```

### Prepare

The converter script is included in this repo: `data/cityscapes/cityscapes.py`.

```shell
DATASET_PATH=../cityscapes
python cityscapes.py $DATASET_PATH
```
