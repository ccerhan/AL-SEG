#!/bin/bash

echo "Downloading Pascal VOC 2012 Dataset..."
curl -L -o ./pascal-voc-2012-dataset.zip https://www.kaggle.com/api/v1/datasets/download/gopalbhattrai/pascal-voc-2012-dataset

echo "Downloading Pascal VOC 2012 + Augmentation Dataset..."
wget https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz

echo "Extracting Pascal VOC 2012 Dataset..."
unzip -q pascal-voc-2012-dataset.zip
mv  VOC2012_train_val/VOC2012_train_val/* .
rm -rf VOC2012_train_val VOC2012_test

echo "Extracting Pascal VOC 2012 + Augmentation Dataset..."
tar -xf benchmark.tgz

echo "Done"
