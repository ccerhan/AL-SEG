#!/bin/bash

echo "Downloading Pascal VOC 2012 Dataset..."
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar

echo "Downloading Pascal VOC 2012 + Augmentation Dataset..."
wget https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz

echo "Extracting Pascal VOC 2012 Dataset..."
tar -xf VOCtrainval_11-May-2012.tar

echo "Extracting Pascal VOC 2012 + Augmentation Dataset..."
tar -xf benchmark.tgz

mv VOCdevkit/VOC2012/* .
rm -rf VOCdevkit

echo "Done"
