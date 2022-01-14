#!/bin/bash

device=$1

echo "Using device" $device

# for p in 0 0.2 0.4 0.6 0.8 1
for p in 0 0.2
do
python main.py --dataset cifar100 --model densenet40-40 --device $device\
    --out-name ../results/cifar100/cifar100_${p}_svm --loss svm --noise $p --no-visdom;
done