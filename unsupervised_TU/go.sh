#!/bin/bash -ex

for seed in 0 1 2 3 4 
do
  CUDA_VISIBLE_DEVICES=1 python gsimclr.py --DS MUTAG --lr 0.01 --local --num-gc-layers 3 --aug dnodes --seed $seed

done

