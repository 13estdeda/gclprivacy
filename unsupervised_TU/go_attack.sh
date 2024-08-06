#!/bin/bash -ex

for seed in 0
do
  CUDA_VISIBLE_DEVICES=1 python gsimclr_attack.py --DS MUTAG --lr 0.01 --local --num-gc-layers 3 --aug dnodes --seed $seed

done

