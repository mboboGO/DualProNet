#!/bin/bash
script_name1=`basename $0`
script_name=${script_name1:0:${#script_name1}-3}
cd ..

export CUDA_VISIBLE_DEVICES=2,4

MODEL=pro_net
DATANAME=cub
BACKBONE=resnet101
DATAPATH=../../data/CUB_200_2011/CUB_200_2011/images/
SAVEPATH=output/${DATANAME}/${script_name}

STAGE1=1
STAGE2=1

if [ ${STAGE1} = 1 ]
then
  python main.py \
    --batch-size 128 \
    --lr 1e-4 \
    --n-dec 3 \
    --epochs 90 \
    --backbone ${BACKBONE} \
    --model ${MODEL} \
    --data-name ${DATANAME} \
    --save-path ${SAVEPATH} \
    --data ${DATAPATH} \
    --is_fix 
fi

if [ ${STAGE2} = 1 ]
then
  python main.py \
    --batch-size 16 \
    --lr 1e-4 \
    --n-dec 3 \
    --epochs 90 \
    --backbone ${BACKBONE} \
    --model ${MODEL} \
    --data-name ${DATANAME} \
    --save-path ${SAVEPATH} \
    --data ${DATAPATH} \
    --resume ${SAVEPATH}/fix.model
fi

