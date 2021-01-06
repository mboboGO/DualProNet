#!/bin/bash
script_name1=`basename $0`
script_name=${script_name1:0:${#script_name1}-3}
cd ..

export CUDA_VISIBLE_DEVICES=5

MODEL=pro_net
DATANAME=cub
BACKBONE=resnet101
DATAPATH=../../data/CUB_200_2011/CUB_200_2011/images/
SAVEPATH=${DATANAME}/output/${script_name}

STAGE1=1
STAGE2=0

if [ ${STAGE1} = 1 ]
then
  python main.py \
    --batch-size 128 \
    --lr_ada 1e-3 \
    --lr_sgd 1e-1 \
    --n-enc 0 \
    --n-dec 0 \
    --epochs 90 \
    --backbone ${BACKBONE} \
    --model-name ${MODEL} \
    --data-name ${DATANAME} \
    --save-path ${SAVEPATH} \
    --data ${DATAPATH} \
    --is_fix 
fi

if [ ${STAGE2} = 1 ]
then
  python main.py \
    --batch-size 12 \
    --lr_ada 1e-3 \
    --n-enc 0 \
    --n-dec 0 \
    --backbone ${BACKBONE} \
    --model-name ${MODEL} \
    --data-name ${DATANAME} \
    --save-path ${SAVEPATH} \
    --data ${DATAPATH} \
    --resume ${SAVEPATH}/fix.model
fi


