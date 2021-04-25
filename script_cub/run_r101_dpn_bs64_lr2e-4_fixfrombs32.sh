#!/bin/bash
script_name1=`basename $0`
script_name=${script_name1:0:${#script_name1}-3}
cd ..

export CUDA_VISIBLE_DEVICES=0,1

MODEL=dpn
DATANAME=cub
BACKBONE=resnet101
DATAPATH=../../data/CUB_200_2011/CUB_200_2011/images/
SAVEPATH=output/${DATANAME}/${script_name}

STAGE1=0
STAGE2=1

if [ ${STAGE1} = 1 ]
then
  python main.py \
    --batch-size 512 \
    --lr 4e-4 \
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
    --batch-size 64 \
    --lr 2e-4 \
    --n-dec 3 \
    --epochs 90 \
    --backbone ${BACKBONE} \
    --model ${MODEL} \
    --data-name ${DATANAME} \
    --save-path ${SAVEPATH} \
    --data ${DATAPATH} \
    --resume ${SAVEPATH}/fix.model
fi


