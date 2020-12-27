#!/bin/bash
script_name1=`basename $0`
script_name=${script_name1:0:${#script_name1}-3}
cd ..

export CUDA_VISIBLE_DEVICES=4

MODEL=base_r101
DATANAME=cub
DATAPATH=/data06/v-shaomi/data/CUB_200_2011/CUB_200_2011/images/
SAVEPATH=${DATANAME}/output/${script_name}
nvidia-smi

STAGE1=1
STAGE2=1

if [ ${STAGE1} = 1 ]
then
  python main.py \
    --batch-size 128 \
    --aug v6 \
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
    --lr 0.001 \
    --aug v4 \
    --epochs 180 \
    --model-name ${MODEL} \
    --data-name ${DATANAME} \
    --save-path ${SAVEPATH} \
    --data ${DATAPATH} \
    --resume ${SAVEPATH}/fix.model
fi

