#!/bin/bash


# parameters
DATA='~/.datasets/tgs_salt/'
NAMEDATASET='tgs_salt'
PROJECT='../netruns'
EPOCHS=60
BATCHSIZE=1
LEARNING_RATE=0.0001
MOMENTUM=0.5
PRINT_FREQ=10
WORKERS=1
RESUME='checkpointxx.pth.tar'
GPU=0
ARCH='unet'
LOSS='mcedice'
OPT='adam'
SCHEDULER='fixed'
IMAGESIZE=101
SNAPSHOT=10
EXP_NAME='exp_tgs_'$ARCH'_'$LOSS'_'$OPT'_'$NAMEDATASET'_001'


rm -rf $PROJECT/$EXP_NAME/$EXP_NAME.log
rm -rf $PROJECT/$EXP_NAME/
mkdir $PROJECT    
mkdir $PROJECT/$EXP_NAME  


python ../train.py \
$DATA \
--project=$PROJECT \
--name=$EXP_NAME \
--epochs=$EPOCHS \
--batch-size=$BATCHSIZE \
--learning-rate=$LEARNING_RATE \
--momentum=$MOMENTUM \
--image-size=$IMAGESIZE \
--print-freq=$PRINT_FREQ \
--snapshot=$SNAPSHOT \
--workers=$WORKERS \
--resume=$RESUME \
--gpu=$GPU \
--loss=$LOSS \
--opt=$OPT \
--scheduler=$SCHEDULER \
--arch=$ARCH \
--finetuning \
2>&1 | tee -a $PROJECT/$EXP_NAME/$EXP_NAME.log \

#--parallel \