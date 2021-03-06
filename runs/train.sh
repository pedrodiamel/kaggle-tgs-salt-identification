#!/bin/bash


# parameters
DATA='~/.kaggle/competitions/tgs-salt-identification-challenge/'
NAMEDATASET='tgs-salt-identification-challenge'
PROJECT='../netruns'
EPOCHS=120
BATCHSIZE=30
LEARNING_RATE=0.0001
MOMENTUM=0.5
PRINT_FREQ=75
WORKERS=15
RESUME='model_best.pth.tar'
GPU=0
ARCH='unetresnet'
LOSS='mcedice'
OPT='adam'
SCHEDULER='fixed'
IMAGESIZE=101
SNAPSHOT=5
EXP_NAME='exp_tgs_'$ARCH'_152_'$LOSS'_'$OPT'_'$NAMEDATASET'_001'


#rm -rf $PROJECT/$EXP_NAME/$EXP_NAME.log
#rm -rf $PROJECT/$EXP_NAME/
#mkdir $PROJECT    
#mkdir $PROJECT/$EXP_NAME  


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
--parallel \
--finetuning \
2>&1 | tee -a $PROJECT/$EXP_NAME/$EXP_NAME.log \

