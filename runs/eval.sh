#!/bin/bash


PATHDATASET='~/.kaggle/competitions/'
NAMEDATASET='tgs-salt-identification-challenge'
PROJECT='../netruns'
PROJECTNAME='exp_tgs_unet_mcedice_adam_tgs-salt-identification-challenge_001'
PATHNAMEOUT='.'
FILENAME='result.txt'
PATHMODEL='models'
NAMEMODEL='chk000135.pth.tar' #'model_best.pth.tar'
MODEL=$PROJECT/$PROJECTNAME/$PATHMODEL/$NAMEMODEL  

python ../eval.py \
--project=$PROJECT \
--projectname=$PROJECTNAME \
--pathdataset=$PATHDATASET \
--namedataset=$NAMEDATASET \
--pathnameout=$PATHNAMEOUT \
--filename=$FILENAME \
--model=$MODEL \


