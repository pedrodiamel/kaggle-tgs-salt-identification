#!/bin/bash


PATHDATASET='~/.kaggle/competitions/'
NAMEDATASET='tgs-salt-identification-challenge'
PROJECT='../netruns'
#PROJECTNAME='exp_tgs_unetresnet_152_mcedice_adam_tgs-salt-identification-challenge_001'
PROJECTNAME='exp_tgs_unetpreactresnet34_mcedice_adam_tgs-salt-identification-challenge_002'
PATHNAMEOUT='.'
FILENAME='result.txt'
PATHMODEL='models'
NAMEMODEL='chk000270.pth.tar' #'model_best.pth.tar' #'chk000510.pth.tar'
MODEL=$PROJECT/$PROJECTNAME/$PATHMODEL/$NAMEMODEL  

python ../eval.py \
--project=$PROJECT \
--projectname=$PROJECTNAME \
--pathdataset=$PATHDATASET \
--namedataset=$NAMEDATASET \
--pathnameout=$PATHNAMEOUT \
--filename=$FILENAME \
--model=$MODEL \


