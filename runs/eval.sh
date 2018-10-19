#!/bin/bash


PATHDATASET='~/.kaggle/competitions/'
NAMEDATASET='tgs-salt-identification-challenge'
PROJECT='../netruns'
#PROJECTNAME='exp_tgs_unetresnet34_mcedice_adam_tgs-salt-identification-challenge_003'
PROJECTNAME='kaggle_tgs_unetresnet152_lovasz_adam_tgs-salt-identification-challenge_006'
PATHNAMEOUT='.'
FILENAME='result.txt'
PATHMODEL='models'
NAMEMODEL='model_best_000.pth.tar' #'model_best.pth.tar' #'chk000335.pth.tar'
MODEL=$PROJECT/$PROJECTNAME/$PATHMODEL/$NAMEMODEL  

python ../eval.py \
--project=$PROJECT \
--projectname=$PROJECTNAME \
--pathdataset=$PATHDATASET \
--namedataset=$NAMEDATASET \
--pathnameout=$PATHNAMEOUT \
--filename=$FILENAME \
--model=$MODEL \


