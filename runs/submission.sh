#!/bin/bash


# kaggle competitions submit -c tgs-salt-identification-challenge -f submission.csv -m "Message"
PATHDATASET='~/.kaggle/competitions/'
NAMEDATASET='tgs-salt-identification-challenge'
PROJECT='../netruns'
PROJECTNAME='kaggle_tgs_unetresnet152_lovasz_adam_tgs-salt-identification-challenge_006'
PATHNAMEOUT='.'
FILENAME='submission.csv'
PATHMODEL='models'
NAMEMODEL='model_best_000.pth.tar' 
MODEL=$PROJECT/$PROJECTNAME/$PATHMODEL/$NAMEMODEL  

python ../submission.py \
--project=$PROJECT \
--projectname=$PROJECTNAME \
--pathdataset=$PATHDATASET \
--namedataset=$NAMEDATASET \
--pathnameout=$PATHNAMEOUT \
--filename=$FILENAME \
--model=$MODEL \


