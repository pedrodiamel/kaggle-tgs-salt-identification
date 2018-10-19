#!/bin/bash


# kaggle competitions submit -c tgs-salt-identification-challenge -f submission.csv -m "Message"
PATHDATASET='~/.kaggle/competitions/'
NAMEDATASET='tgs-salt-identification-challenge'
PROJECT='../netruns'
PROJECTNAME='kaggle_tgs_unetresnet152_mcedice_rmsprop_tgs-salt-identification-challenge_007'
PATHNAMEOUT='.'
FILENAME='submission.csv'
PATHMODEL='models'
NAMEMODEL='model_best.pth.tar' 
MODEL=$PROJECT/$PROJECTNAME/$PATHMODEL/$NAMEMODEL  

python ../submission_ens.py \
--project=$PROJECT \
--projectname=$PROJECTNAME \
--pathdataset=$PATHDATASET \
--namedataset=$NAMEDATASET \
--pathnameout=$PATHNAMEOUT \
--filename=$FILENAME \
--model=$MODEL \


