#!/bin/bash


# kaggle competitions submit -c tgs-salt-identification-challenge -f submission.csv -m "Message"
PATHDATASET='~/.kaggle/competitions/'
NAMEDATASET='tgs-salt-identification-challenge'
PROJECT='../netruns'
PROJECTNAME='exp_tgs_unetresnet34_mcedice_adam_tgs-salt-identification-challenge_004'
#PROJECTNAME='exp_tgs_unet11_mcedice_adam_tgs-salt-identification-challenge_001'
PATHNAMEOUT='.'
FILENAME='submission.csv'
PATHMODEL='models'
NAMEMODEL='model_best.pth.tar' 
MODEL=$PROJECT/$PROJECTNAME/$PATHMODEL/$NAMEMODEL  

python ../submission.py \
--project=$PROJECT \
--projectname=$PROJECTNAME \
--pathdataset=$PATHDATASET \
--namedataset=$NAMEDATASET \
--pathnameout=$PATHNAMEOUT \
--filename=$FILENAME \
--model=$MODEL \


