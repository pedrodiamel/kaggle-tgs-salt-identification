#!/bin/bash



PATHDATASET='~/.datasets/'
NAMEDATASET='tgs_salt'
PROJECT='../netruns'
PROJECTNAME='exp_tgs_unet_mcedice_adam_tgs_salt_001'
PATHNAMEOUT='.'
FILENAME='submission.csv'
PATHMODEL='models'
NAMEPROJECT='exp_tgs_unet_mcedice_adam_tgs_salt_001'
NAMEMODEL='model_best.pth.tar'
MODEL=$PROJECT/$NAMEPROJECT/$PATHMODEL/$NAMEMODEL  

python ../submission.py \
--project=$PROJECT \
--projectname=$PROJECTNAME \
--pathdataset=$PATHDATASET \
--namedataset=$NAMEDATASET \
--pathnameout=$PATHNAMEOUT \
--filename=$FILENAME \
--model=$MODEL \


