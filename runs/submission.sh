#!/bin/bash


# kaggle competitions submit -c tgs-salt-identification-challenge -f submission.csv -m "Message"
PATHDATASET='~/.kaggle/competitions/'
NAMEDATASET='tgs-salt-identification-challenge'
PROJECT='../netruns'
PROJECTNAME='exp_tgs_unetresnet_mcedice_adam_tgs-salt-identification-challenge_002'
#PROJECTNAME='exp_tgs_unet11_mcedice_adam_tgs-salt-identification-challenge_001'
PATHNAMEOUT='.'
FILENAME='submission.csv'
PATHMODEL='models'
NAMEMODEL='chk000210.pth.tar' 
MODEL=$PROJECT/$PROJECTNAME/$PATHMODEL/$NAMEMODEL  

python ../submission_ens.py \
--project=$PROJECT \
--projectname=$PROJECTNAME \
--pathdataset=$PATHDATASET \
--namedataset=$NAMEDATASET \
--pathnameout=$PATHNAMEOUT \
--filename=$FILENAME \
--model=$MODEL \


