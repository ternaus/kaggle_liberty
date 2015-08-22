#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -V
#$ -N job_name
#$ -pe orte 1
#cd $SGE_O_WORKDIR
python RF_cv.py