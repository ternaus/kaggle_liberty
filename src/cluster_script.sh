#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -pe orte 1
#cd $SGE_O_WORKDIR
python RF_cv.py