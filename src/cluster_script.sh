#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -V
#$ -N job_name
#$ -pe orte 10
cd $SGE_O_WORKDIR
orterun -n 10 python RF_cv.py