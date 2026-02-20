#!/bin/bash -l
#$ -P proj_name
#$ -N singularity_exec
#$ -j y


#singularity pull docker://minepy/mictools


scc-singularity --nolocal exec mictools_latest.sif python3 mic_sig.py