#!/bin/bash

###

#SBATCH --qos=debug

###

#SBATCH --cpus-per-task=40

#SBATCH --gres gpu:1

###SBATCH --time=24:00:00

###

#SBATCH --job-name="p3_main"

#SBATCH --chdir=.

#SBATCH --output=../data/03_logs/p3_main_%j.out

#SBATCH --error=../data/03_logs/p3_main_%j.err

###

module purgue
module load  impi  intel  hdf5  mkl  python/3.12.1-gcc

time python main.py
