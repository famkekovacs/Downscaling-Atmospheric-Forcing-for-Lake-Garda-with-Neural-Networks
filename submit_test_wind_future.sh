#!/bin/bash
#SBATCH -p gpu
#SBATCH -t 6:00:00
#SBATCH -J famke
#SBATCH -o log.%j.out
#SBATCH -e err.%j.out

cd /home/uu_imau_ocean/fkovacs/More_Time

python3 Wind_Test_Future.py
