#!/bin/bash
#SBATCH -p gpu

#SBATCH -t 96:00:00
#SBATCH -J famke
#SBATCH -o log.%j.out
#SBATCH -e err.%j.out

cd /home/uu_imau_ocean/fkovacs/More_Time

#python3 Wind_Pretrain.py
python3 Wind_Train.py
