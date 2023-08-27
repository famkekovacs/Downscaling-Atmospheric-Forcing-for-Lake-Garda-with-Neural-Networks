#!/bin/bash
#SBATCH -p gpu
#SBATCH -t 6:00:00
#SBATCH -J famke
#SBATCH -o log.%j.out
#SBATCH -e err.%j.out

cd /home/uu_imau_ocean/fkovacs/More_Time

python3 Temp_Test_Train.py
python3 Temp_Test_Test.py
