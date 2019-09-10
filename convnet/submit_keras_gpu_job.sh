#!/bin/bash
#SBATCH --account=
#SBATCH --mail-user=
#SBATCH --mail-type=ALL

#SBATCH --nodes=1
#SBATCH --gres=gpu:2            # request GPU "generic resource"
#SBATCH --ntasks-per-node=32
#SBATCH --mem=0                 # memory per node

#SBATCH --time=0-03:00          # time (DD-HH:MM)
#SBATCH --job-name=sensor_2D_d64_b25
#SBATCH --output=%x-%j.out      # %N for node name, %j for jobID

module load cuda cudnn python/3.5.2
source ~/tensorflow/bin/activate
time python convnet_sensor.py 'sensor_2D_d64_b25'
