#!/bin/bash -l

# Job name
#SBATCH --job-name=cornet_z_cl
# Mail events (NONE, BEGIN, END, FAIL, ALL)
###############################################
########## example #SBATCH --mail-type=END,FAIL 
##############################################
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vayzenb@cmu.edu
 
# Submit job to cpu queue                
#SBATCH -p gpu

#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --exclude=mind-1-1,mind-1-3,mind-1-5,mind-1-26,mind-1-32 
# Job memory request
#SBATCH --mem=52gb

# Time limit days-hrs:min:sec
#SBATCH --time 3-00:00:00

# Standard output and error log
#SBATCH --output=slurm_out/cornet_z_cl.out

conda activate open_cl

python train_ipcl.py -a cornet_z -b 64 --gpu 0 /lab_data/behrmannlab/image_sets/imagenet_objects


