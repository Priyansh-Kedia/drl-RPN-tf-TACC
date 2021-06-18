#!/bin/bash

#SBATCH -J drlrpn           # Job name
#SBATCH -o drlrpn.o%j       # Name of stdout output file
#SBATCH -e drlrpn.e%j       # Name of stderr error file
#SBATCH -p rtx          # Queue (partition) name
#SBATCH -N 10               # Total # of nodes (must be 1 for serial)
#SBATCH -n 1               # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 02:30:00        # Run time (hh:mm:ss)
#SBATCH --mail-type=all    # Send email at begin and end of job
#SBATCH -A CHE20013       # Project/Allocation name (req'd if you have more than 1)
#SBATCH --mail-user=kediapriyansh@gmail.com

module load cuda cudnn

source ~/anaconda3/etc/profile.d/conda.sh
conda activate env

conda install pip==10.0.0 -y
pyv="$(pip -V 2>&1)"
echo "$pyv"

pip install -r requirements.txt

cd lib
make clean
make 
cd ..
cd data/coco/PythonAPI
make
cd ..//..//..

# Launch serial code...
bash ./experiments/scripts/train_drl_rpn.sh 0 pascal_voc_0712 1 20000 0 2700  # Do not use ibrun or any other MPI launcher

conda deactivate
