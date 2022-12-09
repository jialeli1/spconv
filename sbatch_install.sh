#!/usr/bin/env bash

#SBATCH --partition=default-short
#SBATCH --time=1:00:00 
#SBATCH -o ./slurm_out/job.%j.out
#SBATCH --job-name=spconv_install
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1

python setup.py bdist_wheel


# pip uninstall spconv-1.2.1 -y

cd /path/to/spconv/dist
pip install ./spconv-1.2.1-cp37-cp37m-linux_x86_64.whl --force-reinstall