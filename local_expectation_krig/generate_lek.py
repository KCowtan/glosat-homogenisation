#!/usr/bin/env python
    
#------------------------------------------------------------------------------
# PROGRAM: generate_lek.py
#------------------------------------------------------------------------------
# Version 0.1
# 22 May, 2022
# Michael Taylor
# https://patternizer.github.io
# patternizer AT gmail DOT com
# michael DOT a DOT taylor AT uea DOT ac DOT uk
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# IMPORT PYTHON LIBRARIES
#------------------------------------------------------------------------------
import numpy as np
import os, glob
import stat
import subprocess

#------------------------------------------------------------------------------
# SLURM SHELL SCRIPT GENERATING FUNCTION
#------------------------------------------------------------------------------

def make_shell_command(cluster, file_in, file_out, ncycles):
         
    job_id = '#SBATCH --job-name=lek_cluster.{0}\n'.format(cluster)      
    job_file = 'run.{0}.lek_cluster.sh'.format(cluster)
#    job_command = 'python calc_homogenization_full.py -i={0} -o={1} -cycles={2}\n'.format(file_in, file_out, ncycle)
    job_command = 'python calc_homogenization_iter.py -i={0} -o={1} -cycles={2}\n'.format(file_in, file_out, ncycle)
    with open(job_file,'w') as fp:
        fp.write('#!/bin/bash\n')
        fp.write('#SBATCH --partition=short-serial\n')
        fp.write(job_id)
        fp.write('#SBATCH -o %j.out\n')
        fp.write('#SBATCH -e %j.err\n')
        fp.write('#SBATCH --time=12:00:00\n') 
        fp.write('module load jaspy\n')
        fp.write(job_command)

    # Make script executable

    os.chmod(job_file,stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)

    # Submit script to SLURM

    job = ['sbatch',job_file]
    subprocess.call(job)

if __name__ == "__main__":
    
    ncycle = 2
    cluster_dir = '/gws/nopw/j04/glosat/development/data/raw/UEA/mataylor/glosat-homogenisation/glosat-hca/OUT/40/cluster-pkl/*.pkl'
    cluster_pkl = sorted(glob.glob(cluster_dir))
    for cluster in range(len(cluster_pkl)):
        file_in = cluster_pkl[cluster]
        file_out = 'df_temp_expect_{0}.pkl'.format(str(cluster).zfill(2))     
        if not os.path.exists(file_out):
            print(file_in)
            make_shell_command(cluster, file_in, file_out, ncycle)


        
