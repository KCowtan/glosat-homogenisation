#!/usr/bin/env python
    
#------------------------------------------------------------------------------
# PROGRAM: krig_all_stations.py
#------------------------------------------------------------------------------
# Version 0.1
# 30 July, 2021
# Michael Taylor
# https://patternizer.github.io
# patternizer AT gmail DOT com
# michael DOT a DOT taylor AT uea DOT ac DOT uk
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# IMPORT PYTHON LIBRARIES
#------------------------------------------------------------------------------
import numpy as np
import os
import stat
import subprocess

#------------------------------------------------------------------------------
# SLURM SHELL SCRIPT GENERATING FUNCTION
#------------------------------------------------------------------------------

def make_shell_command(stationcode):
         
    job_id = '#SBATCH --job-name=krig.{0:02d}\n'.format(stationcode)      
    job_krig = 'python calc_expectations.py -filter={0:02d}'.format(stationcode)
    job_file = 'run.{0:02d}.sh'.format(stationcode)
    with open(job_file,'w') as fp:
        fp.write('#!/bin/bash\n')
        fp.write('#SBATCH --partition=short-serial\n')
        fp.write(job_id)
        fp.write('#SBATCH -o %j.out\n')
        fp.write('#SBATCH -e %j.err\n')
        fp.write('#SBATCH --time=05:00\n')
        fp.write('module load jaspy\n')
        fp.write(job_krig)

    # Make script executable

    os.chmod(job_file,stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)

    # Submit script to SLURM

    job = ['sbatch',job_file]
    subprocess.call(job)

if __name__ == "__main__":
    
   for stationcode in range(100):         
        file_out = 'df_temp_expect' + '_' + str(stationcode).zfill(2) + '.pkl'
        
        if not os.path.exists(file_out):
            make_shell_command(stationcode)


