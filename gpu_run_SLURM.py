# -*- coding: utf-8 -*-

"""
Given a bunch of commands to run, check the available GPUs and run them on the GPUs in separate tmux sessions.
Usage: Just modify the settings in the Config class and then run python3 gpu_run.py
"""

import GPUtil
import subprocess
import sys
import time

class Config:
    """
    Global class that houses all configurations
    """
    
    # Shared args to put onto all of the JOBS
    SHARED_ARGS = " "

    SLURM_HEADER = "conda activate rl-pad-3; srun --pty -p gpu_jsteinhardt -w shadowfax -c 11 --gres=gpu:1"

    # Specifies tasks to run. It maps tmux session name to the command to run in that session.
    JOBS = {        
        "walker_walk_noss_noise2net_seed0" : "python3 src/train.py \
                --domain_name walker \
                --task_name walk \
                --action_repeat 4 \
                --mode train \
                --num_shared_layers 8 \
                --seed 0 \
                --work_dir logs/walker_walk/no_ss/noise2net_seed0 \
                --save_model \
                --neural_aug_type=noise2net",
        
        "walker_walk_noss_noise2net_seed1" : "python3 src/train.py \
                --domain_name walker \
                --task_name walk \
                --action_repeat 4 \
                --mode train \
                --num_shared_layers 8 \
                --seed 1 \
                --work_dir logs/walker_walk/no_ss/noise2net_seed1 \
                --save_model \
                --neural_aug_type=noise2net",
        
        "walker_walk_noss_noise2net_seed2" : "python3 src/train.py \
                --domain_name walker \
                --task_name walk \
                --action_repeat 4 \
                --mode train \
                --num_shared_layers 8 \
                --seed 2 \
                --work_dir logs/walker_walk/no_ss/noise2net_seed2 \
                --save_model \
                --neural_aug_type=noise2net",
        
        "walker_walk_noss_noise2net_seed3" : "python3 src/train.py \
                --domain_name walker \
                --task_name walk \
                --action_repeat 4 \
                --mode train \
                --num_shared_layers 8 \
                --seed 3 \
                --work_dir logs/walker_walk/no_ss/noise2net_seed3 \
                --save_model \
                --neural_aug_type=noise2net",
    }

    # Time to wait between putting jobs on GPUs (in seconds). This is useful because it might take time 
    # for a process to actually load the network onto the GPU, so we wait until that is done before 
    # selecting the GPU for the next process.
    SLEEP_TIME = 1


# Stick the shared args onto each JOB
for key, value in Config.JOBS.items():
    new_value = value + " " + Config.SHARED_ARGS
    Config.JOBS[key] = new_value


for index, (tmux_session_name, command) in enumerate(Config.JOBS.items()):
    # Get the best available GPU
    print("Running \"{0}\" : \"{1}\" with SLURM".format(tmux_session_name, command))

    result = subprocess.run("tmux new-session -d -s {0}".format(tmux_session_name), shell=True)        
    if result.returncode != 0:
        print("Failed to create new tmux session called {0}".format(tmux_session_name))
        sys.exit(result.returncode)

    result = subprocess.run("tmux send-keys '{0} {1}' C-m".format(
        Config.SLURM_HEADER, command
    ), shell=True)
    if result.returncode != 0:
        print("Failed to run {0} in tmux session".format(command, tmux_session_name))
        sys.exit(result.returncode)

    print("---------------------------------------------------------------")

    if index < len(Config.JOBS) - 1:
        time.sleep(Config.SLEEP_TIME)
