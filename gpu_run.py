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
    SHARED_ARGS = ""

    HEADER = "conda activate rl-pad-3; "

    # Specifies tasks to run. It maps tmux session name to the command to run in that session.
    JOBS = {
        
        #####################################################################################
        #### SHADOWFAX
        #####################################################################################

        "cartpole_balance_noss_noise2net_seed0_replayBuffer500k" : "python3 src/train.py \
                --domain_name cartpole \
                --task_name balance \
                --action_repeat 4 \
                --mode train \
                --num_shared_layers 8 \
                --seed 0 \
                --work_dir logs/cartpole_balance/no_ss/noise2net_seed0_replayBuffer500k \
                --save_model \
                --neural_aug_type=noise2net",

        #####################################################################################
        #### SMAUG
        #####################################################################################

        "finger_spin_noss_noise2net_seed0_replayBuffer500k" : "python3 src/train.py \
                --domain_name finger \
                --task_name spin \
                --action_repeat 4 \
                --mode train \
                --num_shared_layers 8 \
                --seed 0 \
                --work_dir logs/finger_spin/no_ss/noise2net_seed0_replayBuffer500k \
                --save_model \
                --neural_aug_type=noise2net",

        "finger_turn_easy_noss_noise2net_seed0_replayBuffer500k" : "python3 src/train.py \
                --domain_name finger \
                --task_name turn_easy \
                --action_repeat 4 \
                --mode train \
                --num_shared_layers 8 \
                --seed 0 \
                --work_dir logs/finger_turn_easy/no_ss/noise2net_seed0_replayBuffer500k \
                --save_model \
                --neural_aug_type=noise2net",

        "cheetah_run_noss_noise2net_seed0_replayBuffer500k" : "python3 src/train.py \
                --domain_name cheetah \
                --task_name run \
                --action_repeat 4 \
                --mode train \
                --num_shared_layers 8 \
                --seed 0 \
                --work_dir logs/cheetah_run/no_ss/noise2net_seed0_replayBuffer500k \
                --save_model \
                --neural_aug_type=noise2net",
    }

    # Time to wait between putting jobs on GPUs (in seconds). This is useful because it might take time 
    # for a process to actually load the network onto the GPU, so we wait until that is done before 
    # selecting the GPU for the next process.
    SLEEP_TIME = 10

    # Minimum memory required on a GPU to consider putting a job on it (MiB).
    MIN_MEMORY_REQUIRED = 4000


# Stick the shared args onto each JOB
for key, value in Config.JOBS.items():
    new_value = value + " " + Config.SHARED_ARGS
    Config.JOBS[key] = new_value

def select_gpu(GPUs):
    """
    Select the next best available GPU to run on. If nothing exists, return None
    """
    GPUs = list(filter(lambda gpu: gpu.memoryFree > Config.MIN_MEMORY_REQUIRED, GPUs))
    if len(GPUs) == 0:
        return None
    GPUs = sorted(GPUs, key=lambda gpu: gpu.memoryFree)
    return GPUs[-1]

for index, (tmux_session_name, command) in enumerate(Config.JOBS.items()):
    # Get the best available GPU
    print("Finding GPU for command \"{0}\"".format(command))
    curr_gpu = select_gpu(GPUtil.getGPUs())

    if curr_gpu == None:
        print("No available GPUs found. Exiting.")
        sys.exit(1)

    print("SUCCESS! Found GPU id = {0} which has {1} MiB free memory".format(curr_gpu.id, curr_gpu.memoryFree))

    result = subprocess.run("tmux new-session -d -s {0}".format(tmux_session_name), shell=True)        
    if result.returncode != 0:
        print("Failed to create new tmux session called {0}".format(tmux_session_name))
        sys.exit(result.returncode)

    result = subprocess.run("tmux send-keys '{2} CUDA_VISIBLE_DEVICES={0} {1}' C-m".format(
        curr_gpu.id, command, Config.HEADER
    ), shell=True)
    if result.returncode != 0:
        print("Failed to run {0} in tmux session".format(command, tmux_session_name))
        sys.exit(result.returncode)

    print("---------------------------------------------------------------")

    if index < len(Config.JOBS) - 1:
        time.sleep(Config.SLEEP_TIME)
