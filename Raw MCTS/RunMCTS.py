
import numpy as np
from multiprocessing import Process, shared_memory
import tqdm
import time
import argparse

import os
import sys
sys.path.append("../")

from Raw_MCTS import *
from CandyCrushGym import *


NUM_PROCS = 32

NUM_ITERATIONS = 10

def process_runMCTS(process_id, seed, candy_buff_height, num_mcts_iterations, avg_rewards_shm, sum_rewards_shm, times_shm):
    
    env = CandyCrushGym(seed=seed, candy_buff_height=candy_buff_height)
    env.reset()

    avg_rewards_mem = np.ndarray(shape=(NUM_PROCS, ), dtype=np.float32, buffer=avg_rewards_shm.buf)
    sum_rewards_mem = np.ndarray(shape=(NUM_PROCS, ), dtype=np.float32, buffer=sum_rewards_shm.buf)
    times_mem = np.ndarray(shape=(NUM_PROCS, ), dtype=np.float32, buffer=times_shm.buf)

    iterator = range(NUM_ITERATIONS)

    if process_id == -1:
        process_id = NUM_PROCS - 1
        iterator = tqdm.tqdm(iterator, position=0, leave=True)


    rewards = np.zeros(shape=(NUM_ITERATIONS,), dtype=np.float32)
    times = np.zeros(shape=(NUM_ITERATIONS,), dtype=np.float32)
    for i in iterator:
        mcts = Raw_MCTS(env)

        time_start = time.time()
        action, _, _ = mcts.run(num_mcts_iterations)
        time_stop = time.time()

        _, reward, _, _  = env.step(action)

        rewards[i] = reward
        times[i] = time_stop - time_start

    avg_rewards_mem[process_id] = np.average(rewards)
    sum_rewards_mem[process_id] = np.sum(rewards)
    times_mem[process_id] = np.average(times)

def check(num: str, name: str):

    try:
        num = int(num)
    except:
        raise argparse.ArgumentTypeError(f"The {name} must be an integer.")
    

    if num <= 0:
        raise argparse.ArgumentTypeError(f"The {name} must be positive and greater than zero")
    
    return num

def main():
    
    #
    # Set up ArgumentParser
    #

    parser = argparse.ArgumentParser(description="Candy Crush")
    
    parser.add_argument("--start", help="Set the start number of MCTS steps", type=lambda start: check(start, name="step number"), required=True)
    parser.add_argument("--stop", help="Set the end number of MCTS steps", type=lambda start: check(start, name="step number"), required=True)
    parser.add_argument("--step", help="Set the step number going from start to end number of MCTS steps", type=lambda start: check(start, name="step number"), required=True)
    parser.add_argument("--buff", help="Candy buff height", type=lambda start: check(start, name="Candy buff height"), required=True)

    args = parser.parse_args()

    start = args.start 
    stop = args.stop 
    step = args.step
    candy_buff_height = args.buff

    if stop <= start:
        raise argparse.ArgumentTypeError("Start must be less than stop")

    #
    # Create dirs where log files are stored
    #

    root = f"./logs/candy_buff_height_{candy_buff_height}"

    if not os.path.exists(root):
        os.mkdir(root)

    root_performance_logs = root + "/performance"

    if not os.path.exists(root_performance_logs):
        os.mkdir(root_performance_logs)

    root_time_logs = root + "/time"

    if not os.path.exists(root_time_logs):
        os.mkdir(root_time_logs)

    

    #
    # Start processes which execute MCTS
    #
    
    num_mcts_iterations_list = range(start, stop, step)
    

    for num_mcts_iterations in tqdm.tqdm(num_mcts_iterations_list, position=0, leave=True):

        print(f"Num_mcts_iteration: {num_mcts_iterations}")
        
        avg_rewards_mem = np.zeros(shape=(NUM_PROCS,), dtype=np.float32)
        sum_rewards_mem = np.zeros(shape=(NUM_PROCS,), dtype=np.float32)
        times_mem = np.zeros(shape=(NUM_PROCS,), dtype=np.float32)

        avg_rewards_shm = shared_memory.SharedMemory(create=True, size=avg_rewards_mem.nbytes)
        sum_rewards_shm = shared_memory.SharedMemory(create=True, size=sum_rewards_mem.nbytes)
        times_shm = shared_memory.SharedMemory(create=True, size=times_mem.nbytes)

        procs = []
        for process_id in range(NUM_PROCS - 1):
            seed = np.random.randint(0, 999999999)
            proc = Process(target=process_runMCTS, 
                           args=(process_id, seed, candy_buff_height, num_mcts_iterations, avg_rewards_shm, sum_rewards_shm, times_shm,))
            procs.append(proc)
            proc.start()

        seed = np.random.randint(0, 999999999)
        process_runMCTS(-1, seed, candy_buff_height, num_mcts_iterations, avg_rewards_shm, sum_rewards_shm, times_shm)

        # complete the processes
        for proc in procs:
            proc.join()
    
        
        avg_rewards_mem = np.ndarray(shape=(NUM_PROCS, ), dtype=np.float32, buffer=avg_rewards_shm.buf)
        sum_rewards_mem = np.ndarray(shape=(NUM_PROCS, ), dtype=np.float32, buffer=sum_rewards_shm.buf)
        times_mem = np.ndarray(shape=(NUM_PROCS, ), dtype=np.float32, buffer=times_shm.buf)

        np.save(f"{root_performance_logs}/avg_rewards_{num_mcts_iterations}.npy", avg_rewards_mem)
        np.save(f"{root_performance_logs}/sum_rewards_{num_mcts_iterations}.npy", sum_rewards_mem)
        np.save(f"{root_time_logs}/{num_mcts_iterations}.npy", sum_rewards_mem)

        avg_rewards_shm.close()
        sum_rewards_shm.close()
        times_shm.close()

        avg_rewards_shm.unlink()
        sum_rewards_shm.unlink()
        times_shm.unlink()
    



if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")