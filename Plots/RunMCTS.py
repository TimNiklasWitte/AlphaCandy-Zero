
import numpy as np
from multiprocessing import Process, shared_memory
import tqdm
import time

import sys
sys.path.append("../")

from MCTS import *
from CandyCrushGym import *


root_performance_logs = "./logs/performance"
root_time_logs = "./logs/time"

NUM_PROCS = 32

NUM_ITERATIONS = 10

def process_runMCTS(process_id, num_mcts_iterations, avg_rewards_shm, sum_rewards_shm, times_shm):
    
    env = CandyCrushGym()
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
        mcts = MCTS(env)

        time_start = time.time()
        action = mcts.run(num_mcts_iterations)
        time_stop = time.time()

        _, reward, _, _  = env.step(action)

        rewards[i] = reward
        times[i] = time_stop - time_start

    avg_rewards_mem[process_id] = np.average(rewards)
    sum_rewards_mem[process_id] = np.sum(rewards)
    times_mem[process_id] = np.average(times)


def main():
    
 
    num_mcts_iterations_list = range(50, 500, 50)
    

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

            proc = Process(target=process_runMCTS, 
                           args=(process_id, num_mcts_iterations, avg_rewards_shm, sum_rewards_shm, times_shm,))
            procs.append(proc)
            proc.start()

        
        process_runMCTS(-1, num_mcts_iterations, avg_rewards_shm, sum_rewards_shm, times_shm)

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