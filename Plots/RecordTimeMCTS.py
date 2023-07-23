import gym
import time
import numpy as np
from multiprocessing import Process, shared_memory
import tqdm

from MCTS import *

# openai gym causes a warning - disable it
from warnings import filterwarnings
filterwarnings(action='ignore', category=DeprecationWarning, message='`np.bool8` is a deprecated alias')


root_logs = "./logs/time"

NUM_PROCS = 25
ARRAY_SIZE = 500

NUM_STEPS = 100

def process_runMCTS(process_id, shm, num_mcts_iterations):

    env = gym.make("CartPole-v1")
    env.reset()
    
    durations = np.zeros(shape=(NUM_STEPS))
    for i in range(NUM_STEPS):

        mcts = MCTS(env)    

        time_start = time.time()

        action = mcts.run(num_mcts_iterations)

        time_stop = time.time()

        durations[i] = time_stop - time_start

  

    b = np.ndarray(shape=(ARRAY_SIZE,NUM_STEPS), dtype=np.float64, buffer=shm.buf)

    b[process_id, :] = durations

    shm.close()

def main():

    x = np.zeros(shape=(ARRAY_SIZE,NUM_STEPS), dtype=np.float64)
    shm = shared_memory.SharedMemory(create=True, size=x.nbytes)
    

    num_mcts_iterations = 2
    num_loops = int(ARRAY_SIZE / NUM_PROCS)
    for i in tqdm.tqdm(range(num_loops), position=0, leave=True):
        
        procs = []
        for process_id in range(i*NUM_PROCS,i*NUM_PROCS + NUM_PROCS):
            
            proc = Process(target=process_runMCTS, args=(process_id, shm, num_mcts_iterations,))
            procs.append(proc)
            proc.start()

            num_mcts_iterations += 1
            

        # complete the processes
        for proc in procs:
            proc.join()

    b = np.ndarray(shape=(ARRAY_SIZE,NUM_STEPS), dtype=np.float64, buffer=shm.buf)

    print(b)

    np.save(f"{root_logs}/log.npy", b)

    shm.close()
    shm.unlink()

 


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")