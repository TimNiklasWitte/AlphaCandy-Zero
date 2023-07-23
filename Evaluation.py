import gym
import numpy as np
import tqdm

from CandyCrushGym import *
from MCTS import *
from Config import *

import tensorflow as tf
from multiprocessing import Process, shared_memory, Semaphore



def process_evaluate(process_id, avg_rewards_shm, sum_rewards_shm, request_state_img_shm, response_policy_shm, wait_for_response_sema, use_policy_network_only):
    

    env = CandyCrushGym(process_id)
    state = env.reset()

    avg_rewards_mem = np.ndarray(shape=(EVAL_NUM_PROCS,), dtype=np.float32, buffer=avg_rewards_shm.buf)
    sum_rewards_mem = np.ndarray(shape=(EVAL_NUM_PROCS,), dtype=np.float32, buffer=sum_rewards_shm.buf)

    stateToImageConverter = StateToImageConverter(field_size=FIELD_SIZE, candy_buff_height=CANDY_BUFF_HEIGHT, image_size=CANDY_IMG_SIZE)


    iterator = range(EVAL_NUM_STEPS_PER_PROC)
    if process_id == EVAL_NUM_PROCS - 1:
        print("<INFO> Evaluation")
        iterator = tqdm.tqdm(iterator,position=0, leave=True)

    rewards = np.zeros(shape=(EVAL_NUM_STEPS_PER_PROC,), dtype=np.float32)

    for i in iterator:

        if use_policy_network_only:
        
            request_state_img = np.ndarray(shape=STATE_IMG_SHAPE, dtype=STATE_IMG_DTYPE, buffer=request_state_img_shm.buf)
            state_img = stateToImageConverter(state)
            request_state_img[:] = state_img

            wait_for_response_sema.acquire()

            response_policy = np.ndarray(shape=(NUM_ACTIONS,), dtype=POLICY_DTYPE, buffer=response_policy_shm.buf)
                
            num_iterations = 0
            terminate_rollout = False

            # convert + normalize -> prevent error by np.random.choice 
            response_policy = response_policy.astype('float64')
            response_policy /= response_policy.sum()  

            while True:
                
                action = np.random.choice(np.arange(NUM_ACTIONS), p=response_policy)
            
                if isValidAction(action):
                    state, reward, _, _ = env.step(action)
                    if reward != 0:
                        break 

                    if num_iterations > 1000:
                        terminate_rollout = True 
                        break 
                    
                    num_iterations += 1

                if terminate_rollout:
                    break 

        else:
            mcts = MCTS(env, state, request_state_img_shm, response_policy_shm, wait_for_response_sema, stateToImageConverter)
            action, _ = mcts.run(NUM_MCTS_STEPS)

         
            state, reward, done, _, _ = env.step(action)
    
            if done:
                state, _ = env.reset()
                break
        
        rewards[i] = reward 

    avg_rewards_mem[process_id] = np.average(rewards)
    sum_rewards_mem[process_id] = np.sum(rewards)

  

def evaluate(policyValueNetwork, use_policy_network_only):


    request_state_img_shm_list = []
    response_policy_shm_list = []
    wait_for_response_sema_list = []
    
    # prototypical state and action memory
    state_img = np.zeros(shape=STATE_IMG_SHAPE, dtype=STATE_IMG_DTYPE)
    policy = np.zeros(shape=(NUM_ACTIONS,), dtype=POLICY_DTYPE)

    avg_rewards_mem = np.zeros(shape=(EVAL_NUM_PROCS, ), dtype=np.float32)
    avg_rewards_shm = shared_memory.SharedMemory(create=True, size=avg_rewards_mem.nbytes)

    sum_rewards_mem = np.zeros(shape=(EVAL_NUM_PROCS, ), dtype=np.float32)
    sum_rewards_shm = shared_memory.SharedMemory(create=True, size=sum_rewards_mem.nbytes)


    proc_list = [] 
    for process_id in range(EVAL_NUM_PROCS):

        request_state_img_shm = shared_memory.SharedMemory(create=True, size=state_img.nbytes)
        response_policy_shm = shared_memory.SharedMemory(create=True, size=policy.nbytes)
        wait_for_response_sema = Semaphore(0)


        proc = Process(
            target=process_evaluate, 
            args=(process_id, avg_rewards_shm, sum_rewards_shm, request_state_img_shm, response_policy_shm, wait_for_response_sema, use_policy_network_only)
        )

        proc_list.append(proc)
        request_state_img_shm_list.append(request_state_img_shm)
        response_policy_shm_list.append(response_policy_shm)
        wait_for_response_sema_list.append(wait_for_response_sema)
        proc.start()


    while any([proc.is_alive() for proc in proc_list]):
        
        request_proc_list = []
        data = []

        can_process_at_least_one = False
        for proc_idx in range(EVAL_NUM_PROCS):
   
            wait_for_response_sema = wait_for_response_sema_list[proc_idx]

            if wait_for_response_sema.get_value() == 0:
                
                can_process_at_least_one = True 

                request_state_img_shm = request_state_img_shm_list[proc_idx]
                request_state_img = np.ndarray(shape=STATE_IMG_SHAPE, dtype=STATE_IMG_DTYPE, buffer=request_state_img_shm.buf)

                request_proc_list.append(proc_idx)
                data.append(request_state_img)
          

        if can_process_at_least_one:
            data = np.stack(data, axis=0)
            policy, _ = policyValueNetwork.call_no_tf_func(data)
      
            for data_idx, proc_idx in enumerate(request_proc_list):
                response_policy_shm = response_policy_shm_list[proc_idx]
                response_policy = np.ndarray(shape=(NUM_ACTIONS,), dtype=POLICY_DTYPE, buffer=response_policy_shm.buf)
                response_policy[:] = policy[data_idx]

                wait_for_response_sema = wait_for_response_sema_list[proc_idx]
                wait_for_response_sema.release()

    

    for request_state_img_shm in request_state_img_shm_list:
        request_state_img_shm.close()
        request_state_img_shm.unlink()

    for response_policy_shm in response_policy_shm_list:
        response_policy_shm.close()
        response_policy_shm.unlink()

    # num_steps_mem = np.ndarray(shape=num_steps_mem.shape, dtype=num_steps_mem.dtype, buffer=num_steps_shm.buf)
    # num_steps_copy = np.copy(num_steps_mem)
    # num_steps_copy = np.reshape(num_steps_copy, newshape=(-1,))


    avg_rewards_mem = np.ndarray(shape=(EVAL_NUM_PROCS,), dtype=np.float32, buffer=avg_rewards_shm.buf)
    sum_rewards_mem = np.ndarray(shape=(EVAL_NUM_PROCS,), dtype=np.float32, buffer=sum_rewards_shm.buf)


    avg_rewards_mem_copy = np.copy(avg_rewards_mem)
    sum_rewards_mem_copy = np.copy(sum_rewards_mem)


    avg_rewards_shm.close()
    avg_rewards_shm.unlink()
    
    sum_rewards_shm.close()
    sum_rewards_shm.unlink()

    return avg_rewards_mem_copy, sum_rewards_mem_copy
 