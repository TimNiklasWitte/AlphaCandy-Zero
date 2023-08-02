import tensorflow as tf
import numpy as np
import tqdm

from multiprocessing import Process, Semaphore, shared_memory

from Config import *
from MCTS import *
from CandyCrushGym import *
from StateToImageConverter import *
from CandyAugmentation import *

# from matplotlib import pyplot as plt

class MCTS_Buffer:

    def __init__(self, num_samples, num_actions, state_shape):
        
        self.num_samples = num_samples
        self.num_actions = num_actions

        self.states = np.zeros(shape=(num_samples, *state_shape), dtype=STATE_DTYPE)
        self.states_shm = shared_memory.SharedMemory(create=True, size=self.states.nbytes)

        self.values = np.zeros(shape=(num_samples,), dtype=VALUE_DTYPE)
        self.values_shm = shared_memory.SharedMemory(create=True, size=self.values.nbytes)

        self.policies = np.zeros(shape=(num_samples, num_actions), dtype=POLICY_DTYPE)
        self.policies_shm = shared_memory.SharedMemory(create=True, size=self.policies.nbytes)

        self.candyAugmentation = CandyAugmentation()

    def update_memory(self):

        self.states = np.ndarray(
                        shape=(MCTS_BUFFER_SIZE,*STATE_SHAPE), 
                        dtype=STATE_DTYPE, 
                        buffer=self.states_shm.buf
                    )
    
        self.values = np.ndarray(
                        shape=(MCTS_BUFFER_SIZE,), 
                        dtype=VALUE_DTYPE, 
                        buffer=self.values_shm.buf
                    )
        
        self.policies = np.ndarray(
                        shape=(MCTS_BUFFER_SIZE,), 
                        dtype=POLICY_DTYPE, 
                        buffer=self.policies_shm.buf
                    )

        
        states = self.candyAugmentation(self.states)
        num_permutations = states.shape[0]

        states = np.reshape(states, newshape=(num_permutations * MCTS_BUFFER_SIZE, FIELD_SIZE  + CANDY_BUFF_HEIGHT, FIELD_SIZE))
        self.states = np.concatenate([states, self.states], axis=0)
        
        self.values = np.tile(self.values, num_permutations + 1)
        self.policies = np.tile(self.policies, num_permutations + 1)
        
        
        self.num_samples = (num_permutations + 1) * MCTS_BUFFER_SIZE
        # print(self.states.shape, self.states.nbytes)
        # print(self.values.shape)
        # print(self.actions.shape)

    def dataset_generator(self):
        
        stateToImageConverter = StateToImageConverter(FIELD_SIZE, candy_buff_height=CANDY_BUFF_HEIGHT, image_size=CANDY_IMG_SIZE)

        for i in range(self.num_samples):
            yield stateToImageConverter(self.states[i]), self.values[i], self.policies[i]
    

    def free_shms(self):
        self.states_shm.close()
        self.values_shm.close()
        self.policies_shm.close()

        self.states_shm.unlink()
        self.values_shm.unlink()
        self.policies_shm.unlink()


def prepare_data(dataset):

    #dataset = dataset.map(lambda state, target_value, target_action: (stateToImageConverter(state), target_value, target_action))

    #dataset = tf.py_func(stateToImageConverter)

    #dataset = dataset.map(lambda state, target_value, target_action: (state, target_value, tf.one_hot(target_action, depth=NUM_ACTIONS)))
    
    # Convert data from uint8 to float32
    #dataset = dataset.map(lambda state, target_value, target_action: (tf.cast(state, tf.float32), target_value, target_action))

    # Sloppy input normalization, just bringing image values from range [0, 255] to [-1, 1]
    #dataset = dataset.map(lambda state, target_value, target_action: ((state / 128.) - 1., target_value, target_action))

    #dataset = dataset.cache()
    
    dataset = dataset.shuffle(SHUFFLE_WINDOW_SIZE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset

def process_update_dataset(process_id, seed, states_shm, values_shm, policies_shm, request_state_img_shm, response_policy_shm, response_value_shm, wait_for_response_sema):
    
    reduced_action_space = get_reduced_action_space()
    num_actions = len(reduced_action_space)
    
    states = np.ndarray(shape=(MCTS_BUFFER_SIZE, *STATE_SHAPE), dtype=STATE_DTYPE, buffer=states_shm.buf)
    values = np.ndarray(shape=(MCTS_BUFFER_SIZE,), dtype=VALUE_DTYPE, buffer=values_shm.buf)
    policies = np.ndarray(shape=(MCTS_BUFFER_SIZE, num_actions), dtype=POLICY_DTYPE, buffer=policies_shm.buf)

    iterator = range(iterations_per_process)
    if process_id == NUM_PROCS - 1:
        print("<INFO> Filling MCTS_Buffer")
        iterator = tqdm.tqdm(iterator,position=0, leave=True)
    
    env = CandyCrushGym(seed)
    state = env.reset()

    stateToImageConverter = StateToImageConverter(field_size=FIELD_SIZE, candy_buff_height=CANDY_BUFF_HEIGHT, image_size=CANDY_IMG_SIZE)
    for i in iterator:
        
        mcts = MCTS(env, request_state_img_shm, response_policy_shm, response_value_shm, wait_for_response_sema, stateToImageConverter)

        _, policy, value = mcts.run(NUM_MCTS_STEPS)

        # action = env.action_space.sample()
        # value = 1

        # synchronize write by coordinated access
        idx = NUM_PROCS * i + process_id
        states[idx, :, :] = state 
        values[idx] = value
        policies[idx, :] = policy[:]

    
        idx_reduced_action_space = np.argmax(policy)
        action = reduced_action_space[idx_reduced_action_space]

        state, reward, done, _  = env.step(action)
 
        if done:
            state, _ = env.reset()


def update_dataset(mcts_buffer, policyValueNetwork):

    request_state_img_shm_list = []
    response_policy_shm_list = []
    response_value_shm_list = []
    wait_for_response_sema_list = []
    
    reduced_action_space = get_reduced_action_space()
    num_actions = len(reduced_action_space)

    # prototypical state and action memory
    state_img = np.zeros(shape=STATE_IMG_SHAPE, dtype=STATE_IMG_DTYPE)
    policy = np.zeros(shape=(num_actions,), dtype=POLICY_DTYPE)
    value = np.zeros(shape=(1,), dtype=VALUE_DTYPE)


    proc_list = [] 
    for process_id in range(NUM_PROCS):
        
        request_state_img_shm = shared_memory.SharedMemory(create=True, size=state_img.nbytes)
        response_policy_shm = shared_memory.SharedMemory(create=True, size=policy.nbytes)
        response_value_shm = shared_memory.SharedMemory(create=True, size=value.nbytes)

        wait_for_response_sema = Semaphore(0)

        seed = np.random.randint(low=0, high=9999999)

        proc = Process(
            target=process_update_dataset, 
            args=(process_id, seed, mcts_buffer.states_shm, mcts_buffer.values_shm, mcts_buffer.policies_shm, request_state_img_shm, response_policy_shm, response_value_shm, wait_for_response_sema,)
        )

        proc_list.append(proc)
        request_state_img_shm_list.append(request_state_img_shm)
        response_policy_shm_list.append(response_policy_shm)
        response_value_shm_list.append(response_value_shm)
        wait_for_response_sema_list.append(wait_for_response_sema)
        proc.start()


    # env = CandyCrushGym()
    # invalid_actions = [action for action in range(env.action_space.n) if not env.isValidAction(action)]
    # invalid_actions = np.array(invalid_actions)

    while any([proc.is_alive() for proc in proc_list]):
        
        request_proc_list = []
        data = []

        can_process_at_least_one = False
        for proc_idx in range(NUM_PROCS):
   
            wait_for_response_sema = wait_for_response_sema_list[proc_idx]

            if wait_for_response_sema.get_value() == 0:
                
                can_process_at_least_one = True 

                request_state_img_shm = request_state_img_shm_list[proc_idx]
                request_state_img = np.ndarray(shape=STATE_IMG_SHAPE, dtype=STATE_IMG_DTYPE, buffer=request_state_img_shm.buf)

                request_proc_list.append(proc_idx)
                data.append(request_state_img)

                # plt.imshow(request_state_img)
                # plt.show()
          

        if can_process_at_least_one:
            data = np.stack(data, axis=0)
            policy, value = policyValueNetwork.call_no_tf_func(data)
            #policy = policy.numpy()
         
            #policy[:, invalid_actions] = 0
            #actions = np.argmax(policy, axis=-1)

            for data_idx, proc_idx in enumerate(request_proc_list):
                response_policy_shm = response_policy_shm_list[proc_idx]
                response_policy = np.ndarray(shape=(num_actions,), dtype=POLICY_DTYPE, buffer=response_policy_shm.buf)
                response_policy[:] = policy[data_idx, :]

                response_value_shm = response_value_shm_list[proc_idx]
                response_value = np.ndarray(shape=(1,), dtype=VALUE_DTYPE, buffer=response_value_shm.buf)
                response_value[:] = value[data_idx]

                wait_for_response_sema = wait_for_response_sema_list[proc_idx]
                wait_for_response_sema.release()

    for request_state_img_shm in request_state_img_shm_list:
        request_state_img_shm.close()
        request_state_img_shm.unlink()

    for response_policy_shm in response_policy_shm_list:
        response_policy_shm.close()
        response_policy_shm.unlink()

    for response_value_shm in response_value_shm_list:
        response_value_shm.close()
        response_value_shm.unlink()

    mcts_buffer.update_memory()