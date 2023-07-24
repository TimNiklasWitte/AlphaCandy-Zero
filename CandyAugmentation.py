import numpy as np
import itertools

from multiprocessing import Process, shared_memory
from Config import *
from StateToImageConverter import *

from matplotlib import pyplot as plt

class CandyAugmentation:

    def __init__(self):
        
        self.colors_ids = np.array(range(1,7))

    
    def __call__(self, states):
        permutations = list(itertools.permutations(self.colors_ids))

        valid_permutations = []

        for permutation in permutations:

            valid = True 
            for idx, color_id in enumerate(permutation):

                if idx + 1 == color_id:
                    valid = False
                    break 
            
            if valid:
                valid_permutations.append(permutation)

        permutations = valid_permutations
       
        

        num_permutations = len(permutations)

        num_permutations = int(num_permutations / 2)
        augmentated_states = np.zeros(shape=(num_permutations, MCTS_BUFFER_SIZE, *STATE_SHAPE), dtype=STATE_DTYPE)
        augmentated_states_shm = shared_memory.SharedMemory(create=True, size=augmentated_states.nbytes)

        num_launched_processes = 0

        while num_launched_processes != num_permutations:
            
            bound = num_launched_processes + AUG_NUM_PROCS
            if bound <= num_permutations:
                bound = num_permutations

            proc_list = [] 
            for process_id in range(num_launched_processes, bound):
                
                candy_swap_table = np.array(permutations[process_id])
         
                proc = Process(
                    target=self.process_augmentation, 
                    args=(process_id, num_permutations, states, candy_swap_table, augmentated_states_shm,)
                )
                
                proc_list.append(proc)

                proc.start()
            
            for proc in proc_list:
                proc.join()

            num_launched_processes += bound
        

        augmentated_states = np.ndarray(shape=(num_permutations, MCTS_BUFFER_SIZE, *STATE_SHAPE), dtype=STATE_DTYPE, buffer=augmentated_states_shm.buf)

        augmentated_states_copy = np.copy(augmentated_states)

        augmentated_states_shm.close()
        augmentated_states_shm.unlink()

        return augmentated_states_copy

    
    def process_augmentation(self, process_id, num_permutations, states, candy_swap_table, augmentated_states_shm):
        
        augmentated_states = np.ndarray(shape=(num_permutations, MCTS_BUFFER_SIZE, *STATE_SHAPE), dtype=STATE_DTYPE, buffer=augmentated_states_shm.buf)
        

        for idx in range(MCTS_BUFFER_SIZE):

            for x in range(FIELD_SIZE):
                for y in range(FIELD_SIZE + CANDY_BUFF_HEIGHT):

                    candy_id = states[idx, y, x]

                    if isNormalCandy(candy_id):
                        candy_id = candy_swap_table[candy_id - 1]
                        augmentated_states[process_id, idx, y, x] = candy_id
                        
                        continue

                    if isWrappedCandyID(candy_id):
                        candy_id = convertWrappedCandy_toNormal(candy_id)
                        candy_id = candy_swap_table[candy_id - 1]
                        candy_id = getWrappedCandyID(candy_id)
                        augmentated_states[process_id, idx, y, x] = candy_id
                        continue
                    
                    if isHorizontalStrippedCandy(candy_id):
                        candy_id = convertHorizontalStrippedCandy_toNormal(candy_id)
                        candy_id = candy_swap_table[candy_id - 1]
                        candy_id = getHorizontalStrippedCandyID(candy_id)
                        augmentated_states[process_id, idx, y, x] = candy_id
                        continue
                    
                    if isVerticalStrippedCandy(candy_id):
                        candy_id = convertVerticalStrippedCandy_toNormal(candy_id)
                        candy_id = candy_swap_table[candy_id - 1]
                        candy_id = getVerticalStrippedCandyID(candy_id)
                        augmentated_states[process_id, idx, y, x] = candy_id
                        continue

                    if candy_id == COLOR_BOMB_CANDY_ID:
                        continue
                    
                    #print(augmentated_states[process_id, idx, y, x], process_id, idx, y, x)

        #print(process_id, candy_swap_table, augmentated_states.shape)
        #print(states.shape)

# states = np.random.randint(1, 7, size=(MCTS_BUFFER_SIZE,*STATE_SHAPE))

# candyAugmentation = CandyAugmentation()
# augmentated_states = candyAugmentation(states)

# stateToImageConverter = StateToImageConverter(FIELD_SIZE, candy_buff_height=CANDY_BUFF_HEIGHT, image_size=CANDY_IMG_SIZE)

# x = stateToImageConverter(states[0])

# plt.imshow(x)
# print(x.shape)
# plt.savefig("1.png")

# y = stateToImageConverter(augmentated_states[0][0])

# plt.clf()
# plt.imshow(y)
# plt.savefig("2.png")
# print(states[0].shape)
# print(augmentated_states[0][0].shape)








            