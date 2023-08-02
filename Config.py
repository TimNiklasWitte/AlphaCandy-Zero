#
# Training
#

SHUFFLE_WINDOW_SIZE = 5000
BATCH_SIZE = 64

#
# Create dataset
#

MCTS_BUFFER_SIZE = 700
NUM_PROCS = 100
iterations_per_process = int(MCTS_BUFFER_SIZE / NUM_PROCS) # // better?

NUM_MCTS_STEPS = 150
NUM_EPOCHS = 500

NUM_TRAIN_LOOPS = 5

# Train: 90 % 
# Test: 10 % 
TEST_DATASET_SIZE_PERCENTAGE = 10 



#
# Evaluation
#
EVAL_NUM_PROCS = 32
EVAL_NUM_STEPS_PER_PROC = 10


#
# Data Augmentation
#

AUG_NUM_PROCS = 32

#
# State shape
#
from CandyCrushGym import *
env = CandyCrushGym()
state = env.reset()

STATE_SHAPE = state.shape
STATE_DTYPE = state.dtype

NUM_ACTIONS = env.action_space.n
import numpy as np


#ACTION_DTYPE = np.float32


CANDY_BUFF_HEIGHT = 5
CANDY_IMG_SIZE = 10

STATE_IMG_SHAPE = ((env.FIELD_SIZE + CANDY_BUFF_HEIGHT)* CANDY_IMG_SIZE, env.FIELD_SIZE * CANDY_IMG_SIZE, 3)
STATE_IMG_DTYPE = np.float32


VALUE_DTYPE = np.float32
POLICY_SHAPE = (NUM_ACTIONS, )
POLICY_DTYPE = np.float32
ACTION_DTYPE = np.uint8


logs_file_path = "test_logs/test"