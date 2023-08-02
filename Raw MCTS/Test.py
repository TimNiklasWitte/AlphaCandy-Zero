import sys
sys.path.append("../")
from CandyCrushGym import *
from CandyCrushUtiles import *

from Raw_MCTS import *



env = CandyCrushGym()

state = env.reset()

mcts = Raw_MCTS(env)

mcts.run(50)
#FIELD_SIZE = 3

#top_actions = [FIELD_SIZE * 4 + (FIELD_SIZE + 1) * i for i in range((FIELD_SIZE * FIELD_SIZE) - FIELD_SIZE)]
#print(top_actions)
#num_top_actions = (env.FIELD_SIZE * env.FIELD_SIZE) - env.FIELD_SIZE

#top_actions = [env.FIELD_SIZE * env.NUM_DIRECTIONS + env.NUM_DIRECTIONS * i for i in range(num_top_actions)]

# print(top_actions)

# for action in top_actions:
#     print(action, get_x_y_direction(action), isValidAction(action))

# action = 32
# print(get_x_y_direction(action))
# print(isValidAction(action))

# action_space = []
# for action in range(env.action_space.n):
#     if isValidAction(action):

#         x_y_direction = get_x_y_direction(action)

#         if "top" in x_y_direction:
#             action_space.append(action)
            
#         elif "right" in x_y_direction:
#             action_space.append(action)
    

