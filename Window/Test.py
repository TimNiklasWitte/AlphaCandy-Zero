import sys
sys.path.append("../")
from CandyCrushGym import *
from StateToImageConverter import *
from matplotlib import pyplot as plt

env = CandyCrushGym()
state = env.reset()

# env.state = np.array(
# [[4,2,4,2,1,6,1,6],
#  [1,5,4,6,6,1,1,3],
#  [2,3,6,6,3,6,1,1],
#  [5,3,1,1,3,4,1,6],
#  [3,2,4,3,1,5,1,3],
#  [5,6,3,5,3,4,3,6],
#  [3,5,5,2,4,3,3,3],
#  [3,6,2,4,1,5,6,4],
#  [6,1,5,2,1,4,3,4],
#  [5,5,1,2,1,1,4,5],
#  [1,3,6,3,1,1,2,3],
#  [6,3,6,2,1,1,3,3],
#  [5,5,6,4,4,3,1,3]]
# )

env.state = np.array(
[[ 4, 2, 2, 2, 2, 4, 2, 3],
 [ 5, 1, 4, 3, 1, 6, 2, 6],
 [ 4, 4, 2, 3, 5, 2, 3, 5],
 [ 5, 2, 3, 4, 2, 1, 6, 1],
 [ 5, 5, 2, 2, 5, 4, 4, 2],
 [ 3, 1, 1, 3, 4, 1, 3, 4],
 [ 6, 5, 3, 1, 6, 2, 4, 1],
 [ 5, 4, 4, 4,12, 1, 2, 4],
 [ 3, 3, 3, 5, 2, 3, 1, 4],
 [ 3, 3, 3, 1, 2, 2, 3, 1],
 [ 6, 2, 2, 5, 3, 5, 5, 5],
 [ 2, 4,25, 2, 1, 2, 3, 5],
 [ 6, 5, 2, 5, 4, 1, 5, 1]]
)


stateToImageConverter = StateToImageConverter(env.FIELD_SIZE, env.CANDY_BUFF_HEIGHT, 60)

state_img = stateToImageConverter(env.state)
plt.imshow(state_img)
plt.savefig("test1.png")

env.step(48)

state_img = stateToImageConverter(env.state)
plt.imshow(state_img)
plt.savefig("test2.png")
